# Explainable Deep Learning for Diabetic Retinopathy Detection

This repository contains a deep learning pipeline for detecting and grading diabetic retinopathy from retinal fundus images. The project compares transfer-learning architectures with a custom convolutional neural network, evaluates retinal-image preprocessing methods, and applies Grad-CAM to explain model predictions.

**Evaluation update (2026-09-18):** the content-level audit found 47 exact duplicate pairs across the supplied splits; 24 test images match training or validation images. Some exact copies also have conflicting labels. Earlier scores are preliminary internal results, not a clean final evaluation. See [the audit and error-analysis report](notes/FINAL_BINARY_REVIEW.md) and checkpoint DR-CP-016 before citing them.

The separate [corrected binary experiment](notes/CLEAN_BINARY_EXPERIMENT.md) uses exact deduplication, conflicting-label quarantine, and conservative similarity groups. Its frozen split has 2,822 training, 354 validation, and 354 test images. Historical model rankings do not automatically apply to this repartition.

The corrected EfficientNet-B0 run is complete: accuracy **98.31%**, F1 **98.26%**, recall **97.13%**, and ROC-AUC **0.9952**, with five false negatives and one false positive. These are corrected internal results, not external or patient-verified validation.

The implementation supports both:

- **binary classification** for diabetic retinopathy screening;
- **multi-class classification** using the original five severity classes.

## Project Objectives

The main objectives of the project are to:

- train and compare `ResNet18`, `DenseNet121`, and `EfficientNet-B0`;
- compare transfer learning with a custom CNN trained from scratch;
- evaluate the effect of retinal-image preprocessing;
- address class imbalance in multi-class classification;
- analyze model performance using classification and error-analysis metrics;
- visualize image regions that influence predictions using `Grad-CAM`;
- investigate whether the models rely on clinically relevant retinal regions or shortcut features such as borders, shadows, and the optic disc.

## Dataset

The project is configured to work with the **APTOS 2019 diabetic retinopathy dataset**.

The data loader supports the standard APTOS columns:

- `id_code` — image identifier;
- `diagnosis` — diabetic retinopathy class.

It also supports predefined training, validation, and test CSV files.

For custom datasets, the project can read either of the following CSV formats:

```text
image_id,label
img_001,0
img_002,1
```

or:

```text
image_path,label
class0/img_001.png,0
class1/img_002.png,1
```

## Project Structure

```text
Detection of Diabetic Retinopathy/
├── .vscode/
├── configs/
│   ├── default.yaml
│   ├── densenet121_baseline.yaml
│   ├── efficientnet_b0_baseline.yaml
│   ├── efficientnet_b0_multiclass.yaml
│   ├── efficientnet_b0_multiclass_crop.yaml
│   ├── efficientnet_b0_multiclass_crop_clahe.yaml
│   ├── resnet18_preprocessed.yaml
│   └── custom_cnn_scratch.yaml
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   ├── comparison/
│   ├── custom_cnn/
│   ├── multiclass/
│   └── transfer_learning/
├── outputs/
│   ├── comparison/
│   ├── comparison_all/
│   ├── custom_cnn/
│   │   └── gradcam/
│   ├── figures/
│   └── gradcam/
├── scripts/
│   ├── download_aptos.py
│   ├── check_dataset.py
│   └── train_baseline.py
├── src/
│   └── dr_detection/
│       ├── __init__.py
│       ├── config.py
│       ├── dataset.py
│       ├── explainability.py
│       └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

`Python 3.11` is recommended for compatibility with deep learning libraries such as PyTorch and OpenCV.

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Preparation

### Automatic Download

After installing the dependencies, download the APTOS 2019 dataset with:

```bash
python3 scripts/download_aptos.py
```

For interactive Kaggle authentication, use:

```bash
python3 scripts/download_aptos.py --login
```

The script downloads the `mariaherrerot/aptos2019` dataset to:

```text
data/raw/aptos2019/
```

It also reports the detected locations of the CSV files and image directories.

### Manual Dataset Placement

The expected APTOS directory structure is:

```text
data/raw/aptos2019/
├── train_1.csv
├── valid.csv
├── test.csv
├── train_images/
│   └── train_images/
│       ├── 000c1434d8d7.png
│       ├── 001639a390f0.png
│       └── ...
├── val_images/
│   └── val_images/
└── test_images/
    └── test_images/
```

Update the dataset paths in the relevant YAML configuration file when the local directory structure is different.

## Dataset Validation

Before training, verify that the CSV file and image directory are correctly configured:

```bash
python3 scripts/check_dataset.py \
  --csv data/raw/aptos2019/train_1.csv \
  --images-dir data/raw/aptos2019/train_images/train_images \
  --binary
```

The validation script checks:

- the number of records in the CSV file;
- the detected number of classes;
- missing image files;
- whether an image sample can be loaded successfully.

## Running Experiments

All experiments are controlled through YAML configuration files.

The general training command is:

```bash
python3 scripts/train_baseline.py --config <config-path>
```

### Available Configurations

| Configuration | Experiment |
|---|---|
| `configs/default.yaml` | ResNet18 baseline |
| `configs/densenet121_baseline.yaml` | DenseNet121 transfer-learning baseline |
| `configs/efficientnet_b0_baseline.yaml` | EfficientNet-B0 transfer-learning baseline |
| `configs/resnet18_preprocessed.yaml` | ResNet18 with retinal-image preprocessing |
| `configs/custom_cnn_scratch.yaml` | Custom CNN trained from scratch |
| `configs/efficientnet_b0_multiclass.yaml` | EfficientNet-B0 five-class baseline |
| `configs/efficientnet_b0_multiclass_crop.yaml` | Five-class EfficientNet-B0 with retinal-focused cropping |
| `configs/efficientnet_b0_multiclass_crop_clahe.yaml` | Five-class EfficientNet-B0 with cropping and CLAHE |

### Baseline Models

Run the ResNet18 baseline:

```bash
python3 scripts/train_baseline.py --config configs/default.yaml
```

Run the additional transfer-learning models:

```bash
python3 scripts/train_baseline.py --config configs/densenet121_baseline.yaml
python3 scripts/train_baseline.py --config configs/efficientnet_b0_baseline.yaml
```

These experiments compare three established convolutional architectures under a consistent experimental pipeline:

- `ResNet18` — a compact baseline architecture;
- `DenseNet121` — a densely connected model with extensive feature reuse;
- `EfficientNet-B0` — an efficient architecture that balances model depth, width, and input resolution.

### Preprocessing Experiment

Run the preprocessing-enhanced ResNet18 experiment:

```bash
python3 scripts/train_baseline.py --config configs/resnet18_preprocessed.yaml
```

The preprocessing pipeline includes:

- removal of dark image borders;
- retinal-field cropping;
- conversion to a square image region;
- CLAHE-based contrast enhancement.

This experiment is designed to compare the original baseline with a preprocessing-enhanced version while keeping the baseline configuration unchanged.

### Custom CNN From Scratch

Run the custom CNN experiment:

```bash
python3 scripts/train_baseline.py --config configs/custom_cnn_scratch.yaml
```

This configuration uses:

```text
model_name: custom_cnn
pretrained: false
```

Its model weights and generated outputs are stored separately:

```text
models/custom_cnn_scratch/
outputs/custom_cnn/
```

### Multi-Class Classification

Run the five-class EfficientNet-B0 baseline:

```bash
python3 scripts/train_baseline.py --config configs/efficientnet_b0_multiclass.yaml
```

The multi-class configuration:

- preserves the original labels `0`, `1`, `2`, `3`, and `4`;
- sets `binary_classification: false`;
- uses balanced class weights to reduce the effect of class imbalance;
- uses macro F1-score as the main validation metric for best-epoch selection.

Run the preprocessing variants:

```bash
python3 scripts/train_baseline.py --config configs/efficientnet_b0_multiclass_crop.yaml
python3 scripts/train_baseline.py --config configs/efficientnet_b0_multiclass_crop_clahe.yaml
```

The comparison follows a controlled sequence:

1. EfficientNet-B0 without additional preprocessing;
2. EfficientNet-B0 with retinal-focused cropping;
3. EfficientNet-B0 with retinal-focused cropping and CLAHE.

This design makes it possible to evaluate whether preprocessing improves overall classification and the recognition of rare severe classes.

## Model Evaluation

Depending on the experiment, the analysis pipeline produces:

- accuracy;
- precision;
- recall;
- F1-score;
- macro F1-score;
- per-class precision, recall, F1-score, and support;
- confusion matrices;
- row-normalized confusion matrices;
- ROC curves;
- true-positive, true-negative, false-positive, and false-negative counts;
- comparisons of false positives and false negatives;
- rare-class recall analysis;
- prediction-change summaries.

For diabetic retinopathy screening, particular attention is given to **recall** and **false negatives**, because missed positive cases are important when interpreting screening performance.

## Explainability With Grad-CAM

Grad-CAM is used to visualize the image regions that contribute to model predictions.

The explainability analysis includes examples from four prediction categories:

- true positives;
- true negatives;
- false positives;
- false negatives.

This analysis is used to examine whether model attention is concentrated on retinal structures or on potentially misleading image characteristics such as borders, shadows, and the optic disc region.

Grad-CAM outputs for EfficientNet-B0 are stored in:

```text
outputs/efficientnet_b0_baseline/gradcam/
```

Grad-CAM outputs for the custom CNN are stored in:

```text
outputs/custom_cnn/gradcam/
```

## Analysis Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/04_model_comparison.ipynb` | Comparison of baseline and preprocessing-enhanced ResNet18 |
| `notebooks/custom_cnn/05_custom_cnn_results.ipynb` | Evaluation of the custom CNN trained from scratch |
| `notebooks/comparison/06_all_model_comparison.ipynb` | Overall comparison of custom CNN, ResNet18, DenseNet121, and EfficientNet-B0 |
| `notebooks/custom_cnn/07_custom_cnn_gradcam.ipynb` | Grad-CAM analysis for the custom CNN |
| `notebooks/transfer_learning/08_densenet121_results.ipynb` | DenseNet121 result analysis |
| `notebooks/transfer_learning/09_efficientnet_b0_results.ipynb` | EfficientNet-B0 result analysis |
| `notebooks/transfer_learning/10_efficientnet_b0_gradcam.ipynb` | Grad-CAM analysis for EfficientNet-B0 |
| `notebooks/multiclass/11_efficientnet_b0_multiclass_results.ipynb` | Five-class EfficientNet-B0 evaluation |
| `notebooks/multiclass/12_multiclass_preprocessing_comparison.ipynb` | Comparison of multi-class preprocessing strategies |

## Generated Tables and Figures

The notebooks generate experiment-specific tables and visualizations, including:

- summary tables of test metrics;
- `TP/TN/FP/FN` tables;
- per-class performance tables;
- confusion matrices;
- normalized confusion matrices;
- ROC curves;
- model-comparison charts;
- false-positive and false-negative comparisons;
- rare-class recall comparisons;
- Grad-CAM visualizations.

Generated files are organized under the `outputs/` directory to keep results from different experiment groups separate.

## Recommended Workflow

1. Install the project dependencies.
2. Download or manually add the APTOS 2019 dataset.
3. Validate the dataset paths and image loading.
4. Run the ResNet18 baseline.
5. Train DenseNet121 and EfficientNet-B0 under the same setup.
6. Compare the baseline models using the analysis notebooks.
7. Run the preprocessing-enhanced ResNet18 experiment.
8. Train and evaluate the custom CNN.
9. Perform Grad-CAM analysis on selected models.
10. Run the five-class EfficientNet-B0 experiments.
11. Compare cropping and CLAHE preprocessing strategies.

## Repository Scope

This repository is intended for academic experimentation with diabetic retinopathy classification and explainable deep learning. The current pipeline focuses on model comparison, preprocessing analysis, error analysis, and Grad-CAM-based interpretation of retinal fundus image classifiers.

## Integrity Audit and Final Demonstration

Run these notebooks in order with the project `.venv` kernel:

1. [18: dataset integrity audit](notebooks/validation/18_dataset_integrity_audit.ipynb) scans all images and exports duplicate and conflicting-label evidence.
2. [19: complete EfficientNet-B0 error analysis](notebooks/transfer_learning/19_efficientnet_b0_error_analysis.ipynb) reproduces the historical scores and explains every error with maps for both classes.
3. [20: single-image demonstration](notebooks/transfer_learning/20_single_image_demo.ipynb) takes an editable image path and exports predictions and Grad-CAM.

The equivalent commands are:

```bash
.venv/bin/python3 scripts/audit_dataset.py
.venv/bin/python3 scripts/analyze_final_errors.py
.venv/bin/python3 scripts/predict_image.py --image data/raw/aptos2019/test_images/test_images/e4dcca36ceb4.png
```

The audit reads the original data without modifying it. Inference requires the existing `models/efficientnet_b0_baseline/best_model.pt`; model weights and dataset images are excluded from Git. No model training or pretrained-weight download is needed for these three steps. Output folders are `outputs/data_audit`, `outputs/final_error_analysis`, and `outputs/single_image_demo`.

The single-image notebook uses the first test image when `image_path = None`. Set `image_path` to an absolute path or a path relative to this project for another image. Re-running it replaces its demonstration outputs. Scores are uncalibrated, and the prototype does not reject non-fundus or low-quality inputs.

Focused verification:

```bash
.venv/bin/python3 -m unittest discover -s tests -v
```

## Corrected Binary Experiment

- [21: corrected split verification](notebooks/validation/21_clean_binary_split.ipynb) documents exclusions, frozen CSVs, and independent overlap checks.
- [22: corrected EfficientNet-B0 results](notebooks/transfer_learning/22_efficientnet_b0_clean_binary_results.ipynb) reads the validation-selected model's test results, ROC, confusion matrix, learning curves, and error explanations.

```bash
.venv/bin/python3 scripts/prepare_clean_split.py --verify-only
.venv/bin/python3 scripts/train_baseline.py --config configs/efficientnet_b0_clean_binary.yaml
.venv/bin/python3 scripts/evaluate_clean_binary.py
```

The saved split CSVs reference the original image files; no image duplication is required. Split preparation and training refuse to overwrite an existing frozen split or model. Test inference is deferred until training has finished, and later evaluation runs reuse the saved results. The [experimental record](notes/CLEAN_BINARY_EXPERIMENT.md) explains the limitations of similarity grouping and internal evaluation.

`scripts/predict_image.py` accepts `--config configs/efficientnet_b0_clean_binary.yaml` to use the corrected checkpoint. Use a separate `--output-dir`, such as `outputs/clean_binary_demo`, to retain the historical demonstration outputs.
