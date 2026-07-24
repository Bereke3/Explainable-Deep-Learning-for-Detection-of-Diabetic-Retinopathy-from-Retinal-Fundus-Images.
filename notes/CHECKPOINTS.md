# Project Checkpoints

## DR-CP-001

- Date: 2026-07-09
- Status: Completed
- Theme: Project setup and dataset verification

### What I did

I created the initial project structure for the thesis project and set up the Python environment in `.venv`.
I added the base dependencies for deep learning, notebooks, and Grad-CAM.
I prepared helper scripts for dataset checking and baseline training.
I added a download script for APTOS 2019, then loaded the dataset manually into the project.
I verified that the dataset files are readable and that the image paths are valid.
I also created the first notebook, `notebooks/01_dataset_check.ipynb`, and ran it successfully without errors.

### Technical decisions I fixed at this stage

- Main thesis baseline task: binary classification
- Labels:
  - `0` = No Diabetic Retinopathy
  - `1` = Diabetic Retinopathy Present
- Baseline model for the first experiment: `ResNet18`
- Workflow:
  - notebooks for exploration, figures, and visual analysis
  - scripts/modules for reproducible training code

### Dataset status

- Train CSV: `data/raw/aptos2019/train_1.csv`
- Validation CSV: `data/raw/aptos2019/valid.csv`
- Test CSV: `data/raw/aptos2019/test.csv`
- Train images: `data/raw/aptos2019/train_images/train_images`
- Validation images: `data/raw/aptos2019/val_images/val_images`
- Test images: `data/raw/aptos2019/test_images/test_images`

The dataset was connected correctly, images were readable, and the class balance looked acceptable for a first binary baseline.

### Why this checkpoint mattered

At this point, I confirmed that the project had a working technical base.
The dataset was connected and validated.
The notebook workflow in VS Code was also confirmed, so I was ready to move from setup to the first actual model training.


## DR-CP-002

- Date: 2026-07-09
- Status: Completed
- Theme: First baseline training with ResNet18

### What I did

I trained the first baseline model for diabetic retinopathy detection using `ResNet18` on the APTOS 2019 dataset in binary classification mode.
I used the predefined train/validation/test split files instead of randomly splitting the dataset again.
The training finished successfully, the best model was saved, and I obtained the first full set of baseline metrics on the test set.

### Training setup I used

- Task: binary classification
- Labels:
  - `0` = No Diabetic Retinopathy
  - `1` = Diabetic Retinopathy Present
- Model: `ResNet18`
- Pretrained weights: `true`
- Config file: `configs/default.yaml`
- Best model output: `models/best_model.pt`

### Dataset split I used

- Train size: `2930`
- Validation size: `366`
- Test size: `366`

### Input preprocessing and augmentation

For all images, I resized the input to `224 x 224`.

For training images, I used:
- `RandomHorizontalFlip(p=0.5)`
- `RandomRotation(degrees=10)`
- `ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02)`
- normalization with ImageNet mean and standard deviation

For validation and test images, I used:
- resize to `224 x 224`
- conversion to tensor
- normalization with ImageNet mean and standard deviation

### Optimization settings

- Loss function: `CrossEntropyLoss`
- Optimizer: `Adam`
- Learning rate: `0.0003`
- Batch size: `16`
- Epochs: `5`
- Number of workers: `2`
- Seed: `42`

### Download detail during training

During the first full training run, the pretrained `ResNet18` weights were downloaded successfully:

- `44.7M / 44.7M`

This confirms that the pretrained backbone was actually used in the baseline experiment.

### Epoch-by-epoch results

- Epoch 1:
  - `train_loss = 0.1863`
  - `train_f1 = 0.9365`
  - `val_loss = 0.1086`
  - `val_f1 = 0.9749`
- Epoch 2:
  - `train_loss = 0.1079`
  - `train_f1 = 0.9638`
  - `val_loss = 0.0523`
  - `val_f1 = 0.9896`
- Epoch 3:
  - `train_loss = 0.1298`
  - `train_f1 = 0.9559`
  - `val_loss = 0.0562`
  - `val_f1 = 0.9898`
- Epoch 4:
  - `train_loss = 0.0847`
  - `train_f1 = 0.9728`
  - `val_loss = 0.0606`
  - `val_f1 = 0.9845`
- Epoch 5:
  - `train_loss = 0.0702`
  - `train_f1 = 0.9790`
  - `val_loss = 0.0525`
  - `val_f1 = 0.9844`

### Final test metrics

- Accuracy: `0.9781420765027322`
- Precision: `0.9649122807017544`
- Recall: `0.9880239520958084`
- F1-score: `0.9763313609467456`
- ROC-AUC: `0.9973520296091235`
- Test loss: `0.09183882669510086`

### My interpretation of the result

This is a very strong result for the first baseline.
The recall is especially important because the task is medical screening and missing DR-positive cases is risky.
The ROC-AUC is also extremely high, which suggests that the model separates the two classes very well.

At the same time, because the result is so strong even in the baseline stage, I need to be careful and do sanity checks next.
I should verify that there is no data leakage, inspect confusion matrix results, and later check Grad-CAM visualizations to make sure the model is focusing on clinically meaningful retinal regions.

### What this checkpoint proves

This checkpoint proves that:

- the dataset pipeline works end-to-end;
- the baseline model training works end-to-end;
- the predefined splits are usable;
- the first baseline already gives a strong quantitative result;
- the project is ready to move from baseline training to result analysis and explainability.

### Next step

My next step is to analyze the baseline result more carefully:

- create a confusion matrix;
- plot ROC-related visualizations;
- review failure cases;
- move to Grad-CAM for explainability.

### Checkpoint note for myself

I should create checkpoints only after major project milestones, not after every single figure, chart, or image.
Visualizations such as confusion matrix, ROC curve, example images, or Grad-CAM outputs belong inside the current research stage unless they mark a clearly new phase of the work.
So in this project, checkpoints should represent larger progress blocks such as:

- dataset setup and validation;
- first baseline training;
- baseline result analysis;
- Grad-CAM explainability stage;
- model comparison or final thesis-ready evaluation.


## DR-CP-003

- Date: 2026-07-09
- Status: Completed
- Theme: Grad-CAM explainability analysis of the baseline model

### What I did

I ran the notebook `notebooks/03_gradcam_analysis.ipynb` and completed the first explainability stage for the baseline `ResNet18` model.
In this stage, I visualized Grad-CAM heatmaps for four prediction outcome groups:

- true positives;
- true negatives;
- false positives;
- false negatives.

This allowed me to move beyond numerical performance and inspect where the model actually focuses when making decisions on retinal fundus images.

### Files produced in this stage

- `outputs/gradcam/table_3_1_outcome_summary.csv`
- `outputs/gradcam/figure_3_1_outcome_distribution.png`
- `outputs/gradcam/image_3_1_true_positive_gradcam.png`
- `outputs/gradcam/image_3_2_true_negative_gradcam.png`
- `outputs/gradcam/image_3_3_false_positive_gradcam.png`
- `outputs/gradcam/image_3_4_false_negative_gradcam.png`

### Outcome summary on the test split

- True Negative: `193`
- True Positive: `165`
- False Positive: `6`
- False Negative: `2`

These counts are consistent with the strong baseline metrics obtained earlier and confirm that the model makes only a small number of test-time mistakes.

### What I observed from the Grad-CAM visualizations

In the true positive examples, the model often placed strong attention on regions that visually contain suspicious lesions or abnormal retinal structures.
This is a good sign because it suggests that the classifier is not completely random in its decision behavior.

At the same time, the Grad-CAM maps also revealed an important weakness:

- in several true negative examples, the model focused strongly on peripheral image regions and dark border areas;
- in some false positive and false negative examples, the highlighted regions were not tightly concentrated around clinically meaningful lesions;
- the optic disc and peripheral retinal boundary appeared repeatedly as high-attention areas.

### My interpretation of this stage

This explainability stage was very important because it showed that the model is strong, but not yet fully trustworthy from a thesis-quality medical AI perspective.
Even though the classification metrics are excellent, the model may still be using shortcut cues related to image framing, brightness, border artifacts, or acquisition style.

This is exactly the type of finding that makes the project more serious academically:

- I am not only reporting high scores;
- I am checking whether the model reasons in a plausible way;
- I am using explainability to motivate a better experimental design.

### What this checkpoint changes in the project direction

Because of the Grad-CAM findings, the next improvement should not only be “try a larger network.”
A more justified next step is to improve the image preprocessing pipeline first, so that the model is less likely to rely on non-clinical cues.

For that reason, I prepared the next experiment around:

- cropping dark image borders;
- centering the retinal field more consistently;
- enhancing local contrast with CLAHE;
- retraining `ResNet18` with the same split and same evaluation protocol.

### Why this checkpoint matters

This checkpoint marks the transition from a simple working baseline to a more research-oriented pipeline.
The project now includes not only:

- dataset preparation;
- training;
- metric evaluation;

but also:

- explainability analysis;
- qualitative error inspection;
- evidence-based motivation for the next model improvement.

### Next step

My next step is to train an improved `ResNet18` pipeline using preprocessing-aware input preparation and then compare it directly against the original baseline in both quantitative metrics and qualitative Grad-CAM behavior.


## DR-CP-004

- Date: 2026-07-09
- Status: Completed
- Theme: Preprocessing-enhanced ResNet18 experiment and model comparison

### What I did

I trained a second `ResNet18` model on the same binary diabetic retinopathy task, but this time I added a preprocessing stage before the standard resize and normalization pipeline.
The purpose of this experiment was not to change the backbone architecture, but to test whether better retinal image preparation could reduce shortcut behavior observed during the Grad-CAM analysis.

I kept the same dataset split and the same core training settings as in the original baseline so that the comparison would remain fair and interpretable.

### Experimental setup

- Task: binary classification
- Labels:
  - `0` = No Diabetic Retinopathy
  - `1` = Diabetic Retinopathy Present
- Model: `ResNet18`
- Config file: `configs/resnet18_preprocessed.yaml`
- Best model output: `models/resnet18_preprocessed/best_model.pt`
- Split sizes:
  - train: `2930`
  - validation: `366`
  - test: `366`

### What changed compared with the baseline

I kept the same:

- pretrained backbone;
- batch size;
- learning rate;
- number of epochs;
- optimizer;
- train/validation/test split.

I changed only the image preprocessing stage.

The new preprocessing pipeline included:

- dark border cropping;
- center square cropping;
- CLAHE-based local contrast enhancement.

This design choice was directly motivated by the Grad-CAM stage, where the baseline sometimes focused on border regions, peripheral dark areas, and non-lesion visual cues.

### Optimization settings

- Loss function: `CrossEntropyLoss`
- Optimizer: `Adam`
- Learning rate: `0.0003`
- Batch size: `16`
- Epochs: `5`
- Number of workers: `2`
- Seed: `42`

### Epoch-by-epoch results

- Epoch 1:
  - `train_loss = 0.1826`
  - `train_f1 = 0.9383`
  - `val_loss = 0.1303`
  - `val_f1 = 0.9626`
- Epoch 2:
  - `train_loss = 0.1136`
  - `train_f1 = 0.9625`
  - `val_loss = 0.1259`
  - `val_f1 = 0.9618`
- Epoch 3:
  - `train_loss = 0.0892`
  - `train_f1 = 0.9710`
  - `val_loss = 0.0519`
  - `val_f1 = 0.9897`
- Epoch 4:
  - `train_loss = 0.0773`
  - `train_f1 = 0.9755`
  - `val_loss = 0.0431`
  - `val_f1 = 0.9896`
- Epoch 5:
  - `train_loss = 0.0566`
  - `train_f1 = 0.9823`
  - `val_loss = 0.1474`
  - `val_f1 = 0.9600`

### Best-checkpoint note

Although training ran for `5` epochs, the final test evaluation was computed from the best validation checkpoint, not from the last epoch.
The best validation F1-score in this run was reached at epoch `3` with `val_f1 = 0.9897`.

### Final test metrics for the preprocessing-enhanced model

- Accuracy: `0.9808743169398907`
- Precision: `0.9819277108433735`
- Recall: `0.9760479041916168`
- F1-score: `0.978978978978979`
- ROC-AUC: `0.9971113050281347`
- True Negative: `196`
- False Positive: `3`
- False Negative: `4`
- True Positive: `163`

### Direct comparison with the original baseline

Original baseline result:

- Accuracy: `0.9781420765027322`
- Precision: `0.9649122807017544`
- Recall: `0.9880239520958084`
- F1-score: `0.9763313609467456`
- ROC-AUC: `0.9973520296091235`
- True Negative: `193`
- False Positive: `6`
- False Negative: `2`
- True Positive: `165`

Preprocessing-enhanced result:

- Accuracy: `0.9808743169398907`
- Precision: `0.9819277108433735`
- Recall: `0.9760479041916168`
- F1-score: `0.978978978978979`
- ROC-AUC: `0.9971113050281347`
- True Negative: `196`
- False Positive: `3`
- False Negative: `4`
- True Positive: `163`

### What improved and what became worse

Compared with the original baseline:

- accuracy increased slightly;
- precision increased clearly;
- F1-score increased slightly;
- test loss improved;
- false positives decreased from `6` to `3`.

At the same time:

- recall decreased;
- false negatives increased from `2` to `4`;
- ROC-AUC remained almost unchanged.

### Prediction-change analysis

When I compared image-level predictions between the two models, I found:

- errors fixed by preprocessing: `4`
- new errors introduced by preprocessing: `3`
- wrong in both models: `4`
- correct in both models: `355`
- `False Positive -> True Negative`: `3`
- `False Negative -> True Positive`: `1`
- `True Positive -> False Negative`: `3`
- `True Negative -> False Positive`: `0`

This means the preprocessing pipeline mostly helped by reducing false alarms, but it also caused the model to miss several previously detected DR-positive cases.

### My interpretation of this result

This is a useful and realistic thesis result because the second experiment did not simply dominate the first one.
Instead, it created a clinically meaningful trade-off:

- the preprocessing-enhanced model became more precise and more conservative;
- the original baseline remained stronger in recall and therefore stronger in sensitivity to DR-positive cases.

For a medical screening context, recall is especially important because missing a diseased case is usually riskier than producing an extra false alarm.
So even though the preprocessing-enhanced model looks slightly better on some aggregate metrics, the original baseline may still remain preferable if screening sensitivity is treated as the top priority.

### Why this checkpoint matters

This checkpoint shows that I am now doing real experimental comparison rather than only training one model and reporting one score.
I used explainability findings to motivate a targeted pipeline change, then I evaluated the effect quantitatively and interpreted the trade-off in clinically relevant terms.

That makes the project substantially stronger as a diploma thesis because it demonstrates:

- hypothesis-driven experimentation;
- controlled comparison under the same split and settings;
- interpretation beyond raw accuracy;
- awareness of medical AI priorities such as false negatives and recall.

### Next step

My next step is to document the comparison visually in a dedicated notebook and then decide whether to continue with:

- threshold tuning to recover recall while keeping some preprocessing benefits; or
- architecture comparison using another backbone such as `EfficientNet-B0`.


## DR-CP-005

- Date: 2026-07-10
- Status: Completed
- Theme: First custom CNN from-scratch baseline

### What I did

I added and trained my first `custom CNN` from scratch as a reference model for the diabetic retinopathy classification task.
The purpose of this experiment was not to beat the pretrained transfer learning models, but to create a fair from-scratch baseline that can show the practical value of transfer learning in this project.

I also started organizing model-specific notebooks into separate folders so that experiments are easier to track and compare.
For this model, I created:

- `configs/custom_cnn_scratch.yaml`
- `notebooks/custom_cnn/05_custom_cnn_results.ipynb`
- `outputs/custom_cnn/`

### Why this experiment matters

Up to this point, the project used pretrained transfer learning backbones such as `ResNet18`.
That is completely valid, but I wanted to include one model trained from random initialization so that the thesis can answer an important methodological question:

- how much do pretrained transfer learning models help compared with a compact CNN trained from scratch on the same retinal dataset?

This makes the thesis stronger because it shows not only which model worked best, but also why transfer learning is justified for this medical imaging task.

### Experimental setup

- Task: binary classification
- Labels:
  - `0` = No Diabetic Retinopathy
  - `1` = Diabetic Retinopathy Present
- Model: `custom_cnn`
- Pretrained weights: `false`
- Config file: `configs/custom_cnn_scratch.yaml`
- Best model output: `models/custom_cnn_scratch/best_model.pt`
- Result notebook: `notebooks/custom_cnn/05_custom_cnn_results.ipynb`

### Architecture summary

The custom CNN is a compact convolutional model with approximately `1.01M` parameters.
It uses stacked convolution, batch normalization, ReLU, max-pooling, global average pooling, and a small fully connected classifier with dropout.

This makes it simple enough to explain clearly in the thesis and different enough from pretrained transfer learning backbones to be a meaningful reference baseline.

### Training setup I used

- Dataset split:
  - train: `2930`
  - validation: `366`
  - test: `366`
- Image size: `224 x 224`
- Batch size: `64`
- Epochs: `3`
- Learning rate: `0.001`
- Weight decay: `0.0001`
- Optimizer: `Adam`
- Loss function: `CrossEntropyLoss`
- Number of workers: `0`
- Seed: `42`

### Epoch-by-epoch results

- Epoch 1:
  - `train_loss = 0.3723`
  - `train_f1 = 0.8334`
  - `val_loss = 0.2237`
  - `val_f1 = 0.9380`
- Epoch 2:
  - `train_loss = 0.2660`
  - `train_f1 = 0.9011`
  - `val_loss = 0.1694`
  - `val_f1 = 0.9251`
- Epoch 3:
  - `train_loss = 0.2444`
  - `train_f1 = 0.9067`
  - `val_loss = 0.1491`
  - `val_f1 = 0.9465`

### Final test metrics

- Accuracy: `0.9426229508196722`
- Precision: `0.974025974025974`
- Recall: `0.8982035928143712`
- F1-score: `0.9345794392523364`
- ROC-AUC: `0.9756868173201336`
- Test loss: `0.19239683555123585`

### Confusion-matrix counts

- True Negative: `195`
- False Positive: `4`
- False Negative: `17`
- True Positive: `150`

### My interpretation of the result

This result is useful precisely because it is weaker than the pretrained transfer learning baselines.
The custom CNN from scratch can learn meaningful retinal patterns and reaches a respectable result, but it is clearly behind the pretrained `ResNet18` experiments.

The most important weakness is recall:

- custom CNN recall: `0.8982`
- pretrained ResNet18 recall: `0.9880`
- preprocessing-enhanced ResNet18 recall: `0.9760`

This means the from-scratch model misses substantially more DR-positive cases.
Its `False Negative` count is especially important:

- custom CNN: `17`
- pretrained ResNet18 baseline: `2`
- preprocessing-enhanced ResNet18: `4`

For medical screening, this is a serious disadvantage because false negatives are clinically risky.

### What this checkpoint proves

This checkpoint proves that:

- a simple CNN trained from scratch can solve the task to a reasonable degree;
- transfer learning is still much more effective for this dataset;
- pretrained models provide stronger sensitivity to DR-positive cases;
- the thesis now includes both from-scratch learning and transfer learning, which makes the methodology more complete.

### Files produced in this stage

- `models/custom_cnn_scratch/best_model.pt`
- `notebooks/custom_cnn/05_custom_cnn_results.ipynb`
- `outputs/custom_cnn/table_5_1_custom_cnn_metrics.csv`
- `outputs/custom_cnn/table_5_2_custom_cnn_confusion_counts.csv`
- `outputs/custom_cnn/figure_5_1_custom_cnn_confusion_matrix.png`
- `outputs/custom_cnn/figure_5_2_custom_cnn_roc_curve.png`
- `outputs/custom_cnn/image_5_1_custom_cnn_misclassified_examples.png`

### Next step

My next step is to place this from-scratch model into a broader comparison table against:

- pretrained `ResNet18`;
- preprocessing-enhanced `ResNet18`;
- a next transfer learning candidate such as `EfficientNet-B0`.


## DR-CP-006

- Date: 2026-07-10
- Status: Completed
- Theme: Grad-CAM explainability analysis of the custom CNN from scratch

### What I did

I completed the first Grad-CAM explainability stage for the `custom CNN` trained from scratch.
The purpose of this step was to move beyond performance metrics and inspect how the from-scratch model actually uses visual information from retinal fundus images.

For this stage, I created:

- `notebooks/custom_cnn/07_custom_cnn_gradcam.ipynb`
- `outputs/custom_cnn/gradcam/`

I visualized Grad-CAM explanations for:

- true positives;
- true negatives;
- false positives;
- false negatives.

### Files produced in this stage

- `outputs/custom_cnn/gradcam/table_7_1_custom_cnn_outcome_summary.csv`
- `outputs/custom_cnn/gradcam/figure_7_1_custom_cnn_outcome_distribution.png`
- `outputs/custom_cnn/gradcam/image_7_1_custom_cnn_true_positive_gradcam.png`
- `outputs/custom_cnn/gradcam/image_7_2_custom_cnn_true_negative_gradcam.png`
- `outputs/custom_cnn/gradcam/image_7_3_custom_cnn_false_positive_gradcam.png`
- `outputs/custom_cnn/gradcam/image_7_4_custom_cnn_false_negative_gradcam.png`

### Outcome summary on the test split

- True Negative: `195`
- True Positive: `150`
- False Negative: `17`
- False Positive: `4`

These counts are consistent with the earlier custom-CNN result and again confirm that the main weakness of the model is the high number of false negatives.

### What I observed from the Grad-CAM visualizations

The Grad-CAM maps for the custom CNN were much less clinically convincing than the maps produced earlier for the pretrained `ResNet18`.

In many examples, especially:

- true negatives;
- false positives;
- false negatives;

the model placed strong attention on:

- the outer retinal boundary;
- image corners;
- border contrast regions;
- peripheral non-lesion areas.

Even in several true positive examples, the strongest activation did not clearly center on the lesion clusters themselves.
Instead, the model often emphasized vertical edge-like regions near the left and right boundaries of the image.

### My interpretation of this stage

This is a very important finding for the thesis.
The custom CNN is not only weaker in quantitative performance, but it also appears less reliable in where it focuses.

Compared with the transfer learning models, the from-scratch network shows a stronger tendency to rely on shortcut-like cues related to image framing and border structure.
This helps explain why its recall is much worse and why it misses more DR-positive cases.

In other words:

- the model can still classify many images correctly;
- but its visual reasoning appears less clinically meaningful;
- therefore, its predictions are harder to trust in a medical setting.

### Why this checkpoint matters

This checkpoint strengthens the thesis in two ways:

- it shows that explainability is useful not only for strong models, but also for diagnosing weak models;
- it provides visual evidence for why transfer learning is preferable to a simple from-scratch CNN in this task.

This is a much stronger argument than reporting metrics alone, because the Grad-CAM results reveal that the custom CNN may be relying on non-clinical visual shortcuts rather than robust retinal pathology patterns.

### What this checkpoint proves

This checkpoint proves that:

- a from-scratch CNN can reach a reasonable score numerically;
- but its attention behavior may still be poor;
- Grad-CAM can expose this mismatch between performance and trustworthiness;
- transfer learning is justified not only by better metrics, but also by more plausible visual attention behavior.

### Next step

My next step is to use these findings to motivate the next transfer learning comparison stage, most likely with `EfficientNet-B0`, and later compare whether its Grad-CAM behavior is more clinically meaningful than the custom CNN.


## DR-CP-007

- Date: 2026-07-10
- Status: Completed
- Theme: Expanded transfer learning benchmark and baseline model selection

### What I did

I expanded the baseline stage from a single-model setup into a broader benchmark of both scratch and transfer learning models for binary diabetic retinopathy detection.

At this stage, I:

- trained `DenseNet121` as an additional pretrained transfer learning baseline;
- trained `EfficientNet-B0` as another pretrained transfer learning baseline;
- created separate result notebooks for both models;
- updated the global comparison notebook so that it now compares all major baseline candidates in one place.

This was an important step for the thesis, because the project now includes not just one successful model, but a structured experimental comparison between multiple architectures under the same evaluation protocol.

### Models included in the updated benchmark

The updated comparison now includes:

- `Custom CNN` trained from scratch
- `Baseline ResNet18`
- `ResNet18 + preprocessing`
- `DenseNet121`
- `EfficientNet-B0`

### Training setup I kept consistent

For the transfer learning benchmark, I kept the experimental protocol aligned as much as possible:

- Task: binary classification
- Labels:
  - `0` = No Diabetic Retinopathy
  - `1` = Diabetic Retinopathy Present
- Dataset split:
  - train: `2930`
  - validation: `366`
  - test: `366`
- Image size: `224 x 224`
- Optimizer: `Adam`
- Loss function: `CrossEntropyLoss`
- Learning rate: `0.0003`
- Batch size: `16`
- Epochs: `5`
- Seed: `42`

This consistency is important because it makes the comparison academically cleaner and easier to justify in the thesis.

### New transfer learning experiments completed

#### DenseNet121

- Config file: `configs/densenet121_baseline.yaml`
- Best model output: `models/densenet121_baseline/best_model.pt`
- Result notebook: `notebooks/transfer_learning/08_densenet121_results.ipynb`

Final test metrics:

- Accuracy: `0.9672`
- Precision: `0.9697`
- Recall: `0.9581`
- F1-score: `0.9639`
- ROC-AUC: `0.9932`

Confusion-matrix counts:

- True Negative: `194`
- False Positive: `5`
- False Negative: `7`
- True Positive: `160`

#### EfficientNet-B0

- Config file: `configs/efficientnet_b0_baseline.yaml`
- Best model output: `models/efficientnet_b0_baseline/best_model.pt`
- Result notebook: `notebooks/transfer_learning/09_efficientnet_b0_results.ipynb`

Final test metrics:

- Accuracy: `0.9891`
- Precision: `0.9939`
- Recall: `0.9820`
- F1-score: `0.9880`
- ROC-AUC: `0.9983`

Confusion-matrix counts:

- True Negative: `198`
- False Positive: `1`
- False Negative: `3`
- True Positive: `164`

### Updated all-model comparison

I updated:

- `notebooks/comparison/06_all_model_comparison.ipynb`

The notebook now saves an updated five-model comparison to:

- `outputs/comparison_all/table_6_1_all_model_metrics.csv`
- `outputs/comparison_all/table_6_2_all_model_confusion_counts.csv`
- `outputs/comparison_all/table_6_3_clinical_summary.csv`
- `outputs/comparison_all/figure_6_1_all_model_metric_comparison.png`
- `outputs/comparison_all/figure_6_2_all_model_error_tradeoff.png`
- `outputs/comparison_all/figure_6_3_all_model_roc_curves.png`

### Main comparison result

The final benchmark summary is:

- `EfficientNet-B0` achieved the best overall balance of metrics.
- `Baseline ResNet18` still achieved the highest recall.
- `ResNet18 + preprocessing` remained a strong balanced variant with fewer false positives than the original baseline.
- `DenseNet121` did not outperform the stronger ResNet18 and EfficientNet-B0 baselines in this setup.
- `Custom CNN` remained clearly weaker than the pretrained transfer learning models.

### Clinical interpretation

From a thesis perspective, the most interesting result is not only which model has the highest accuracy, but how the models behave in terms of false negatives and false positives.

The updated clinical ranking looks like this:

- `Baseline ResNet18` had the best recall: `0.9880` with only `2` false negatives.
- `EfficientNet-B0` had the best overall performance: `accuracy = 0.9891`, `f1 = 0.9880`, `roc_auc = 0.9983`, with only `1` false positive and `3` false negatives.
- `ResNet18 + preprocessing` gave a very good compromise with stronger precision and fewer false positives than the baseline.
- `DenseNet121` was acceptable but not competitive with the strongest baselines.
- `Custom CNN` had the weakest recall and the highest number of false negatives.

This means that `EfficientNet-B0` is currently the strongest overall model candidate, while `Baseline ResNet18` still remains valuable as the most recall-oriented screening model.

### What this checkpoint proves

This checkpoint proves that:

- the project now has a real multi-model baseline benchmark;
- transfer learning is clearly superior to the simple custom CNN trained from scratch;
- not all pretrained architectures perform equally well on this task;
- `EfficientNet-B0` is the strongest overall baseline candidate so far;
- `ResNet18` remains clinically interesting because of its very high recall.

### Files produced or updated in this stage

- `models/densenet121_baseline/best_model.pt`
- `models/efficientnet_b0_baseline/best_model.pt`
- `notebooks/transfer_learning/08_densenet121_results.ipynb`
- `notebooks/transfer_learning/09_efficientnet_b0_results.ipynb`
- `notebooks/comparison/06_all_model_comparison.ipynb`
- `outputs/densenet121_baseline/table_8_1_densenet121_metrics.csv`
- `outputs/densenet121_baseline/table_8_2_densenet121_confusion_counts.csv`
- `outputs/densenet121_baseline/figure_8_1_densenet121_confusion_matrix.png`
- `outputs/densenet121_baseline/figure_8_2_densenet121_roc_curve.png`
- `outputs/efficientnet_b0_baseline/table_9_1_efficientnet_b0_metrics.csv`
- `outputs/efficientnet_b0_baseline/table_9_2_efficientnet_b0_confusion_counts.csv`
- `outputs/efficientnet_b0_baseline/figure_9_1_efficientnet_b0_confusion_matrix.png`
- `outputs/efficientnet_b0_baseline/figure_9_2_efficientnet_b0_roc_curve.png`
- updated `outputs/comparison_all/*`

### Next step

My next step is to treat `EfficientNet-B0` as the main overall baseline candidate and continue with the next thesis-quality stage:

- explainability analysis for the strongest transfer learning model;
- or a more advanced extension such as multi-class classification.


## DR-CP-008

- Date: 2026-07-10
- Status: Completed
- Theme: Grad-CAM explainability analysis of the strongest transfer learning baseline

### What I did

I completed a dedicated Grad-CAM explainability stage for the strongest overall baseline model, `EfficientNet-B0`.
The purpose of this step was to check whether the best quantitative model is also visually more trustworthy and clinically plausible.

For this stage, I created:

- `notebooks/transfer_learning/10_efficientnet_b0_gradcam.ipynb`
- `outputs/efficientnet_b0_baseline/gradcam/`

I visualized Grad-CAM explanations for:

- true positives;
- true negatives;
- false positives;
- false negatives.

### Files produced in this stage

- `outputs/efficientnet_b0_baseline/gradcam/table_10_1_efficientnet_b0_outcome_summary.csv`
- `outputs/efficientnet_b0_baseline/gradcam/figure_10_1_efficientnet_b0_outcome_distribution.png`
- `outputs/efficientnet_b0_baseline/gradcam/image_10_1_efficientnet_b0_true_positive_gradcam.png`
- `outputs/efficientnet_b0_baseline/gradcam/image_10_2_efficientnet_b0_true_negative_gradcam.png`
- `outputs/efficientnet_b0_baseline/gradcam/image_10_3_efficientnet_b0_false_positive_gradcam.png`
- `outputs/efficientnet_b0_baseline/gradcam/image_10_4_efficientnet_b0_false_negative_gradcam.png`

### Outcome summary on the test split

- True Negative: `198`
- True Positive: `164`
- False Positive: `1`
- False Negative: `3`

These counts are fully consistent with the earlier confusion-matrix results and confirm again that `EfficientNet-B0` made only a very small number of mistakes on the test set.

### What I observed from the Grad-CAM visualizations

In the true positive examples, the model often focused on the central retinal region and lesion-containing areas near the macula.
In several cases, the strongest activation was concentrated in clinically meaningful central zones rather than only on the outer image border.

In the true negative examples, the attention was generally more centered inside the retinal field than what I observed earlier for the custom CNN.
However, the heatmaps were still not perfectly lesion-specific:

- the optic disc sometimes received substantial attention;
- some maps remained broad and diffuse across the central retina;
- not every correct prediction was based on a very tightly localized hotspot.

The false negative examples were especially informative.
In those images, the model often focused on broad central regions but did not sharply isolate the subtle suspicious lesions that were actually present.
This suggests that some mild or ambiguous DR patterns may still be under-emphasized by the model even when the overall benchmark result is excellent.

The single false positive example also revealed an important weakness:

- the model placed strong attention on bright peripheral regions and non-lesion-looking image structure;
- this suggests that even the strongest baseline may still be partially sensitive to visual artifacts or brightness cues.

### My interpretation of this stage

This is a strong result for the thesis.
`EfficientNet-B0` appears clearly more convincing than the custom CNN from scratch:

- its errors are much fewer;
- its attention is more centered within the retinal field;
- its Grad-CAM maps are generally more plausible for a medical imaging setting.

At the same time, the explainability analysis also shows that the model is not perfect.
Even the best baseline can still rely on broad central patterns, optic-disc-related structure, or bright peripheral cues in some difficult cases.

This is academically important because it prevents the thesis from making an overly simplistic claim that the top-scoring model is automatically fully trustworthy.

### Why this checkpoint matters

This checkpoint matters because it connects the quantitative benchmark with qualitative medical interpretability.
At this point, the project now shows:

- which model is strongest numerically;
- how that model behaves visually;
- and where even the strongest model can still fail.

This is much stronger than reporting test metrics alone, and it makes the baseline section look more like a real thesis experiment rather than a simple leaderboard.

### What this checkpoint proves

This checkpoint proves that:

- `EfficientNet-B0` is not only the strongest overall baseline by metrics, but also visually more credible than the custom scratch model;
- Grad-CAM remains useful even after strong quantitative performance is achieved;
- the best model still has some shortcut-risk and ambiguity in difficult cases;
- explainability should remain part of the final methodology, not just an optional add-on.

### Next step

My next step is to decide how to extend the thesis beyond the completed binary baseline benchmark:

- either move to multi-class diabetic retinopathy grading;
- or use `EfficientNet-B0` as the final main baseline and build the next advanced chapter around it.


## DR-CP-009

- Date: 2026-07-10
- Status: Completed
- Theme: First multi-class grading experiment with EfficientNet-B0

### What I did

I moved from the binary screening setup to the original five-class diabetic retinopathy grading task and trained the strongest current backbone, `EfficientNet-B0`, in multi-class mode.
This was an important transition because the task now became closer to the real clinical grading problem rather than only detecting whether DR is present or absent.

For this stage, I:

- enabled the original `APTOS 2019` labels `0, 1, 2, 3, 4`;
- added class-weighted multi-class training;
- changed the validation monitor metric to `f1_macro`;
- trained the first multi-class baseline;
- created a dedicated result-analysis notebook with class-wise evaluation and a `5 x 5` confusion matrix.

### Dataset status for multi-class

I did not need to download another dataset version.
The current split files already contain the original five severity labels:

- Train:
  - `0 = 1434`
  - `1 = 300`
  - `2 = 808`
  - `3 = 154`
  - `4 = 234`
- Validation:
  - `0 = 172`
  - `1 = 40`
  - `2 = 104`
  - `3 = 22`
  - `4 = 28`
- Test:
  - `0 = 199`
  - `1 = 30`
  - `2 = 87`
  - `3 = 17`
  - `4 = 33`

### Target labels used in this experiment

- `0` = No DR
- `1` = Mild DR
- `2` = Moderate DR
- `3` = Severe DR
- `4` = Proliferative DR

### Training setup I used

- Config file: `configs/efficientnet_b0_multiclass.yaml`
- Model: `EfficientNet-B0`
- Pretrained weights: `true`
- Task: multi-class classification
- Number of classes: `5`
- Dataset split:
  - train: `2930`
  - validation: `366`
  - test: `366`
- Image size: `224 x 224`
- Batch size: `16`
- Epochs: `5`
- Learning rate: `0.0003`
- Loss function: `CrossEntropyLoss`
- Class weighting: `balanced`
- Validation selection metric: `f1_macro`
- Seed: `42`

### Class weights used

To partially compensate for class imbalance, I used the following class weights:

- class `0`: `0.4086`
- class `1`: `1.9533`
- class `2`: `0.7252`
- class `3`: `3.8052`
- class `4`: `2.5043`

These weights reflect the fact that `Mild DR`, `Severe DR`, and `Proliferative DR` are much less frequent than `No DR` and `Moderate DR`.

### Epoch-by-epoch results

- Epoch 1:
  - `train_loss = 1.0782`
  - `train_f1_macro = 0.4977`
  - `val_loss = 0.8763`
  - `val_f1_macro = 0.6466`
- Epoch 2:
  - `train_loss = 0.8641`
  - `train_f1_macro = 0.6027`
  - `val_loss = 0.8337`
  - `val_f1_macro = 0.6230`
- Epoch 3:
  - `train_loss = 0.7545`
  - `train_f1_macro = 0.6492`
  - `val_loss = 0.7441`
  - `val_f1_macro = 0.6622`
- Epoch 4:
  - `train_loss = 0.6877`
  - `train_f1_macro = 0.6707`
  - `val_loss = 0.7933`
  - `val_f1_macro = 0.6738`
- Epoch 5:
  - `train_loss = 0.6205`
  - `train_f1_macro = 0.6995`
  - `val_loss = 0.8005`
  - `val_f1_macro = 0.6752`

### Final test metrics

- Accuracy: `0.7814`
- Precision macro: `0.6728`
- Recall macro: `0.6233`
- F1 macro: `0.5931`
- F1 weighted: `0.7841`
- Quadratic Weighted Kappa: `0.8567`
- ROC-AUC OVR macro: `0.9416`
- ROC-AUC OVR weighted: `0.9666`
- Test loss: `0.9883`

### Why these metrics matter

For the multi-class grading task, `accuracy` alone is not enough.
The most important metrics here are:

- `f1_macro`, because it reflects how fairly the model handles all five classes;
- `f1_weighted`, because it reflects performance while accounting for class frequency;
- `QWK`, because the labels are ordinal and mistakes between neighboring severity grades are less serious than completely wrong jumps.

The most encouraging result in this stage is the `QWK = 0.8567`.
This suggests that even when the model makes mistakes, it often predicts a nearby severity grade rather than a completely unrelated one.

### Per-class result analysis

The class-wise metrics showed a very uneven difficulty profile:

- `No DR`
  - precision: `0.9898`
  - recall: `0.9799`
  - f1: `0.9848`
- `Mild DR`
  - precision: `0.4068`
  - recall: `0.8000`
  - f1: `0.5393`
- `Moderate DR`
  - precision: `0.6712`
  - recall: `0.5632`
  - f1: `0.6125`
- `Severe DR`
  - precision: `0.2963`
  - recall: `0.4706`
  - f1: `0.3636`
- `Proliferative DR`
  - precision: `1.0000`
  - recall: `0.3030`
  - f1: `0.4651`

### What the confusion matrix revealed

The `5 x 5` confusion matrix showed several important patterns:

- `No DR` was classified very reliably:
  - `195 / 199` correct
- `Mild DR` was often detected, but also over-predicted:
  - `24 / 30` correct
- `Moderate DR` was frequently confused with neighboring grades:
  - `26` cases predicted as `Mild DR`
  - `10` cases predicted as `Severe DR`
- `Severe DR` was strongly confused with `Moderate DR`:
  - `9 / 17` severe cases predicted as moderate
- `Proliferative DR` was predicted very conservatively:
  - only `10 / 33` correct
  - many cases shifted down to `Severe` or `Moderate`

This is exactly why `QWK` is much stronger than `f1_macro`.
The model still struggles with rare severe classes, but its mistakes are often ordinally close.

### My interpretation of the result

This is a good first multi-class result.
It is clearly harder than the binary task, and the reduction in macro metrics is expected.

The experiment shows that:

- the model already understands the grading structure reasonably well;
- the dominant weakness is class imbalance and confusion between neighboring disease stages;
- the severe end of the scale remains the hardest part of the task;
- the project can now move from screening-oriented classification toward real DR severity grading.

From a thesis perspective, this is useful because it creates a more advanced and clinically realistic chapter.
At the same time, the result also gives a clear motivation for future improvements such as:

- stronger preprocessing for the multi-class task;
- weighted sampling;
- focal loss;
- ordinal classification strategies.

### Files produced in this stage

- `models/efficientnet_b0_multiclass/best_model.pt`
- `notebooks/multiclass/11_efficientnet_b0_multiclass_results.ipynb`
- `outputs/efficientnet_b0_multiclass/table_11_0_multiclass_predictions.csv`
- `outputs/efficientnet_b0_multiclass/table_11_1_multiclass_metrics.csv`
- `outputs/efficientnet_b0_multiclass/table_11_2_multiclass_per_class_metrics.csv`
- `outputs/efficientnet_b0_multiclass/table_11_3_multiclass_confusion_matrix_counts.csv`
- `outputs/efficientnet_b0_multiclass/figure_11_1_multiclass_confusion_matrix.png`
- `outputs/efficientnet_b0_multiclass/figure_11_2_multiclass_confusion_matrix_normalized.png`
- `outputs/efficientnet_b0_multiclass/figure_11_3_multiclass_per_class_metrics.png`

### Next step

My next step is to improve the multi-class setting in a targeted way.
The most justified next improvement would be to keep `EfficientNet-B0` and test whether better data handling can improve the rare classes:

- weighted sampling;
- stronger retinal preprocessing;
- or an ordinal-learning style approach.


## DR-CP-010

- Date: 2026-07-10
- Status: Completed
- Theme: Multi-class preprocessing comparison for EfficientNet-B0

### What I did

After the first multi-class baseline, I tested whether retinal-focused preprocessing could improve diabetic retinopathy grading performance.
Instead of jumping directly to a very complex preprocessing pipeline, I designed this stage as a step-by-step comparison:

- baseline without preprocessing;
- crop-based retinal field preprocessing;
- crop-based retinal field preprocessing with CLAHE.

This was important because it made the experiment easier to interpret and more defensible in the thesis.

### Experimental variants I compared

#### Variant A: Baseline multi-class

- Config file: `configs/efficientnet_b0_multiclass.yaml`
- Preprocessing:
  - none

#### Variant B: Crop-only multi-class

- Config file: `configs/efficientnet_b0_multiclass_crop.yaml`
- Preprocessing:
  - dark border crop
  - center square crop

#### Variant C: Crop + CLAHE multi-class

- Config file: `configs/efficientnet_b0_multiclass_crop_clahe.yaml`
- Preprocessing:
  - dark border crop
  - center square crop
  - CLAHE

### Training setup I kept fixed

For all three variants, I kept the same:

- Model: `EfficientNet-B0`
- Task: multi-class classification
- Classes: `0, 1, 2, 3, 4`
- Dataset split:
  - train: `2930`
  - validation: `366`
  - test: `366`
- Image size: `224 x 224`
- Batch size: `16`
- Epochs: `5`
- Learning rate: `0.0003`
- Loss function: `CrossEntropyLoss`
- Class weighting: `balanced`
- Validation metric: `f1_macro`
- Seed: `42`

This makes the comparison much cleaner, because the only meaningful difference between variants is the input preprocessing pipeline.

### Crop-only experiment results

#### Epoch-by-epoch validation summary

- Epoch 1:
  - `train_loss = 1.1092`
  - `train_f1_macro = 0.4933`
  - `val_loss = 0.8968`
  - `val_f1_macro = 0.6010`
- Epoch 2:
  - `train_loss = 0.8670`
  - `train_f1_macro = 0.6108`
  - `val_loss = 0.7927`
  - `val_f1_macro = 0.6489`
- Epoch 3:
  - `train_loss = 0.7425`
  - `train_f1_macro = 0.6590`
  - `val_loss = 0.7553`
  - `val_f1_macro = 0.6480`
- Epoch 4:
  - `train_loss = 0.6410`
  - `train_f1_macro = 0.6970`
  - `val_loss = 0.8909`
  - `val_f1_macro = 0.6515`
- Epoch 5:
  - `train_loss = 0.5847`
  - `train_f1_macro = 0.7163`
  - `val_loss = 0.9048`
  - `val_f1_macro = 0.6740`

#### Final test metrics

- Accuracy: `0.7814`
- Precision macro: `0.6106`
- Recall macro: `0.6110`
- F1 macro: `0.5840`
- F1 weighted: `0.7855`
- QWK: `0.8650`
- ROC-AUC OVR macro: `0.9358`
- ROC-AUC OVR weighted: `0.9631`

### Crop + CLAHE experiment results

#### Epoch-by-epoch validation summary

- Epoch 1:
  - `train_loss = 1.0947`
  - `train_f1_macro = 0.4863`
  - `val_loss = 0.9156`
  - `val_f1_macro = 0.5695`
- Epoch 2:
  - `train_loss = 0.8510`
  - `train_f1_macro = 0.6144`
  - `val_loss = 0.8831`
  - `val_f1_macro = 0.6018`
- Epoch 3:
  - `train_loss = 0.7216`
  - `train_f1_macro = 0.6757`
  - `val_loss = 0.8351`
  - `val_f1_macro = 0.6204`
- Epoch 4:
  - `train_loss = 0.6317`
  - `train_f1_macro = 0.7044`
  - `val_loss = 0.8951`
  - `val_f1_macro = 0.6635`
- Epoch 5:
  - `train_loss = 0.5424`
  - `train_f1_macro = 0.7282`
  - `val_loss = 0.9362`
  - `val_f1_macro = 0.6707`

#### Final test metrics

- Accuracy: `0.8033`
- Precision macro: `0.6325`
- Recall macro: `0.6274`
- F1 macro: `0.6156`
- F1 weighted: `0.8051`
- QWK: `0.8702`
- ROC-AUC OVR macro: `0.9325`
- ROC-AUC OVR weighted: `0.9623`

### Overall comparison summary

The three-way comparison produced the following result:

- Baseline:
  - accuracy: `0.7814`
  - f1_macro: `0.5931`
  - f1_weighted: `0.7841`
  - qwk: `0.8567`
- Crop:
  - accuracy: `0.7814`
  - f1_macro: `0.5840`
  - f1_weighted: `0.7855`
  - qwk: `0.8650`
- Crop + CLAHE:
  - accuracy: `0.8033`
  - f1_macro: `0.6156`
  - f1_weighted: `0.8051`
  - qwk: `0.8702`

### What the comparison means

The main conclusion is very clear:

- crop-only preprocessing was not enough to improve the model in a stable overall way;
- crop-only slightly improved ordinal agreement (`QWK`), but did not improve `f1_macro`;
- crop + CLAHE produced the best overall result across the most important multi-class metrics.

This is a strong experimental finding, because it shows that:

- simply isolating the retinal field is not sufficient by itself;
- local contrast enhancement appears to help the model distinguish severity grades more effectively;
- the benefit of preprocessing is real, but it depends on using the right combination of steps.

### Per-class findings

The class-wise comparison also gave useful insight:

- `No DR` remained very strong in all variants;
- `Moderate DR` improved meaningfully with `Crop + CLAHE`:
  - baseline f1: `0.6125`
  - crop f1: `0.6543`
  - crop + CLAHE f1: `0.6936`
- `Mild DR` also improved with `Crop + CLAHE`:
  - baseline f1: `0.5393`
  - crop f1: `0.5067`
  - crop + CLAHE f1: `0.5758`
- `Proliferative DR` improved slightly under `Crop + CLAHE`:
  - baseline f1: `0.4651`
  - crop f1: `0.4167`
  - crop + CLAHE f1: `0.4706`
- `Severe DR` remained difficult in all three variants:
  - baseline f1: `0.3636`
  - crop f1: `0.3600`
  - crop + CLAHE f1: `0.3556`

An interesting rare-class detail was:

- `Severe DR` recall improved under crop-only from `0.4706` to `0.5294`;
- but this did not translate into the best overall multi-class behavior;
- `Crop + CLAHE` gave the best broader balance, especially for `Moderate DR`, `Mild DR`, and the overall ordinal score.

### My interpretation of this stage

This stage is exactly the kind of result that strengthens the thesis.
It is not a trivial “more preprocessing is always better” story.

Instead, the project now shows a more nuanced conclusion:

- one preprocessing step alone may not help enough;
- a retinal-focused crop combined with local contrast enhancement can improve grading quality;
- the effect of preprocessing should be measured carefully with class-wise and ordinal metrics, not only accuracy.

This makes the multi-class chapter much more research-oriented.

### Files produced in this stage

- `notebooks/multiclass/12_multiclass_preprocessing_comparison.ipynb`
- `outputs/multiclass_comparison/table_12_1_multiclass_preprocessing_metrics.csv`
- `outputs/multiclass_comparison/table_12_2_multiclass_per_class_f1.csv`
- `outputs/multiclass_comparison/table_12_3_multiclass_rare_class_recall.csv`
- `outputs/multiclass_comparison/figure_12_1_multiclass_preprocessing_metric_comparison.png`
- `outputs/multiclass_comparison/figure_12_2_multiclass_preprocessing_per_class_f1.png`
- `outputs/multiclass_comparison/figure_12_3_multiclass_preprocessing_rare_class_recall.png`

### What this checkpoint proves

This checkpoint proves that:

- preprocessing can improve multi-class DR grading, but not every preprocessing step is equally useful;
- `Crop + CLAHE` is the strongest preprocessing variant tested so far for multi-class EfficientNet-B0;
- ordinal agreement (`QWK`) is a very important metric for this task;
- multi-class evaluation should focus on both overall scores and rare severe classes.

### Next step

My next step is to keep `EfficientNet-B0 + multi-class + crop + CLAHE` as the new strongest multi-class baseline and improve the hardest rare classes through one of the following:

- weighted sampling;
- focal loss;
- or an ordinal-classification strategy.


## DR-CP-011

- Date: 2026-07-22
- Status: Completed
- Theme: Weighted sampler experiment for rare severe classes

### What I did

After selecting `EfficientNet-B0 + multi-class + crop + CLAHE` as the strongest preprocessing-based setup, I tested whether a `weighted sampler` could help the rare severe classes.

In this experiment I kept the retinal preprocessing fixed and changed only the training data sampling strategy.
The goal was to show rare classes more often during training and check whether this improves the hardest clinical cases without redesigning the model itself.

### Experimental setup

- Config file: `configs/efficientnet_b0_multiclass_crop_clahe_weighted_sampler.yaml`
- Model: `EfficientNet-B0`
- Task: multi-class classification
- Classes: `0, 1, 2, 3, 4`
- Preprocessing:
  - dark border crop
  - center square crop
  - CLAHE
- Loss function: `CrossEntropyLoss`
- Class weighting: `none`
- Sampling:
  - `weighted sampler`
  - `replacement = true`
- Batch size: `16`
- Epochs: `5`
- Learning rate: `0.0003`
- Validation metric: `f1_macro`
- Seed: `42`
- Data split:
  - train: `2930`
  - validation: `366`
  - test: `366`

### Training summary

- Epoch 1:
  - `train_loss = 0.9212`
  - `train_f1_macro = 0.6263`
  - `val_loss = 0.5686`
  - `val_f1_macro = 0.6288`
- Epoch 2:
  - `train_loss = 0.6106`
  - `train_f1_macro = 0.7691`
  - `val_loss = 0.5817`
  - `val_f1_macro = 0.6441`
- Epoch 3:
  - `train_loss = 0.4381`
  - `train_f1_macro = 0.8386`
  - `val_loss = 0.5933`
  - `val_f1_macro = 0.6433`
- Epoch 4:
  - `train_loss = 0.3515`
  - `train_f1_macro = 0.8745`
  - `val_loss = 0.6795`
  - `val_f1_macro = 0.6493`
- Epoch 5:
  - `train_loss = 0.3144`
  - `train_f1_macro = 0.8856`
  - `val_loss = 0.7135`
  - `val_f1_macro = 0.6770`

### Final test metrics

- Accuracy: `0.8115`
- Precision macro: `0.6348`
- Recall macro: `0.6198`
- F1 macro: `0.6154`
- F1 weighted: `0.8122`
- QWK: `0.8760`
- ROC-AUC OVR macro: `0.8991`
- ROC-AUC OVR weighted: `0.9526`
- Loss: `0.7213`

### Rare-class findings

Compared with the previous `Crop + CLAHE` setup:

- `Severe DR recall` decreased from `0.4706` to `0.3529`
- `Severe DR F1` decreased from `0.3556` to `0.2857`
- `Proliferative DR recall` stayed at `0.3636`
- `Proliferative DR F1` stayed at `0.4706`

### My interpretation of this stage

This result was useful because it was not a trivial success story.

The `weighted sampler` improved several overall metrics:

- accuracy improved from `0.8033` to `0.8115`
- weighted F1 improved from `0.8051` to `0.8122`
- QWK improved from `0.8702` to `0.8760`

However, the main target of this stage was the rare severe classes, and here the result was disappointing.
The sampler did not improve `Proliferative DR`, and it made `Severe DR` worse.

This means that:

- oversampling rare classes alone is not enough;
- the model may still struggle to learn the truly difficult visual boundaries;
- improving global performance is not the same as improving the clinically hardest cases.

### Files produced in this stage

- `notebooks/multiclass/13_efficientnet_b0_multiclass_weighted_sampler_results.ipynb`
- `outputs/efficientnet_b0_multiclass_crop_clahe_weighted_sampler/table_13_1_multiclass_weighted_sampler_metrics.csv`
- `outputs/efficientnet_b0_multiclass_crop_clahe_weighted_sampler/table_13_2_multiclass_weighted_sampler_per_class_metrics.csv`
- `outputs/efficientnet_b0_multiclass_crop_clahe_weighted_sampler/table_13_3_multiclass_weighted_sampler_confusion_matrix_counts.csv`
- `outputs/efficientnet_b0_multiclass_crop_clahe_weighted_sampler/table_13_4_weighted_sampler_rare_class_comparison.csv`
- `outputs/efficientnet_b0_multiclass_crop_clahe_weighted_sampler/figure_13_1_multiclass_weighted_sampler_confusion_matrix.png`
- `outputs/efficientnet_b0_multiclass_crop_clahe_weighted_sampler/figure_13_2_multiclass_weighted_sampler_confusion_matrix_normalized.png`
- `outputs/efficientnet_b0_multiclass_crop_clahe_weighted_sampler/figure_13_3_multiclass_weighted_sampler_per_class_metrics.png`
- `outputs/efficientnet_b0_multiclass_crop_clahe_weighted_sampler/figure_13_4_weighted_sampler_rare_class_comparison.png`

### What this checkpoint proves

This checkpoint proves that:

- `weighted sampler` can improve overall multi-class stability;
- better global metrics do not automatically mean better severe-stage detection;
- rare-class experiments must be judged with class-wise recall and F1, not only accuracy and weighted averages.

### Next step

My next step is to try `Focal Loss`, because it should focus the optimization process on harder examples rather than only changing how often classes appear in the batches.


## DR-CP-012

- Date: 2026-07-22
- Status: Completed
- Theme: Focal loss experiment for rare severe classes

### What I did

After the weighted sampler experiment, I tested `Focal Loss` on the same strong multi-class setup:

- `EfficientNet-B0`
- `crop + CLAHE`
- multi-class diabetic retinopathy grading

The goal of this stage was to see whether changing the loss function would help the model pay more attention to difficult and misclassified examples, especially `Severe DR`.

### Experimental setup

- Config file: `configs/efficientnet_b0_multiclass_crop_clahe_focal.yaml`
- Model: `EfficientNet-B0`
- Task: multi-class classification
- Classes: `0, 1, 2, 3, 4`
- Preprocessing:
  - dark border crop
  - center square crop
  - CLAHE
- Loss function:
  - `Focal Loss`
  - `gamma = 2.0`
- Class weighting: `balanced`
- Sampling:
  - standard shuffled batches
- Batch size: `16`
- Epochs: `5`
- Learning rate: `0.0003`
- Validation metric: `f1_macro`
- Seed: `42`
- Data split:
  - train: `2930`
  - validation: `366`
  - test: `366`

### Training summary

- Epoch 1:
  - `train_loss = 0.6133`
  - `train_f1_macro = 0.4758`
  - `val_loss = 0.4920`
  - `val_f1_macro = 0.5582`
- Epoch 2:
  - `train_loss = 0.4318`
  - `train_f1_macro = 0.6085`
  - `val_loss = 0.4753`
  - `val_f1_macro = 0.6115`
- Epoch 3:
  - `train_loss = 0.3478`
  - `train_f1_macro = 0.6591`
  - `val_loss = 0.4788`
  - `val_f1_macro = 0.6238`
- Epoch 4:
  - `train_loss = 0.2944`
  - `train_f1_macro = 0.7005`
  - `val_loss = 0.4938`
  - `val_f1_macro = 0.6659`
- Epoch 5:
  - `train_loss = 0.2350`
  - `train_f1_macro = 0.7315`
  - `val_loss = 0.6242`
  - `val_f1_macro = 0.6186`

### Final test metrics

- Accuracy: `0.7869`
- Precision macro: `0.6311`
- Recall macro: `0.6434`
- F1 macro: `0.6120`
- F1 weighted: `0.7942`
- QWK: `0.8586`
- ROC-AUC OVR macro: `0.9316`
- ROC-AUC OVR weighted: `0.9626`
- Loss: `0.4635`

### Rare-class findings

Compared with the previous `Crop + CLAHE` setup:

- `Severe DR recall` improved from `0.4706` to `0.5882`
- `Severe DR F1` improved from `0.3556` to `0.4082`
- `Proliferative DR recall` stayed at `0.3636`
- `Proliferative DR F1` improved slightly from `0.4706` to `0.4800`

### My interpretation of this stage

This experiment gave a much more clinically interesting result than the weighted sampler stage.

`Focal Loss` did not improve the global scores:

- accuracy dropped from `0.8033` to `0.7869`
- weighted F1 dropped from `0.8051` to `0.7942`
- QWK dropped from `0.8702` to `0.8586`

But it improved the class that was most problematic in the earlier experiments:

- `Severe DR recall` improved to `0.5882`
- `Severe DR F1` improved to `0.4082`

This means the experiment supports an important thesis-level conclusion:

- one optimization strategy may improve global performance;
- another may improve sensitivity to clinically dangerous rare stages;
- therefore, model selection should depend on the real objective of the system, not only on one overall metric.

### Files produced in this stage

- `notebooks/multiclass/14_efficientnet_b0_multiclass_focal_results.ipynb`
- `outputs/efficientnet_b0_multiclass_crop_clahe_focal/table_14_1_multiclass_focal_metrics.csv`
- `outputs/efficientnet_b0_multiclass_crop_clahe_focal/table_14_2_multiclass_focal_per_class_metrics.csv`
- `outputs/efficientnet_b0_multiclass_crop_clahe_focal/table_14_3_multiclass_focal_confusion_matrix_counts.csv`
- `outputs/efficientnet_b0_multiclass_crop_clahe_focal/table_14_4_focal_rare_class_comparison.csv`
- `outputs/efficientnet_b0_multiclass_crop_clahe_focal/table_14_5_multiclass_strategy_comparison.csv`
- `outputs/efficientnet_b0_multiclass_crop_clahe_focal/figure_14_1_multiclass_focal_confusion_matrix.png`
- `outputs/efficientnet_b0_multiclass_crop_clahe_focal/figure_14_2_multiclass_focal_confusion_matrix_normalized.png`
- `outputs/efficientnet_b0_multiclass_crop_clahe_focal/figure_14_3_multiclass_focal_per_class_metrics.png`
- `outputs/efficientnet_b0_multiclass_crop_clahe_focal/figure_14_4_focal_rare_class_comparison.png`

### What this checkpoint proves

This checkpoint proves that:

- `Focal Loss` can improve the hardest severe class even when overall accuracy becomes lower;
- rare-class sensitivity and overall score may move in opposite directions;
- this project is already producing non-trivial experimental tradeoffs, which is exactly what makes the thesis more research-oriented.

### Next step

My next step is to combine both ideas in a new experiment:

- `Focal Loss`
- `weighted sampler`

The reason is simple:

- `weighted sampler` improved overall stability;
- `Focal Loss` improved `Severe DR`;
- combining them may improve both rare-class sensitivity and broader multi-class balance.


## DR-CP-013

- Date: 2026-07-22
- Status: Completed
- Theme: Combined `Focal Loss + Weighted Sampler` experiment

### What I did

After testing `weighted sampler` and `Focal Loss` separately, I combined both ideas into a single multi-class experiment.

The goal of this stage was to check whether I could get the best parts of both strategies at the same time:

- stronger attention to difficult examples through `Focal Loss`;
- more frequent exposure to rare classes through `weighted sampler`.

This was an important experiment because the previous two stages showed complementary behavior:

- `weighted sampler` improved several overall metrics but did not help `Severe DR`;
- `Focal Loss` improved `Severe DR`, but reduced global performance.

So this combined setup was a natural next step.

### Experimental setup

- Config file: `configs/efficientnet_b0_multiclass_crop_clahe_focal_weighted_sampler.yaml`
- Model: `EfficientNet-B0`
- Task: multi-class classification
- Classes: `0, 1, 2, 3, 4`
- Preprocessing:
  - dark border crop
  - center square crop
  - CLAHE
- Loss function:
  - `Focal Loss`
  - `gamma = 2.0`
- Class weighting: `balanced`
- Sampling:
  - `weighted sampler`
  - `replacement = true`
- Batch size: `16`
- Epochs: `5`
- Learning rate: `0.0003`
- Validation metric: `f1_macro`
- Seed: `42`
- Data split:
  - train: `2930`
  - validation: `366`
  - test: `366`

### Training summary

- Epoch 1:
  - `train_loss = 0.8368`
  - `train_f1_macro = 0.5302`
  - `val_loss = 0.5587`
  - `val_f1_macro = 0.5084`
- Epoch 2:
  - `train_loss = 0.4662`
  - `train_f1_macro = 0.6898`
  - `val_loss = 0.4953`
  - `val_f1_macro = 0.5930`
- Epoch 3:
  - `train_loss = 0.3398`
  - `train_f1_macro = 0.7759`
  - `val_loss = 0.5771`
  - `val_f1_macro = 0.5750`
- Epoch 4:
  - `train_loss = 0.2590`
  - `train_f1_macro = 0.8062`
  - `val_loss = 0.5991`
  - `val_f1_macro = 0.5697`
- Epoch 5:
  - `train_loss = 0.2194`
  - `train_f1_macro = 0.8427`
  - `val_loss = 0.7609`
  - `val_f1_macro = 0.6182`

### Final test metrics

- Accuracy: `0.8005`
- Precision macro: `0.6317`
- Recall macro: `0.6527`
- F1 macro: `0.6325`
- F1 weighted: `0.8047`
- QWK: `0.8587`
- ROC-AUC OVR macro: `0.9077`
- ROC-AUC OVR weighted: `0.9550`
- Loss: `0.6788`

### Per-class findings

- `No DR`:
  - precision: `0.9846`
  - recall: `0.9648`
  - F1: `0.9746`
- `Mild DR`:
  - precision: `0.4667`
  - recall: `0.7000`
  - F1: `0.5600`
- `Moderate DR`:
  - precision: `0.7571`
  - recall: `0.6092`
  - F1: `0.6752`
- `Severe DR`:
  - precision: `0.4615`
  - recall: `0.3529`
  - F1: `0.4000`
- `Proliferative DR`:
  - precision: `0.4884`
  - recall: `0.6364`
  - F1: `0.5526`

### Rare-class findings

Compared with the previous `Crop + CLAHE` setup:

- `Severe DR recall` decreased from `0.4706` to `0.3529`
- `Severe DR F1` increased from `0.3556` to `0.4000`
- `Proliferative DR recall` improved strongly from `0.3636` to `0.6364`
- `Proliferative DR F1` improved from `0.4706` to `0.5526`

Compared with the earlier `Focal Loss` setup:

- `Severe DR recall` became lower
- `Severe DR F1` became slightly lower
- `Proliferative DR recall` improved strongly
- `Proliferative DR F1` improved strongly

This means the combined strategy shifted the benefit from `Severe DR` toward `Proliferative DR`, while still keeping a decent `Severe DR` F1.

### Strategy comparison insight

Across the four strongest multi-class strategies so far:

- `Crop + CLAHE`
- `Weighted Sampler`
- `Focal Loss`
- `Focal Loss + Weighted Sampler`

the combined setup produced:

- the best `recall_macro`: `0.6527`
- the best `f1_macro`: `0.6325`
- near-baseline `f1_weighted`: `0.8047`

At the same time:

- it was not the best `accuracy` model
- it was not the best `QWK` model
- it was not the best `Severe DR recall` model

So this experiment did not create one universally best model, but it did produce the best overall macro balance across classes.

### My interpretation of this stage

This is one of the strongest thesis-level findings in the project so far.

The result is not simplistic.
The combined method did not dominate every metric, but it improved the balance of the system in a meaningful way.

My interpretation is:

- `weighted sampler` alone mostly helped overall stability;
- `Focal Loss` alone mostly helped `Severe DR`;
- the combined version improved the macro-level class balance and strongly helped `Proliferative DR`;
- the effect of rare-class optimization depends on which severe stage I want to prioritize.

This is a much more serious and research-like conclusion than simply saying that one method is always better.

### Files produced in this stage

- `notebooks/multiclass/15_efficientnet_b0_multiclass_focal_weighted_results.ipynb`
- `outputs/efficientnet_b0_multiclass_crop_clahe_focal_weighted_sampler/table_15_1_multiclass_focal_weighted_metrics.csv`
- `outputs/efficientnet_b0_multiclass_crop_clahe_focal_weighted_sampler/table_15_2_multiclass_focal_weighted_per_class_metrics.csv`
- `outputs/efficientnet_b0_multiclass_crop_clahe_focal_weighted_sampler/table_15_3_multiclass_focal_weighted_confusion_matrix_counts.csv`
- `outputs/efficientnet_b0_multiclass_crop_clahe_focal_weighted_sampler/table_15_4_focal_weighted_rare_class_comparison.csv`
- `outputs/efficientnet_b0_multiclass_crop_clahe_focal_weighted_sampler/table_15_5_multiclass_strategy_comparison.csv`
- `outputs/efficientnet_b0_multiclass_crop_clahe_focal_weighted_sampler/figure_15_1_multiclass_focal_weighted_confusion_matrix.png`
- `outputs/efficientnet_b0_multiclass_crop_clahe_focal_weighted_sampler/figure_15_2_multiclass_focal_weighted_confusion_matrix_normalized.png`
- `outputs/efficientnet_b0_multiclass_crop_clahe_focal_weighted_sampler/figure_15_3_multiclass_focal_weighted_per_class_metrics.png`
- `outputs/efficientnet_b0_multiclass_crop_clahe_focal_weighted_sampler/figure_15_4_focal_weighted_rare_class_comparison.png`
- `outputs/efficientnet_b0_multiclass_crop_clahe_focal_weighted_sampler/figure_15_5_multiclass_strategy_comparison.png`

### What this checkpoint proves

This checkpoint proves that:

- combining loss engineering and sampling strategy can improve macro-level class balance;
- rare-class optimization produces tradeoffs between different severe stages;
- the best model depends on the evaluation priority:
  - best overall stability;
  - best `Severe DR`;
  - or best broader rare-class balance.

### Strategic decision after this stage

At this point, I decided not to continue spending more time on further multi-class fine-tuning.

The reason is not that the multi-class branch failed.
On the contrary, it already produced a strong and useful research result:

- I tested several meaningful strategies;
- I showed that the task is substantially harder than binary classification;
- I documented real tradeoffs between global metrics and rare severe classes;
- I demonstrated that there is no single simple method that dominates all criteria.

This means the multi-class direction is already valuable enough for the thesis as an advanced experimental chapter.

However, the recent experiments also show diminishing returns:

- further improvements are becoming small;
- gains in one class often create losses in another;
- continuing this branch too long would likely cost a lot of time without a proportional thesis benefit.

Because of this, I treat the current multi-class stage as sufficiently explored for now and I keep it as a completed research branch rather than the main development direction.

### Next step

My next step should focus on the strongest and most defensible direction of the project:

- keep the multi-class experiments as a completed advanced branch;
- return to the stronger binary pipeline;
- strengthen explainability and final result presentation;
- prepare a cleaner final comparison and thesis discussion chapter.

In practical terms, the best next work is:

- refine the binary classification chapter;
- expand explainability analysis for the final selected binary model;
- prepare publication-quality tables, figures, and narrative conclusions;
- and connect all experimental stages into a coherent thesis story.


## DR-CP-014

- Date: 2026-07-22
- Status: Completed
- Theme: `EfficientNet-B4` binary baseline extension

### What I did

After deciding to return attention to the binary branch, I added `EfficientNet-B4` as one more transfer learning baseline candidate.

The purpose of this stage was simple:

- test whether a larger EfficientNet variant can outperform `EfficientNet-B0`;
- keep the same binary diabetic retinopathy task;
- and compare the result fairly against the current strongest binary baseline models.

This was useful because `EfficientNet-B0` had already become the strongest overall binary candidate, so `EfficientNet-B4` was a natural architecture-level extension.

### Experimental setup

- Config file: `configs/efficientnet_b4_baseline.yaml`
- Model: `EfficientNet-B4`
- Task: binary classification
- Classes:
  - `0 = No DR`
  - `1 = DR Present`
- Preprocessing:
  - none
- Image size: `224 x 224`
- Batch size: `8`
- Epochs: `5`
- Learning rate: `0.0003`
- Number of workers: `0`
- Pretrained ImageNet weights: `true`
- Seed: `42`
- Data split:
  - train: `2930`
  - validation: `366`
  - test: `366`

### Training summary

- Epoch 1:
  - `train_loss = 0.1976`
  - `train_f1 = 0.9272`
  - `val_loss = 0.0654`
  - `val_f1 = 0.9871`
- Epoch 2:
  - `train_loss = 0.0995`
  - `train_f1 = 0.9682`
  - `val_loss = 0.0785`
  - `val_f1 = 0.9820`
- Epoch 3:
  - `train_loss = 0.0693`
  - `train_f1 = 0.9803`
  - `val_loss = 0.0566`
  - `val_f1 = 0.9897`
- Epoch 4:
  - `train_loss = 0.0525`
  - `train_f1 = 0.9806`
  - `val_loss = 0.0488`
  - `val_f1 = 0.9795`
- Epoch 5:
  - `train_loss = 0.0517`
  - `train_f1 = 0.9837`
  - `val_loss = 0.0530`
  - `val_f1 = 0.9764`

### Final test metrics

- Accuracy: `0.9754`
- Precision: `0.9647`
- Recall: `0.9820`
- F1-score: `0.9733`
- ROC-AUC: `0.9983`
- Loss: `0.0637`

### Confusion-matrix summary

- True Negative: `193`
- False Positive: `6`
- False Negative: `3`
- True Positive: `164`

### Direct comparison with `EfficientNet-B0`

The current `EfficientNet-B0` baseline remains:

- Accuracy: `0.9891`
- Precision: `0.9939`
- Recall: `0.9820`
- F1-score: `0.9880`
- ROC-AUC: `0.9983`
- False Positive: `1`
- False Negative: `3`

Compared with `EfficientNet-B0`, the `EfficientNet-B4` result shows:

- same recall
- same number of false negatives
- worse accuracy
- worse precision
- worse F1-score
- many more false positives

Its ROC-AUC is numerically extremely close and even slightly higher at the fourth decimal place, but this difference is too small to matter practically in this experiment.

### My interpretation of this stage

This result is very informative even though `EfficientNet-B4` did not become the new best model.

It shows that a larger architecture does not automatically improve the binary DR classifier in the current setup.
In fact, under the present training conditions:

- `EfficientNet-B0` remains clearly better as a decision model;
- `EfficientNet-B4` preserved recall, but lost specificity and precision;
- simply increasing model size is not enough to guarantee a stronger result.

This is a useful thesis finding because it prevents a simplistic assumption that “bigger network = better model.”

### Files produced in this stage

- `models/efficientnet_b4_baseline/best_model.pt`
- `configs/efficientnet_b4_baseline.yaml`
- `notebooks/transfer_learning/16_efficientnet_b4_results.ipynb`
- `outputs/efficientnet_b4_baseline/table_16_1_efficientnet_b4_metrics.csv`
- `outputs/efficientnet_b4_baseline/table_16_2_efficientnet_b4_confusion_counts.csv`
- `outputs/efficientnet_b4_baseline/figure_16_1_efficientnet_b4_confusion_matrix.png`
- `outputs/efficientnet_b4_baseline/figure_16_2_efficientnet_b4_roc_curve.png`

### What this checkpoint proves

This checkpoint proves that:

- `EfficientNet-B4` can be trained successfully in the same project pipeline;
- a larger EfficientNet variant does not automatically outperform `EfficientNet-B0`;
- `EfficientNet-B0` remains the strongest binary baseline candidate under the current experimental setup.

### Next step

My next step should be to treat `EfficientNet-B0` as the leading final binary model for now and move to:

- final binary-model selection;
- stronger final result presentation;
- and a thesis-ready summary that explains why `EfficientNet-B0` remains preferable to the larger `EfficientNet-B4`.


## Checkpoint numbering rule

I use this format for checkpoints:

- `DR-CP-001`
- `DR-CP-002`
- `DR-CP-003`

Where:

- `DR` = Diabetic Retinopathy project
- `CP` = Checkpoint
- the last number is the sequential milestone number
