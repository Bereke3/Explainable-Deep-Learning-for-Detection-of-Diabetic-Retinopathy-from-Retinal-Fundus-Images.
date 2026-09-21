# Corrected EfficientNet-B0 Binary Experiment

Date: 2026-09-18. Protocol: `aptos2019_clean_binary_v1`. Completed checkpoint: DR-CP-017.

I prepared this experiment after the integrity audit found exact image overlap and contradictory annotations in the supplied APTOS partitions. This document describes a new internal repartition and a fresh model initialization. The earlier experiments remain historical exploratory results.

## Data Preparation

I started from the audited 3,662 original image records and verified their file hashes again before preparing the new split. I quarantined all eight records belonging to four exact RGB groups with conflicting binary labels. Quarantine means exclusion from the experiment, not deletion of the original files or an attempt to select the medically correct annotation.

For the remaining exact groups, I retained the lexicographically first image identifier and omitted 124 redundant copies. This left 3,530 unique images. Different positive severity grades within an exact group did not cause a binary conflict; the original grade values are preserved in `original_grades`. This dataset is prepared for binary classification, not adjudicated five-class grading.

I recomputed perceptual-hash similarities over every pair of retained images, including pairs that previously belonged to the same split. A 63-bit DCT hash with Hamming distance at most four produced 2,344 candidate edges. I used the transitive connected components of this graph as conservative split groups. There were 2,507 groups, and the largest contained 179 images.

This grouping rule deliberately does not declare all visually similar images to be duplicates. It retains those images but prevents detected candidate relationships from crossing the new split. Some components may contain unrelated images because similarity is transitive; other true near-duplicates may be missed by this hash. I did not claim specialist adjudication or exhaustive visual review of all candidates.

I fixed a ten-fold `StratifiedGroupKFold`, with shuffle enabled and seed 42. Fold 0 became test, fold 1 became validation, and folds 2-9 became training. I did not select folds or seeds by training outcomes. Grouped stratification attempts to preserve class proportions while keeping groups disjoint; exact target fractions are not guaranteed. See the [official method documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

| New split | No DR | DR Present | Total |
|---|---:|---:|---:|
| Training | 1,436 | 1,386 | 2,822 |
| Validation | 180 | 174 | 354 |
| Test | 180 | 174 | 354 |
| Total | 1,796 | 1,734 | 3,530 |

I verified zero exact RGB overlap, zero shared similarity groups, and zero cross-split pHash candidate pairs at the specified cutoff. An independent all-pairs recomputation confirmed the candidate-pair result. CSV fingerprints and the splitting policy are frozen in `data/splits/aptos2019_clean_binary_v1/protocol.json`. This confirms content separation under the implemented checks, not patient independence.

## Frozen Training Protocol

| Setting | Value |
|---|---|
| Architecture | EfficientNet-B0, two output logits |
| Initialization | Fresh torchvision ImageNet weights and a new binary head |
| Input | RGB, 224 x 224, ImageNet normalization |
| Optional crop / CLAHE | Disabled |
| Training augmentation | Horizontal flip, rotation up to 10 degrees, mild color jitter |
| Epochs | 5 |
| Batch size | 16 |
| Optimizer | Adam |
| Learning rate | 0.0003 |
| Weight decay | 0 |
| Loss | Unweighted cross-entropy |
| Sampling | Ordinary shuffled training loader |
| Seed | 42 |
| DataLoader workers | 2 |
| Checkpoint selection | Highest validation F1; earliest epoch on a tie |
| Decision rule | Argmax; an exact score tie selects No DR |

I did not initialize from the previous retinal-model checkpoint. The corrected configuration writes to a separate model directory and refuses to overwrite an existing checkpoint. It saves a configuration snapshot, per-epoch training/validation metrics, runtime information, and hashes of the split CSVs and selected checkpoint.

Test evaluation is disabled in the training loop. After training finishes, the evaluator verifies the configuration, split, and model fingerprints before evaluating the selected checkpoint. Subsequent notebook runs reuse the saved test results rather than conducting new test inference. No threshold adjustment is performed using test labels.

## Results

I completed five epochs on Apple MPS, using Python 3.11.15 and PyTorch 2.12.1. The recorded training-loop duration was 562.8 seconds, approximately 9.4 minutes, excluding initial data verification and final evaluation. Epoch 5 had the highest validation F1 and was selected before test evaluation.

| Epoch | Train loss | Train F1 | Validation loss | Validation F1 |
|---|---:|---:|---:|---:|
| 1 | 0.163919 | 0.939776 | 0.064794 | 0.976608 |
| 2 | 0.080387 | 0.975504 | 0.069093 | 0.970588 |
| 3 | 0.067691 | 0.975557 | 0.054696 | 0.976744 |
| 4 | 0.075214 | 0.979415 | 0.053377 | 0.982558 |
| 5 | 0.034861 | 0.987752 | 0.041473 | 0.985507 |

Training metrics are collected during optimization on augmented images; validation uses the unaugmented evaluation transform. They are not measurements under identical input conditions.

The frozen selected checkpoint produced the following results on the new 354-image internal test set:

| Metric | Value |
|---|---:|
| Accuracy | 0.983051 |
| Precision | 0.994118 |
| Recall / sensitivity | 0.971264 |
| F1 | 0.982558 |
| ROC-AUC | 0.995179 |
| Specificity | 0.994444 |
| Cross-entropy loss | 0.094818 |
| TN / FP / FN / TP | 179 / 1 / 5 / 169 |

The model correctly classified 348 of 354 images. All five false negatives have an original grade-1 annotation (Mild DR). The single false positive has a grade-0 annotation. The complete error list is retained, rather than selecting only illustrative examples:

| Image | ID | Original grade | Outcome | DR score |
|---|---|---:|---|---:|
| Image 22.1 | 513b0a4651fa | 1 | FN | 0.042471 |
| Image 22.2 | 73881f55a3ec | 1 | FN | 0.314205 |
| Image 22.3 | a443c4fd489c | 1 | FN | 0.001047 |
| Image 22.4 | d06ccd0cf4b8 | 1 | FN | 0.383838 |
| Image 22.5 | d5b4705ac2ee | 1 | FN | 0.000025 |
| Image 22.6 | d9e58e4d8689 | 0 | FP | 0.999650 |

Each error has a saved panel containing the model input and Grad-CAM maps for both output logits. Several incorrect predictions have extreme scores, so a high softmax output should not be interpreted as independently verified diagnostic certainty. These observations are relative to the retained dataset annotations, which have not been medically re-adjudicated.

The selected model is `models/efficientnet_b0_clean_binary/best_model.pt`. Its SHA-256 is `c49e5aa6c31fd332a401dfbe5eca6bebb5632ab3bc28ac1a2f96163d62fbcb2b`. Configuration snapshots, histories, predictions, metrics, and image fingerprints are saved under `outputs/efficientnet_b0_clean_binary/`.

Figures 22.1-22.3 show the confusion matrix, ROC curve, and training curves. Tables 22.1-22.5 include all predictions, test metrics, confusion counts, errors, and per-epoch metrics. Notebooks 21 and 22 were executed successfully, with all 18 code cells completed. Twelve focused automated tests passed, including checks for quarantining conflicts, transitive grouping, deterministic splits, validation-only checkpoint selection, overwrite prevention, and reuse of a frozen test evaluation.

## Interpretation Limits

The corrected test set contains 354 images and differs from the historical 366-image set. A numerical difference from the historical 98.91% accuracy cannot be attributed solely to deduplication: the training composition, validation composition, test composition, and similarity grouping also changed.

The full collection was already used for exploratory experiments and architecture selection. This is therefore a corrected internal experiment, not a pristine external validation study. Patient independence remains unverified because the annotation files do not provide patient identifiers. Image-content checks do not prove the absence of all related acquisitions.

Other architectures have not been retrained under this protocol. Their historical scores cannot establish a ranking on the corrected split. Grad-CAM remains a coarse model attribution method, with no lesion masks or specialist review available here.

## Reproduction

The split-creation command is intentionally one-time and will not overwrite the frozen CSVs:

```bash
.venv/bin/python3 scripts/prepare_clean_split.py
```

Verification can be repeated without regenerating the split:

```bash
.venv/bin/python3 scripts/prepare_clean_split.py --verify-only
```

The training command refuses to overwrite an existing corrected checkpoint:

```bash
.venv/bin/python3 scripts/train_baseline.py --config configs/efficientnet_b0_clean_binary.yaml
```

Frozen evaluation and its figures:

```bash
.venv/bin/python3 scripts/evaluate_clean_binary.py
```

The notebooks `21_clean_binary_split.ipynb` and `22_efficientnet_b0_clean_binary_results.ipynb` expose the protocol, all numbered tables and figures, and the Grad-CAM panels for every test error. Images themselves remain in their original directories; the new CSVs contain portable project-relative paths.
