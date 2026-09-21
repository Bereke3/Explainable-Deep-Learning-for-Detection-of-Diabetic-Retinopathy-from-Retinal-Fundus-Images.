# Binary Pipeline: Integrity Audit, Error Analysis, and Demonstration

Date: 2026-09-18. Checkpoint: DR-CP-016.

Follow-up completed: [the corrected binary experiment](CLEAN_BINARY_EXPERIMENT.md), recorded in DR-CP-017, implements the new split and fresh training requested below. It achieved accuracy 0.983051 and F1 0.982558 on a different 354-image internal test set. This document preserves the historical audit and its original conclusions; use the follow-up for the corrected model's results.

This document records my finalization work on the historical EfficientNet-B0 baseline. The audit changes how I interpret the earlier model comparison: the implementation and historical scores are reproducible, but cross-split image duplication prevents treating them as a clean final evaluation.

## Dataset Integrity and Evaluation Protocol

I audited all 3,662 APTOS images in the supplied CSV splits: 2,930 training, 366 validation, and 366 test images. I checked file availability, full RGB decoding, original grade validity, repeated image identifiers, file SHA-256 hashes, and decoded RGB hashes including image dimensions. I also generated similarity candidates from a 63-bit DCT perceptual hash using a Hamming-distance cutoff of four. This cutoff is a retrieval heuristic, not a validated duplicate classifier.

Table 18.6 summarizes the audit:

| Check | Observed result |
|---|---:|
| Missing files | 0 |
| Unreadable images | 0 |
| Invalid severity grades | 0 |
| Repeated image identifiers | 0 |
| Exact duplicate groups across the full dataset | 123 |
| Images belonging to those groups | 251 |
| Distinct decoded RGB images | 3,534 |
| Exact duplicate pairs across the full dataset | 134 |
| Exact duplicate pairs between different splits | 47 |
| Test images identical to a training or validation image | 24 |
| Exact pairs with conflicting five-grade annotations | 32 |
| Exact pairs with conflicting binary annotations | 4 |
| Cross-split pairs with conflicting binary annotations | 2 |
| Additional cross-split perceptual similarity candidates | 747 |

All 134 exact pairs also have identical file bytes. Counts of pairs are not counts of unique images. Of the 47 cross-split pairs, 23 connect training with test, 22 connect training with validation, and two connect validation with test. The 24 affected test images represent 6.56% of the test set.

These results show why disjoint filenames alone were insufficient. Images were duplicated under different identifiers. I retained the original files and annotations to preserve the experiment history. I have not adjudicated the 747 similarity candidates; notebook 18 displays six examples, while the full candidate list is saved for review. No patient identifiers are available in these CSVs, so I cannot establish patient-level independence.

The training implementation selects the best epoch using validation F1 and evaluates the saved checkpoint on test. However, I also consulted test results while comparing architectures and experimental variants. Therefore, the historical test set was not reserved as an untouched final assessment. The duplicate overlap and repeated test consultation are distinct limitations. Data leakage can make evaluation optimistic; its numerical effect cannot be determined merely from the number of overlapping images. See the [scikit-learn guidance on data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

## Reproduced Model and Complete Error Analysis

I loaded the saved EfficientNet-B0 checkpoint without additional training or threshold tuning. The original binary setup used ImageNet initialization, a 224 x 224 RGB input, batch size 16, five epochs, Adam with learning rate 0.0003 and no configured weight decay, and seed 42. Optional border cropping and CLAHE were disabled for this model. Training augmentation comprised horizontal flips, rotations up to 10 degrees, and mild color jitter. Evaluation used resizing and ImageNet mean/standard-deviation normalization without augmentation. The predicted class was the argmax of the two softmax outputs.

I reproduced the following historical test metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.989071 |
| Precision | 0.993939 |
| Recall / sensitivity | 0.982036 |
| F1 | 0.987952 |
| ROC-AUC | 0.998315 |
| Specificity | 0.994975 |
| TN / FP / FN / TP | 198 / 1 / 3 / 164 |

These numbers describe the original, overlap-affected test split. They are not estimates from a newly cleaned or external test set, and I do not claim that EfficientNet-B0 is statistically superior to the other architectures from this single comparison.

I exported every test prediction and examined all four incorrect predictions against the original annotations. Table 19.2 contains the complete error list, and Table 19.5 records exact training/validation counterparts of those errors.

| Panel | Image ID | Original grade | Prediction | DR softmax score |
|---|---|---:|---|---:|
| Image 19.1 | ea15a290eb96 | 1 | No DR | 0.338645 |
| Image 19.2 | eed4afc8ec83 | 1 | No DR | 0.000726 |
| Image 19.3 | f9aa35187bf3 | 2 | No DR | 0.214598 |
| Image 19.4 | eb6b1f1c09db | 0 | DR Present | 0.999653 |

The original test-grade supports were 199, 30, 87, 17, and 33 for grades 0 through 4. The three false negatives comprise two grade-1 images and one grade-2 image. The absence of observed errors among the grade-3 and grade-4 images does not establish perfect performance for severe disease on new data.

For each case I generated Grad-CAM maps for both class logits using the last EfficientNet feature block. I denormalized the actual model input for the overlay, ensuring that the displayed spatial coordinates match the image supplied to the model. Each heatmap was normalized independently. Grad-CAM provides coarse class-specific attribution, as described by [Selvaraju et al.](https://arxiv.org/abs/1610.02391); it does not provide a lesion segmentation or a calibrated importance scale shared across images.

My observations are deliberately limited to visible image structure and annotation evidence:

- **Image 19.1:** the test image is an exact copy of training image `1c9c583c10bf`. The training annotation is grade 0, while the test annotation is grade 1. Thus, this false negative occurs in a group with contradictory binary annotations. The No DR map emphasizes a central region and the DR map emphasizes right-edge regions. I cannot determine the correct medical label or attribute the error solely to missed disease.
- **Image 19.2:** the model assigns an extremely low DR score despite a grade-1 annotation. The No DR map covers a broad central/lower-right area; the DR map highlights a smaller upper-right region. This is an example of a highly confident incorrect prediction relative to the annotation. It does not establish that the label is wrong or quantify model calibration.
- **Image 19.3:** the original grade is 2, while the predicted binary class is No DR. The No DR map emphasizes a lower-central region, whereas the DR map highlights a region to the right. The coarse localization does not establish whether particular retinal lesions contributed to the prediction.
- **Image 19.4:** the model assigns a DR score of approximately 0.99965 to an image labeled grade 0. The DR heatmap emphasizes broad peripheral regions, including the upper-right and lower-left areas. This suggests a question for further investigation about boundary or acquisition cues; it is not proof of a causal shortcut.

I did not have lesion masks or an ophthalmologist's adjudication. I therefore describe these observations as exploratory explainability and error analysis, rather than medical verification of model attention. I have not changed labels based on Grad-CAM.

I reproduced the same classification counts and headline metrics on CPU and Apple MPS. Individual scores can differ in the final decimal places between devices. I also executed all 23 code cells across notebooks 18-20 without errors and passed seven focused tests covering duplicate detection, missing/corrupt inputs, evaluation transforms, binary-configuration validation, and Grad-CAM hook cleanup.

## Single-Image Demonstration and Remaining Work

I implemented a reproducible inference workflow that loads one retinal image, applies the same evaluation transform, returns the binary prediction and both softmax scores, and saves two class-specific Grad-CAM overlays. Notebook 20 provides an editable image path and a default historical test example. A command-line equivalent is available in `scripts/predict_image.py`.

Outputs include Table 20.1, Image 20.1, and a JSON record containing the checkpoint and configuration SHA-256 fingerprints. The demonstration does not require the input image's label. It uses the frozen model and does not train on the demonstration image. The displayed scores are uncalibrated model outputs, not individual clinical risk estimates. The prototype has no automatic fundus-image validity, quality, or out-of-distribution screening.

For my thesis, I can now describe the implemented pipeline, reproduce the exploratory benchmark, discuss the complete error analysis, and demonstrate inference. Before presenting final performance claims, I still need to:

1. Group exact duplicates and reviewed near-duplicates before constructing an image-content-disjoint split. Patient independence remains unverified without patient metadata.
2. Obtain an authoritative resolution for contradictory-label groups, or quarantine unresolved groups according to a rule fixed before training. I should not choose labels based on which value improves a score.
3. Freeze the evaluation protocol and retrain the chosen model from its original initialization on the corrected training split. Select checkpoints and any settings using validation only. Moving files after training or deleting only overlapping test rows does not undo the existing training/validation exposure.
4. Report the corrected experiment separately from these historical results, retain its limitations, and avoid calling it external validation. External evaluation and repeated-seed experiments have not been completed here.

The completed audit is itself an important experimental finding. My current defensible conclusion is that the historical classification and explanation workflow is reproducible, while a content-level integrity check revealed limitations that the original filename-based dataset check did not detect.

## Artifact Index

- `notebooks/validation/18_dataset_integrity_audit.ipynb`: full audit, Tables 18.1-18.7, Figure 18.1, Images 18.1-18.2.
- `notebooks/transfer_learning/19_efficientnet_b0_error_analysis.ipynb`: all test predictions, Tables 19.1-19.5, Images 19.1-19.4.
- `notebooks/transfer_learning/20_single_image_demo.ipynb`: Table 20.1, Image 20.1, prediction JSON.
- `outputs/data_audit/`: hashes, complete pair lists, audit summary, and figures.
- `outputs/final_error_analysis/`: reproduced scores, every error, original-grade analysis, and Grad-CAM panels.
- `outputs/single_image_demo/`: saved demonstration result.
