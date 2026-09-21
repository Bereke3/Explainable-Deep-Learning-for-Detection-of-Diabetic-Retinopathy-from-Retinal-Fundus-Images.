# Thesis Evidence and Figure Map

This is an editing guide, not thesis prose. Paths below are relative to the
**machine-learning repository root**, not this Overleaf folder. They are not
required for LaTeX compilation and are not bundled in the Overleaf ZIP.

## Authoritative Project Notes

- `notes/CLEAN_BINARY_EXPERIMENT.md`: final protocol, settings, results, and limits.
- `notes/FINAL_BINARY_REVIEW.md`: historical evaluation and integrity audit.
- `notes/CHECKPOINTS.md`: experiment history; a writing aid, not a literature source.
- `data/splits/aptos2019_clean_binary_v1/protocol.json`: frozen split policy.
- `configs/efficientnet_b0_clean_binary.yaml`: final experiment configuration.

Use the experiment notes to locate the saved artifacts, then check the numbers
against the CSV/JSON files when filling tables. Do not turn checkpoints into
chronological thesis chapters.

## Planned Visuals and Tables

| LaTeX label | Planned material | Existing source or action |
| --- | --- | --- |
| `tab:related-work` | Comparison of published DR research | Read and verify primary studies; do not copy friends' references |
| `fig:workflow` | Study and evaluation diagram | New schematic based on the final protocol, not an invented system |
| `tab:dataset-splits` | Final split and class counts | `outputs/clean_binary_split/table_21_2_split_counts.csv` |
| `fig:binary-architecture` | Actual B0 classifier and Grad-CAM target | Check model construction and attribution implementation before drawing |
| `tab:training-settings` | Final settings | `outputs/efficientnet_b0_clean_binary/config_snapshot.json` and `training_summary.json` |
| `tab:final-metrics` | Final test metrics | `outputs/efficientnet_b0_clean_binary/table_22_2_test_metrics.csv` |
| `fig:confusion-matrix` | Final test confusion matrix | `outputs/efficientnet_b0_clean_binary/figure_22_1_confusion_matrix.png`; counts in `table_22_3_confusion_counts.csv` |
| `fig:roc` | Final ROC curve | `outputs/efficientnet_b0_clean_binary/figure_22_2_roc_curve.png` |
| `fig:learning-curves` | Five-epoch training/validation curves | `outputs/efficientnet_b0_clean_binary/figure_22_3_training_curves.png`; values in `table_22_5_training_history.csv` |
| `tab:final-errors` | All six final test errors | `outputs/efficientnet_b0_clean_binary/table_22_4_all_errors.csv` |
| `fig:gradcam-errors` | Input and both-logit Grad-CAM panels | Six `image_22_*.png` files in `outputs/efficientnet_b0_clean_binary/` |
| `tab:historical-binary` | Preliminary binary comparison | `notebooks/comparison/17_final_binary_summary.ipynb`, checked against saved runs; despite its name this notebook precedes the corrected experiment |
| `tab:historical-multiclass` | Exploratory grading experiments | `notebooks/multiclass/`, matching configurations and saved results |

The notebook figures retain their notebook numbering. Re-export thesis assets
without that embedded numbering; use descriptive filenames. New diagrams and
replacement figures have not been drawn in this scaffold.

## Final Experiment: Consistency Check

| Item | Value |
| --- | --- |
| Source dataset | APTOS 2019, 3,662 received image records |
| Excluded binary-label conflicts | 8 records |
| Omitted redundant exact copies | 124 records |
| Retained images | 3,530 |
| Train / validation / test | 2,822 / 354 / 354 |
| Test No DR / DR Present | 180 / 174 |
| Final architecture | EfficientNet-B0, fresh ImageNet initialization |
| Input / batch size / epochs | RGB 224 x 224 / 16 / 5 |
| Optimizer / learning rate / loss | Adam / 0.0003 / unweighted cross-entropy |
| Optional crop and CLAHE | Disabled |
| Seed / selected epoch | 42 / 5, selected by validation F1 |
| Accuracy / F1 / ROC-AUC | 0.983051 / 0.982558 / 0.995179 |
| TN / FP / FN / TP | 179 / 1 / 5 / 169 |

Full metric precision is available in the saved evaluation files. The 24 old
test images that matched train/validation images are not the total number of
records excluded during final preparation.

## Submission Checks

- Replace every writing note and every figure/table placeholder.
- Confirm the approved title and supervisor spelling without assuming changes.
- Reconcile every reported metric with its experiment, split, and CSV.
- Keep the final internal evaluation distinct from exploratory architecture selection.
- Explain heuristic image grouping without claiming verified patient separation.
- Check figure captions, labels, axis text, bibliography, and source attribution.
- Remove unsupported medical claims and distinguish attribution from diagnosis.
- Keep raw data, credentials, model binaries, and notebook caches out of Overleaf.
