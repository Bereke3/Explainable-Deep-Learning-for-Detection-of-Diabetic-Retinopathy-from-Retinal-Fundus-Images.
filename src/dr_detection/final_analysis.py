"""Re-evaluate the frozen B0 checkpoint and export every test error."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .audit import file_sha256
from .config import resolve_path
from .dataset import FundusImageDataset, load_fundus_records
from .inference import BinaryPredictor, save_explanation_panel


def run_error_analysis(project_root: str | Path, device: str | None = None) -> dict:
    root = Path(project_root).resolve()
    output = root / "outputs/final_error_analysis"
    output.mkdir(parents=True, exist_ok=True)
    audit_path = root / "outputs/data_audit/audit_summary.json"
    if not audit_path.is_file():
        raise FileNotFoundError("Run notebook 18 or scripts/audit_dataset.py first.")
    audit = json.loads(audit_path.read_text())
    predictor = BinaryPredictor(root, device=device)
    cfg = predictor.config["dataset"]
    for split, source in audit["sources"].items():
        if source["sha256"] != file_sha256(root / source["csv"]):
            raise ValueError(f"{split} CSV changed since the audit; rerun the audit first.")
    if audit["config_sha256"] != file_sha256(predictor.config_path):
        raise ValueError("Configuration changed since the audit; rerun the audit first.")
    if audit["invalid_labels"] or audit["missing_images"] or audit["unreadable_images"]:
        raise ValueError("Resolve invalid labels or unreadable/missing images before inference.")
    records = load_fundus_records(
        resolve_path(root, cfg["test_csv_path"]), resolve_path(root, cfg["test_images_dir"]),
        cfg.get("image_extension", ".png"), binary_classification=False,
    ).rename(columns={"label": "diagnosis"})
    records["label"] = records["diagnosis"].gt(0).astype(int)
    dataset = FundusImageDataset(records, transform=predictor.transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    predicted, all_scores = [], []
    for inputs, _, _ in tqdm(loader, desc="Frozen B0 test inference"):
        labels, scores = predictor.predict_tensor(inputs)
        predicted.extend(labels.tolist())
        all_scores.extend(scores[:, 1].tolist())
    predictions = records.rename(columns={"label": "true_label"}).copy()
    predictions["id_code"] = predictions["image_path"].map(lambda p: Path(p).stem)
    predictions["image_path"] = predictions["image_path"].map(lambda p: str(Path(p).relative_to(root)))
    predictions["predicted_label"] = predicted
    predictions["dr_probability"] = all_scores
    predictions["outcome"] = [
        { (0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP" }[(int(y), int(p))]
        for y, p in zip(predictions["true_label"], predicted)
    ]
    predictions["exact_match_in_train_or_valid"] = predictions["id_code"].isin(audit["test_images_matching_train_or_valid"])
    predictions.to_csv(output / "table_19_1_all_test_predictions.csv", index=False)
    errors = predictions.loc[predictions["true_label"] != predictions["predicted_label"]].copy()
    errors = errors.sort_values(["outcome", "id_code"]).reset_index(drop=True)
    pairs = pd.read_csv(root / "outputs/data_audit/table_18_4_exact_duplicate_pairs.csv")
    counterpart_rows = []
    for pair in pairs.to_dict("records"):
        for side, other in [("a", "b"), ("b", "a")]:
            if pair[f"split_{side}"] == "test" and pair[f"split_{other}"] != "test" and pair[f"id_{side}"] in set(errors["id_code"]):
                counterpart_rows.append({
                    "test_id": pair[f"id_{side}"], "test_grade": pair[f"diagnosis_{side}"],
                    "counterpart_split": pair[f"split_{other}"], "counterpart_id": pair[f"id_{other}"],
                    "counterpart_grade": pair[f"diagnosis_{other}"],
                    "binary_label_conflict": (pair[f"diagnosis_{side}"] > 0) != (pair[f"diagnosis_{other}"] > 0),
                })
    counterparts = pd.DataFrame(counterpart_rows, columns=[
        "test_id", "test_grade", "counterpart_split", "counterpart_id", "counterpart_grade", "binary_label_conflict",
    ])
    counterparts.to_csv(output / "table_19_5_error_duplicate_counterparts.csv", index=False)
    conflict_ids = set(counterparts.loc[counterparts["binary_label_conflict"] == True, "test_id"])
    errors["conflicting_binary_label_in_train_or_valid"] = errors["id_code"].isin(conflict_ids)
    errors["image_number"] = [f"Image 19.{i + 1}" for i in range(len(errors))]
    errors["panel_file"] = [f"image_19_{i+1}_{row.id_code}_{row.outcome.lower()}.png" for i, row in enumerate(errors.itertuples())]
    errors.to_csv(output / "table_19_2_all_errors.csv", index=False)

    y = predictions["true_label"]
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y, predicted), "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0), "f1_score": f1_score(y, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y, all_scores), "specificity": float(tn / (tn + fp)),
    }
    pd.DataFrame([metrics]).to_csv(output / "table_19_3_reproduced_metrics.csv", index=False)
    by_grade = predictions.assign(is_error=predictions["true_label"] != predictions["predicted_label"]).groupby("diagnosis").agg(
        support=("id_code", "size"), errors=("is_error", "sum"),
    )
    by_grade["error_rate"] = by_grade["errors"] / by_grade["support"]
    by_grade.to_csv(output / "table_19_4_errors_by_original_grade.csv")

    reference = pd.read_csv(root / "outputs/efficientnet_b0_baseline/table_9_1_efficientnet_b0_metrics.csv")
    # Historical table is rounded; it may be either wide or metric/value format.
    if {"metric", "value"}.issubset(reference.columns):
        expected = dict(zip(reference["metric"], reference["value"]))
    else:
        expected = reference.iloc[0].to_dict()
    expected = {"f1_score" if k == "f1" else k: float(v) for k, v in expected.items() if k != "model"}
    compared = sorted(set(metrics) & set(expected))
    matches = bool(compared) and all(np.isclose(metrics[k], expected[k], atol=0.0001, rtol=0) for k in compared)

    for row in errors.itertuples():
        save_explanation_panel(
            predictor, row.image_path, output / row.panel_file,
            title=f"{row.image_number}. {row.outcome}: {row.id_code}",
            subtitle=f"Original grade={row.diagnosis}; true binary={row.true_label}; predicted={row.predicted_label}; P(DR)={row.dr_probability:.4f}",
        )
    summary = {
        **predictor.provenance, "device": str(predictor.device), "torch_version": str(torch.__version__),
        "test_size": len(predictions), "metrics": metrics,
        "confusion_counts": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "historical_metrics_reproduced": matches, "compared_metrics": compared,
        "exact_overlap_test_images": len(audit["test_images_matching_train_or_valid"]),
        "errors_with_conflicting_duplicate_labels": sorted(conflict_ids),
        "audit_status": audit["status"], "evaluation_scope": "Historical internal test split; not a new external validation.",
        "gradcam_scope": "Class-logit attribution, independently normalized per target; no lesion-level ground truth.",
    }
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
