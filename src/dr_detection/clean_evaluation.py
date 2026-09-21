"""One frozen test evaluation for the corrected binary experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from .audit import file_sha256
from .clean_split import SPLIT_DIR, verify_clean_split
from .config import load_yaml_config
from .dataset import FundusImageDataset, load_fundus_records
from .inference import BinaryPredictor, save_explanation_panel
from .train import _compute_metrics

CONFIG_PATH = "configs/efficientnet_b0_clean_binary.yaml"


def evaluate_clean_model(root: str | Path) -> dict:
    root = Path(root).resolve()
    config = load_yaml_config(root / CONFIG_PATH)
    output = root / config["paths"]["output_dir"]
    result_path = output / "evaluation_summary.json"
    verify_clean_split(root, verify_images=True)
    training = json.loads((output / "training_summary.json").read_text())
    snapshot = json.loads((output / "config_snapshot.json").read_text())
    if config != snapshot:
        raise ValueError("The configuration changed after training. Use the frozen experiment configuration.")
    if training["test_evaluated"]:
        raise ValueError("Expected test evaluation to have been deferred during training.")
    for split, expected in training["split_csv_sha256"].items():
        if file_sha256(root / config["dataset"][f"{split}_csv_path"]) != expected:
            raise ValueError(f"{split} split changed after training.")
    model_path = root / config["paths"]["model_dir"] / "best_model.pt"
    checkpoint_hash = file_sha256(model_path)
    if checkpoint_hash != training["checkpoint_sha256"]:
        raise ValueError("The checkpoint changed after training.")
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Saved test results belong to a different model.")
        if saved["split_protocol_sha256"] != file_sha256(root / SPLIT_DIR / "protocol.json"):
            raise ValueError("Split protocol changed after the saved test evaluation.")
        for name, expected in saved["artifact_sha256"].items():
            if file_sha256(output / name) != expected:
                raise ValueError(f"Frozen evaluation artifact changed: {name}")
        print("Using the saved frozen test evaluation; no new test inference.")
        return saved

    predictor = BinaryPredictor(root, config_path=CONFIG_PATH)
    cfg = config["dataset"]
    frame = pd.read_csv(root / cfg["test_csv_path"], dtype={"phash": str})
    records = load_fundus_records(root / cfg["test_csv_path"], root / cfg["test_images_dir"])
    dataset = FundusImageDataset(records, predictor.transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    labels, predicted, score_rows = [], [], []
    loss_sum = 0.0
    for images, y, _ in tqdm(loader, desc="Frozen clean test evaluation", mininterval=5):
        with torch.no_grad():
            logits = predictor.model(images.to(predictor.device))
            loss_sum += torch.nn.functional.cross_entropy(logits, y.to(predictor.device), reduction="sum").item()
            scores = logits.softmax(dim=1).cpu().numpy()
        classes = scores.argmax(axis=1)
        labels.extend(y.tolist())
        predicted.extend(classes.tolist())
        score_rows.extend(scores.tolist())
    frame["true_label"] = labels
    frame["predicted_label"] = predicted
    frame["dr_probability"] = np.asarray(score_rows)[:, 1]
    frame["outcome"] = [{(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[(y, p)]
                        for y, p in zip(labels, predicted)]
    frame.to_csv(output / "table_22_1_test_predictions.csv", index=False)
    errors = frame.loc[frame.true_label != frame.predicted_label].sort_values(["outcome", "id_code"]).copy()
    errors["image_number"] = [f"Image 22.{i+1}" for i in range(len(errors))]
    errors["panel_file"] = [f"image_22_{i+1}_{row.id_code}_{row.outcome.lower()}.png" for i, row in enumerate(errors.itertuples())]
    errors.to_csv(output / "table_22_4_all_errors.csv", index=False)

    metrics = _compute_metrics(labels, predicted, score_rows)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    metrics["specificity"] = tn / (tn + fp)
    metrics["loss"] = loss_sum / len(labels)
    pd.DataFrame(list(metrics.items()), columns=["metric", "value"]).to_csv(output / "table_22_2_test_metrics.csv", index=False)
    counts = {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
    pd.DataFrame(list(counts.items()), columns=["outcome", "count"]).to_csv(output / "table_22_3_confusion_counts.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=["No DR", "DR Present"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Figure 22.1. Corrected internal test confusion matrix")
    fig.tight_layout()
    fig.savefig(output / "figure_22_1_confusion_matrix.png", dpi=200)
    plt.close(fig)
    fpr, tpr, _ = roc_curve(labels, frame.dr_probability)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#236c78", label=f"EfficientNet-B0 (AUC={metrics['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Figure 22.2. Corrected internal test ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "figure_22_2_roc_curve.png", dpi=200)
    plt.close(fig)
    history = json.loads((output / "training_history.json").read_text())
    history_frame = pd.DataFrame([
        {"epoch": item["epoch"], "seconds": item["seconds"],
         **{f"train_{k}": v for k, v in item["train"].items()},
         **{f"valid_{k}": v for k, v in item["valid"].items()}} for item in history
    ])
    history_frame.to_csv(output / "table_22_5_training_history.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["loss", "f1"]):
        for split in ["train", "valid"]:
            ax.plot(history_frame.epoch, history_frame[f"{split}_{metric}"], marker="o", label=split)
        ax.axvline(training["best_epoch"], color="gray", linestyle=":", label="Selected epoch")
        ax.set(xlabel="Epoch", ylabel=metric)
        ax.legend()
    fig.suptitle("Figure 22.3. Training history and validation-only checkpoint selection")
    fig.tight_layout()
    fig.savefig(output / "figure_22_3_training_curves.png", dpi=200)
    plt.close(fig)
    for row in errors.itertuples():
        save_explanation_panel(
            predictor, row.image_path, output / row.panel_file,
            title=f"{row.image_number}. Corrected test {row.outcome}: {row.id_code}",
            subtitle=f"Annotated binary={row.true_label}; predicted={row.predicted_label}; P(DR)={row.dr_probability:.4f}",
        )
    summary = {
        **predictor.provenance, "device": str(predictor.device), "metrics": metrics,
        "confusion_counts": counts, "test_size": len(frame), "best_epoch": training["best_epoch"],
        "best_validation_f1": training["best_validation_metric"],
        "split_protocol_sha256": file_sha256(root / SPLIT_DIR / "protocol.json"),
        "evaluation_scope": "Corrected internal repartition after historical exploration; not external or patient-verified validation.",
        "artifact_sha256": {p.name: file_sha256(p) for p in sorted(output.iterdir())
                            if p.name.startswith(("table_22_", "figure_22_", "image_22_"))},
    }
    result_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary
