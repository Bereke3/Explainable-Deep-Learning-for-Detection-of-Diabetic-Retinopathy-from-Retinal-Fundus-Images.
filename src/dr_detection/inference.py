"""Inference and explanations using the saved binary model's evaluation transform."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .audit import file_sha256
from .config import load_yaml_config, resolve_path
from .dataset import default_transforms
from .explainability import get_default_target_layer
from .train import build_model

LABELS = {0: "No DR", 1: "DR Present"}


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BinaryPredictor:
    def __init__(
        self, project_root: str | Path,
        config_path: str | Path = "configs/efficientnet_b0_baseline.yaml",
        device: str | None = None,
    ) -> None:
        self.root = Path(project_root).resolve()
        self.config_path = resolve_path(self.root, config_path)
        self.config = load_yaml_config(self.config_path)
        cfg = self.config["dataset"]
        if not cfg.get("binary_classification", False):
            raise ValueError("This predictor requires a binary classification configuration.")
        self.checkpoint_path = resolve_path(self.root, self.config["paths"]["model_dir"]) / "best_model.pt"
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Saved model not found: {self.checkpoint_path}")
        self.device = torch.device(device) if device else select_device()
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self.model_name = checkpoint["model_name"]
        if checkpoint["num_classes"] != 2 or self.model_name != self.config["training"]["model_name"]:
            raise ValueError("Checkpoint architecture/classes do not match the binary configuration.")
        self.model = build_model(self.model_name, num_classes=2, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()
        self.transform = default_transforms(
            image_size=cfg.get("image_size", 224), train=False,
            preprocessing_cfg=cfg.get("preprocessing"),
        )
        self.target_layer = get_default_target_layer(self.model, self.model_name)
        self.provenance = {
            "model_name": self.model_name,
            "checkpoint_sha256": file_sha256(self.checkpoint_path),
            "config_sha256": file_sha256(self.config_path),
            "decision_rule": "argmax of two softmax scores; exact tie selects No DR",
            "score_note": "Uncalibrated model scores, not a clinical probability of disease.",
            "scope": "Research prototype; no automatic fundus-quality or out-of-distribution gate.",
        }

    def prepare(self, image_path: str | Path) -> torch.Tensor:
        path = resolve_path(self.root, image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    def predict_tensor(self, inputs: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            scores = self.model(inputs.to(self.device)).softmax(dim=1)
        return scores.argmax(dim=1).cpu().numpy(), scores.cpu().numpy()

    def predict(self, image_path: str | Path) -> dict:
        labels, scores = self.predict_tensor(self.prepare(image_path))
        predicted = int(labels[0])
        return {
            "image_path": str(image_path), "predicted_label": predicted,
            "predicted_class": LABELS[predicted], "no_dr_probability": float(scores[0, 0]),
            "dr_probability": float(scores[0, 1]), "predicted_class_score": float(scores[0, predicted]),
            **self.provenance,
        }

    def explain(self, image_path: str | Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        inputs = self.prepare(image_path)
        normalized = inputs[0].detach().cpu().permute(1, 2, 0).numpy()
        # Denormalize the actual model input, including precisely the same resize/crop.
        rgb = np.clip(normalized * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1).astype(np.float32)
        overlays = {}
        with torch.enable_grad(), GradCAM(model=self.model, target_layers=[self.target_layer]) as cam:
            for target in LABELS:
                heatmap = cam(input_tensor=inputs, targets=[ClassifierOutputTarget(target)])[0]
                if not np.isfinite(heatmap).all():
                    raise RuntimeError("Grad-CAM produced non-finite values.")
                overlays[target] = show_cam_on_image(rgb, heatmap, use_rgb=True)
        self.model.zero_grad(set_to_none=True)
        return rgb, overlays


def save_explanation_panel(
    predictor: BinaryPredictor, image_path: str | Path, output_path: str | Path,
    title: str, subtitle: str = "",
) -> None:
    rgb, overlays = predictor.explain(image_path)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    panels = [rgb, overlays[0], overlays[1]]
    titles = ["Model input", "Grad-CAM target: No DR", "Grad-CAM target: DR Present"]
    for ax, panel, label in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title + ("\n" + subtitle if subtitle else ""), fontsize=12)
    fig.text(0.5, 0.02, "Maps are normalized separately. They are not lesion segmentations.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.88))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
