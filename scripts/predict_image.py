from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dr_detection.inference import BinaryPredictor, save_explanation_panel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Binary DR prediction and Grad-CAM for one image.")
    parser.add_argument("--image", required=True, help="Absolute path or path relative to the project.")
    parser.add_argument("--config", default="configs/efficientnet_b0_baseline.yaml", help="Configuration paired with the saved model.")
    parser.add_argument("--output-dir", default="outputs/single_image_demo")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    args = parser.parse_args()
    torch.set_num_threads(4)
    predictor = BinaryPredictor(PROJECT_ROOT, config_path=args.config, device=args.device)
    result = predictor.predict(args.image)
    destination = PROJECT_ROOT / args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "prediction.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    save_explanation_panel(
        predictor, args.image, destination / "image_20_1_single_image_prediction.png",
        "Image 20.1. Single-image prediction with class-specific Grad-CAM",
        f"Prediction: {result['predicted_class']}; P(DR)={result['dr_probability']:.4f} (uncalibrated)",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
