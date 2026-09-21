from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dr_detection.config import load_yaml_config  # noqa: E402
from dr_detection.train import run_training  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline DR classifier.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config relative to the project root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(PROJECT_ROOT / args.config)
    results = run_training(config=config, project_root=PROJECT_ROOT)

    print("Training complete.")
    print(f"Split sizes: {results['split_sizes']}")
    print(f"Best model saved to: {results['best_model_path']}")
    if results["test_metrics"] is None:
        print("Test evaluation deferred by configuration until after checkpoint selection.")
    else:
        print(f"Test metrics: {results['test_metrics']}")


if __name__ == "__main__":
    main()
