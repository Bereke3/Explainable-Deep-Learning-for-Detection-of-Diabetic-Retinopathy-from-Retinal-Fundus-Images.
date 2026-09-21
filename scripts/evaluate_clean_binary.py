from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dr_detection.clean_evaluation import evaluate_clean_model  # noqa: E402

if __name__ == "__main__":
    torch.set_num_threads(4)
    print(json.dumps(evaluate_clean_model(PROJECT_ROOT), indent=2))
