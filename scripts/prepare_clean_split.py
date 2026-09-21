from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dr_detection.clean_split import prepare_clean_split, verify_clean_split  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create or verify the frozen cleaned binary split.")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = verify_clean_split(PROJECT_ROOT, verify_images=True) if args.verify_only else prepare_clean_split(PROJECT_ROOT)
    print(json.dumps(result, indent=2))
