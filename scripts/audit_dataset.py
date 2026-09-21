from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dr_detection.audit import run_audit  # noqa: E402

if __name__ == "__main__":
    print(json.dumps(run_audit(PROJECT_ROOT), indent=2))
