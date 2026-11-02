"""Independently validate a matched continual-learning aggregate.

State slice: interpretability-lab-sandbox-v1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interp_lab.continual_validator import validate_matched_continual_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    print(json.dumps(validate_matched_continual_report(report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
