"""Independently recompute a custody-held assessment report.

State slice: interpretability-lab-sandbox-v1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Keep direct invocation hermetic from a clean checkout.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from interp_lab.assessment import PredictionLock  # noqa: E402
from interp_lab.assessment_validator import validate_heldout_report  # noqa: E402
from scripts.run_assessment import _rows  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--effects", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--prediction-lock", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    effects_payload = json.loads(args.effects.read_text(encoding="utf-8"))
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    lock_payload = json.loads(args.prediction_lock.read_text(encoding="utf-8"))
    panel, _ = _rows(corpus)
    if set(lock_payload) != {"schema", "state_slice", "feature_index", "expected_sign", "source"}:
        raise ValueError("prediction lock schema mismatch")
    lock = PredictionLock(feature_index=int(lock_payload["feature_index"]), expected_sign=int(lock_payload["expected_sign"]), source=str(lock_payload["source"]))
    if lock_payload != lock.as_dict():
        raise ValueError("prediction lock payload is not canonical")
    result = validate_heldout_report(report, panel=panel, effects=effects_payload["effects"], prediction_lock=lock)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
