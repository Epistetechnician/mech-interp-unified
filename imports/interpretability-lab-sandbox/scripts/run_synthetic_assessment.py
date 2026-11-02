"""Run the assessment and continual-learning harness on a synthetic fixture.

State slice: interpretability-lab-sandbox-v1.

This script exercises the complete estimator and validator path without model
weights, external data, provider calls, or raw trace retention. Its output is
engineering evidence only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Keep direct invocation hermetic from a clean checkout.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interp_lab.adapters import SyntheticFeatureAdapter, SyntheticModel  # noqa: E402
from interp_lab.assessment import AssessmentPanel, PredictionLock, evaluate_heldout_effects  # noqa: E402
from interp_lab.assessment_validator import validate_heldout_report  # noqa: E402
from interp_lab.continual import run_matched_continual_assessment  # noqa: E402
from interp_lab.continual_validator import validate_matched_continual_report  # noqa: E402


ROWS = {
    "fit": tuple(f"fresh-fit-family-{index:02d}" for index in range(8)),
    "tune": tuple(f"fresh-tune-family-{index:02d}" for index in range(8)),
    "assessment": tuple(f"fresh-assessment-family-{index:02d}" for index in range(8)),
}


def synthetic_effects(rows: tuple[str, ...]) -> dict[str, dict[str, float]]:
    model = SyntheticModel()
    feature = SyntheticFeatureAdapter()
    activations = {row: model._activation(row) for row in rows}
    donor_rows = rows[1:] + rows[:1]

    def score(values: tuple[float, ...]) -> float:
        return sum(weight * value for weight, value in zip(model.weights, values))

    output: dict[str, dict[str, float]] = {}
    for row, donor_row in zip(rows, donor_rows):
        original = activations[row]
        donor = activations[donor_row]
        baseline = score(original)
        intervention = feature.intervene(original, 0, original[0] + 0.25)
        scrub = feature.intervene(original, 0, 0.0)
        constant = feature.intervene(original, 0, 0.125)
        output[row] = {
            "intervention": score(intervention) - baseline,
            "scrub": score(scrub) - baseline,
            "exact_copy": score(feature.intervene(original, 0, original[0])) - baseline,
            "activation_only": score(intervention) - baseline,
            "text_only": score(donor) - baseline,
            "shuffled": score(donor) - baseline,
            "constant": score(constant) - baseline,
            "matched": score(donor) - baseline,
        }
    return output


def main() -> None:
    panel = AssessmentPanel(ROWS)
    lock = PredictionLock(feature_index=0, expected_sign=1)
    effects = synthetic_effects(panel.assessment_rows)
    report = evaluate_heldout_effects(panel=panel, effects=effects, prediction_lock=lock, thresholds={})
    validation = validate_heldout_report(report, panel=panel, effects=effects, prediction_lock=lock)
    continual = run_matched_continual_assessment()
    continual_validation = validate_matched_continual_report(continual)
    print(json.dumps({"assessment": report, "aggregate_validation": validation, "continual_learning": continual, "continual_validation": continual_validation}, sort_keys=True))


if __name__ == "__main__":
    main()
