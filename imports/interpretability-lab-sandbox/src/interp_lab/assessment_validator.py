"""Independent aggregate recomputation for held-out assessment reports.

State slice: interpretability-lab-sandbox-v1.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from .assessment import AssessmentPanel, FROZEN_STATISTICS, PredictionLock, REQUIRED_ARMS, STATE_SLICE
from .canonical import digest


class AssessmentValidationError(ValueError):
    """Raised when an assessment report cannot be independently reproduced."""


def _close(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    except (TypeError, ValueError):
        return left == right


def validate_heldout_report(
    report: Mapping[str, Any],
    *,
    panel: AssessmentPanel,
    effects: Mapping[str, Mapping[str, float]],
    prediction_lock: PredictionLock,
) -> dict[str, Any]:
    """Recompute aggregate values without trusting the report's aggregates."""
    if report.get("state_slice") != STATE_SLICE:
        raise AssessmentValidationError("state slice mismatch")
    recorded_digest = report.get("report_digest")
    if not isinstance(recorded_digest, str):
        raise AssessmentValidationError("report digest is missing")
    body = {key: value for key, value in report.items() if key != "report_digest"}
    if digest(body) != recorded_digest:
        raise AssessmentValidationError("report digest mismatch")
    if set(effects) != set(panel.assessment_rows):
        raise AssessmentValidationError("effect roster mismatch")
    expected_lock_digest = digest(prediction_lock.as_dict())
    if report.get("prediction_lock_digest") != expected_lock_digest:
        raise AssessmentValidationError("prediction lock mismatch")
    if report.get("statistics") != FROZEN_STATISTICS:
        raise AssessmentValidationError("statistical design mismatch")
    trial_values: list[dict[str, float]] = []
    for row in panel.assessment_rows:
        trial = effects[row]
        if set(trial) != set(REQUIRED_ARMS):
            raise AssessmentValidationError(f"{row} arm roster mismatch")
        normalized: dict[str, float] = {}
        for arm in REQUIRED_ARMS:
            value = float(trial[arm])
            if not math.isfinite(value):
                raise AssessmentValidationError(f"{row}.{arm} is nonfinite")
            normalized[arm] = value
        trial_values.append(normalized)
    interventions = [trial["intervention"] for trial in trial_values]
    scrubs = [abs(trial["scrub"]) for trial in trial_values]
    exact = [abs(trial["exact_copy"]) for trial in trial_values]
    controls = [abs(trial[arm]) for trial in trial_values for arm in REQUIRED_ARMS[4:]]
    sign_agreement = sum(1 for value in interventions if (1 if value > 0 else -1 if value < 0 else 0) == prediction_lock.expected_sign) / len(interventions)
    recomputed = {
        "assessment_rows": len(trial_values),
        "intervention_mean_signed": sum(interventions) / len(interventions),
        "intervention_mean_abs": sum(abs(value) for value in interventions) / len(interventions),
        "intervention_sign_agreement": sign_agreement,
        "scrub_mean_abs": sum(scrubs) / len(scrubs),
        "exact_copy_max_abs": max(exact),
        "control_max_abs": max(controls),
        "zero_attrition": True,
    }
    recorded = report.get("aggregate")
    if not isinstance(recorded, Mapping) or set(recorded) != set(recomputed):
        raise AssessmentValidationError("aggregate schema mismatch")
    for key, value in recomputed.items():
        if not _close(recorded.get(key), value):
            raise AssessmentValidationError(f"aggregate mismatch: {key}")
    thresholds = report.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise AssessmentValidationError("thresholds are missing")
    required_thresholds = {
        "effect_min_abs",
        "sign_agreement_min",
        "scrub_min_abs",
        "exact_copy_max_abs",
        "control_max_abs",
        "zero_attrition",
    }
    if set(thresholds) != required_thresholds:
        raise AssessmentValidationError("threshold schema mismatch")
    for name in required_thresholds - {"zero_attrition"}:
        try:
            if not math.isfinite(float(thresholds[name])):
                raise AssessmentValidationError(f"threshold is nonfinite: {name}")
        except (TypeError, ValueError) as error:
            raise AssessmentValidationError(f"threshold is invalid: {name}") from error
    if thresholds["zero_attrition"] is not True:
        raise AssessmentValidationError("zero-attrition requirement is not frozen")
    expected_gates = {
        "complete_assessment_rows": recomputed["assessment_rows"] == len(panel.assessment_rows),
        "prediction_lock_bound": report.get("prediction_lock_digest") == expected_lock_digest,
        "heldout_intervention_effect": recomputed["intervention_mean_abs"] >= float(thresholds["effect_min_abs"]),
        "heldout_prediction_sign": recomputed["intervention_sign_agreement"] >= float(thresholds["sign_agreement_min"]),
        "heldout_causal_scrubbing": recomputed["scrub_mean_abs"] >= float(thresholds["scrub_min_abs"]),
        "exact_copy_no_op": recomputed["exact_copy_max_abs"] <= float(thresholds["exact_copy_max_abs"]),
        "control_finiteness": recomputed["control_max_abs"] <= float(thresholds["control_max_abs"]),
    }
    gates = report.get("gates")
    if (
        not isinstance(gates, list)
        or len(gates) != len(expected_gates)
        or any(not isinstance(gate, Mapping) for gate in gates)
        or len({gate.get("name") for gate in gates}) != len(expected_gates)
        or {gate.get("name") for gate in gates} != set(expected_gates)
    ):
        raise AssessmentValidationError("gate schema mismatch")
    for gate in gates:
        if not isinstance(gate, Mapping) or gate.get("passed") is not expected_gates.get(gate.get("name")):
            raise AssessmentValidationError("gate result mismatch")
    if report.get("all_gates_pass") is not all(expected_gates.values()):
        raise AssessmentValidationError("all_gates_pass mismatch")
    return {"status": "INDEPENDENT_AGGREGATE_VALIDATED", "report_digest": recorded_digest, "aggregate": recomputed}
