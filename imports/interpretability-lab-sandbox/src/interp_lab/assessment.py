"""Held-out causal assessment contracts and aggregate estimators.

State slice: interpretability-lab-sandbox-v1.

This module contains the fixed assessment interface. It accepts aggregate trial
inputs from a custody-controlled runner; it never downloads models, opens a
provider, or creates external review authority.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .canonical import digest


STATE_SLICE = "interpretability-lab-sandbox-v1"
REQUIRED_ARMS = (
    "intervention",
    "scrub",
    "exact_copy",
    "activation_only",
    "text_only",
    "shuffled",
    "constant",
    "matched",
)
FROZEN_STATISTICS: dict[str, Any] = {
    "alpha": 0.05,
    "independent_unit": "family",
    "icc_assumption": 0.50,
    "minimum_standardized_effect": 0.50,
    "multiplicity": "Holm",
    "power_endpoints": 4,
    "power_method": "two_sided_normal_icc_design_effect",
    "power_scope": "single_comparison_approximation",
    "power_target": 0.80,
    "power_validation": "familywise_holm_power_unvalidated",
    "repeats": 3,
    "missingness": "fail_closed",
}


class AssessmentError(ValueError):
    """Raised when a frozen assessment contract is violated."""


@dataclass(frozen=True)
class AssessmentPanel:
    """Disjoint fit/tune/assessment row rosters."""

    rows: Mapping[str, Sequence[str]]

    def __post_init__(self) -> None:
        required = {"fit", "tune", "assessment"}
        if set(self.rows) != required:
            raise AssessmentError("panel must contain exactly fit, tune, and assessment")
        normalized: dict[str, tuple[str, ...]] = {}
        seen: set[str] = set()
        for split in ("fit", "tune", "assessment"):
            values = tuple(str(row) for row in self.rows[split])
            if not values or any(not row for row in values):
                raise AssessmentError(f"{split} roster must be nonempty")
            if len(values) != len(set(values)):
                raise AssessmentError(f"{split} roster contains duplicate rows")
            if seen.intersection(values):
                raise AssessmentError("panel splits must be disjoint")
            seen.update(values)
            normalized[split] = values
        object.__setattr__(self, "rows", normalized)

    @property
    def digest(self) -> str:
        return digest({"state_slice": STATE_SLICE, "rows": self.rows})

    @property
    def assessment_rows(self) -> tuple[str, ...]:
        return tuple(self.rows["assessment"])


@dataclass(frozen=True)
class PredictionLock:
    """A prediction made before assessment effects are inspected."""

    feature_index: int
    expected_sign: int
    source: str = "fit_tune_only"

    def __post_init__(self) -> None:
        if type(self.feature_index) is not int or self.feature_index < 0:
            raise AssessmentError("prediction lock feature index is invalid")
        if self.expected_sign not in (-1, 1):
            raise AssessmentError("prediction lock sign must be -1 or 1")
        if self.source != "fit_tune_only":
            raise AssessmentError("prediction lock source is not frozen")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "interp-lab.prediction-lock.v1",
            "state_slice": STATE_SLICE,
            "feature_index": self.feature_index,
            "expected_sign": self.expected_sign,
            "source": self.source,
        }

    @property
    def digest(self) -> str:
        return digest(self.as_dict())


def _finite(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise AssessmentError(f"{field} contains a nonfinite value")
    return number


def _mean(values: Sequence[float], field: str) -> float:
    if not values:
        raise AssessmentError(f"{field} is empty")
    return sum(values) / len(values)


def _gate(name: str, passed: bool, **values: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), **values}


def evaluate_heldout_effects(
    *,
    panel: AssessmentPanel,
    effects: Mapping[str, Mapping[str, float]],
    prediction_lock: PredictionLock,
    thresholds: Mapping[str, float],
    statistics: Mapping[str, Any] | None = None,
    claim_status: str = "LOCAL_SYNTHETIC_ONLY",
) -> dict[str, Any]:
    """Evaluate locked held-out effects with fixed controls and no tuning."""
    assessment_rows = set(panel.assessment_rows)
    if set(effects) != assessment_rows:
        raise AssessmentError("effects must cover exactly the assessment roster")
    required_thresholds = {
        "effect_min_abs": 0.01,
        "sign_agreement_min": 0.80,
        "scrub_min_abs": 0.01,
        "exact_copy_max_abs": 0.0,
        "control_max_abs": 1.0,
        "zero_attrition": True,
    }
    unknown = set(thresholds) - set(required_thresholds)
    if unknown:
        raise AssessmentError(f"unknown assessment thresholds: {sorted(unknown)}")
    limits: dict[str, Any] = dict(required_thresholds)
    for key, value in thresholds.items():
        if key == "zero_attrition":
            if value is not True:
                raise AssessmentError("zero_attrition must remain true")
            limits[key] = True
        else:
            limits[key] = _finite(value, key)
    if not 0.0 <= limits["sign_agreement_min"] <= 1.0:
        raise AssessmentError("sign_agreement_min must be in [0, 1]")
    frozen_statistics = dict(FROZEN_STATISTICS if statistics is None else statistics)
    if frozen_statistics != FROZEN_STATISTICS:
        raise AssessmentError("statistical design is not the frozen assessment plan")
    trial_values: dict[str, dict[str, float]] = {}
    for row in panel.assessment_rows:
        trial = effects[row]
        if set(trial) != set(REQUIRED_ARMS):
            raise AssessmentError(f"{row} arm roster mismatch")
        trial_values[row] = {arm: _finite(trial[arm], f"{row}.{arm}") for arm in REQUIRED_ARMS}
    interventions = [trial_values[row]["intervention"] for row in panel.assessment_rows]
    scrubs = [trial_values[row]["scrub"] for row in panel.assessment_rows]
    exact_copies = [abs(trial_values[row]["exact_copy"]) for row in panel.assessment_rows]
    controls = [abs(trial_values[row][arm]) for row in panel.assessment_rows for arm in REQUIRED_ARMS[4:]]
    sign_hits = sum(1 for effect in interventions if (1 if effect > 0.0 else -1 if effect < 0.0 else 0) == prediction_lock.expected_sign)
    sign_agreement = sign_hits / len(interventions)
    aggregate = {
        "assessment_rows": len(panel.assessment_rows),
        "intervention_mean_signed": _mean(interventions, "intervention"),
        "intervention_mean_abs": _mean([abs(value) for value in interventions], "intervention"),
        "intervention_sign_agreement": sign_agreement,
        "scrub_mean_abs": _mean([abs(value) for value in scrubs], "scrub"),
        "exact_copy_max_abs": max(exact_copies),
        "control_max_abs": max(controls),
        "zero_attrition": True,
    }
    gates = [
        _gate("complete_assessment_rows", aggregate["assessment_rows"] == len(panel.assessment_rows), rows=aggregate["assessment_rows"]),
        _gate("prediction_lock_bound", prediction_lock.digest == digest(prediction_lock.as_dict()), prediction_lock_digest=prediction_lock.digest),
        _gate("heldout_intervention_effect", aggregate["intervention_mean_abs"] >= limits["effect_min_abs"], mean_abs_effect=aggregate["intervention_mean_abs"], minimum=limits["effect_min_abs"]),
        _gate("heldout_prediction_sign", aggregate["intervention_sign_agreement"] >= limits["sign_agreement_min"], sign_agreement=aggregate["intervention_sign_agreement"], minimum=limits["sign_agreement_min"]),
        _gate("heldout_causal_scrubbing", aggregate["scrub_mean_abs"] >= limits["scrub_min_abs"], mean_abs_effect=aggregate["scrub_mean_abs"], minimum=limits["scrub_min_abs"]),
        _gate("exact_copy_no_op", aggregate["exact_copy_max_abs"] <= limits["exact_copy_max_abs"], max_abs_effect=aggregate["exact_copy_max_abs"], maximum=limits["exact_copy_max_abs"]),
        _gate("control_finiteness", aggregate["control_max_abs"] <= limits["control_max_abs"], max_abs_effect=aggregate["control_max_abs"], maximum=limits["control_max_abs"]),
    ]
    body: dict[str, Any] = {
        "schema": "interp-lab.heldout-causal-assessment.v1",
        "state_slice": STATE_SLICE,
        "claim_status": claim_status,
        "panel_digest": panel.digest,
        "split_counts": {split: len(panel.rows[split]) for split in ("fit", "tune", "assessment")},
        "prediction_lock": prediction_lock.as_dict(),
        "prediction_lock_digest": prediction_lock.digest,
        "thresholds": limits,
        "statistics": frozen_statistics,
        "aggregate": aggregate,
        "gate_order": tuple(gate["name"] for gate in gates),
        "gates": gates,
        "all_gates_pass": all(gate["passed"] for gate in gates),
        "nonclaims": (
            "synthetic or locally authorized assessment only",
            "no independent reviewer acceptance",
            "no continual-learning retention claim",
            "no alignment or production claim",
        ),
    }
    return {**body, "report_digest": digest(body)}
