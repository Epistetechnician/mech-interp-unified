"""Independent checks for matched continual-learning aggregates.

State slice: interpretability-lab-sandbox-v1.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from .canonical import digest


STATE_SLICE = "interpretability-lab-sandbox-v1"
CONTROLLERS = {"sgd", "ewc", "behavioral_only", "causal_monitor"}
CONTROLLER_KEYS = {
    "retention",
    "prior_retention_mean",
    "adaptation",
    "conflict_task_adaptation",
    "calibration_error",
    "task_metrics",
    "monitoring_overhead_units",
    "updates",
    "total_compute_units",
}
V10_RETENTION_SUPERIORITY_MIN_ABS = 0.05
V10_ADAPTATION_NONINFERIORITY_MARGIN = 0.05


class ContinualValidationError(ValueError):
    """Raised when a continual-learning aggregate is not reproducible."""


def _finite(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ContinualValidationError(f"{field} is not numeric") from error
    if not math.isfinite(number):
        raise ContinualValidationError(f"{field} is nonfinite")
    return number


def validate_matched_continual_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate controller roster, matched accounting, finite metrics, and digest."""
    if report.get("schema") != "interp-lab.matched-continual-assessment.v1":
        raise ContinualValidationError("continual report schema mismatch")
    if report.get("state_slice") != STATE_SLICE:
        raise ContinualValidationError("continual report state slice mismatch")
    recorded_digest = report.get("report_digest")
    if not isinstance(recorded_digest, str):
        raise ContinualValidationError("continual report digest is missing")
    body = {key: value for key, value in report.items() if key != "report_digest"}
    if digest(body) != recorded_digest:
        raise ContinualValidationError("continual report digest mismatch")
    if report.get("matched_data") is not True or report.get("matched_update_count") is not True or report.get("matched_total_compute") is not True:
        raise ContinualValidationError("continual accounting is not matched")
    stream_length = _finite(report.get("stream_length"), "stream_length")
    total_budget = _finite(report.get("total_compute_budget"), "total_compute_budget")
    if stream_length <= 0 or total_budget <= 0:
        raise ContinualValidationError("continual stream or compute budget is not positive")
    controllers = report.get("controllers")
    if not isinstance(controllers, Mapping) or set(controllers) != CONTROLLERS:
        raise ContinualValidationError("continual controller roster mismatch")
    update_counts: set[float] = set()
    compute_totals: set[float] = set()
    for name, result in controllers.items():
        if not isinstance(result, Mapping) or set(result) != CONTROLLER_KEYS:
            raise ContinualValidationError(f"{name} metric schema mismatch")
        for field in CONTROLLER_KEYS - {"task_metrics", "monitoring_overhead_units", "updates", "total_compute_units"}:
            _finite(result.get(field), f"{name}.{field}")
        overhead = _finite(result.get("monitoring_overhead_units"), f"{name}.monitoring_overhead_units")
        updates = _finite(result.get("updates"), f"{name}.updates")
        compute = _finite(result.get("total_compute_units"), f"{name}.total_compute_units")
        if overhead < 0 or updates != stream_length or compute != total_budget:
            raise ContinualValidationError(f"{name} matched accounting is invalid")
        update_counts.add(updates)
        compute_totals.add(compute)
        task_metrics = result["task_metrics"]
        if not isinstance(task_metrics, Mapping) or set(task_metrics) != {"task_1", "task_2", "task_3"}:
            raise ContinualValidationError(f"{name} task metric roster mismatch")
        for task_name, metrics in task_metrics.items():
            if not isinstance(metrics, Mapping) or set(metrics) != {"accuracy", "calibration_error"}:
                raise ContinualValidationError(f"{name}.{task_name} metric schema mismatch")
            accuracy = _finite(metrics["accuracy"], f"{name}.{task_name}.accuracy")
            _finite(metrics["calibration_error"], f"{name}.{task_name}.calibration_error")
            if not 0.0 <= accuracy <= 1.0:
                raise ContinualValidationError(f"{name}.{task_name}.accuracy is outside [0, 1]")
        expected_summary = {
            "retention": float(task_metrics["task_1"]["accuracy"]),
            "prior_retention_mean": (float(task_metrics["task_1"]["accuracy"]) + float(task_metrics["task_2"]["accuracy"])) / 2.0,
            "adaptation": float(task_metrics["task_3"]["accuracy"]),
            "conflict_task_adaptation": float(task_metrics["task_2"]["accuracy"]),
            "calibration_error": float(task_metrics["task_3"]["calibration_error"]),
        }
        for field, expected in expected_summary.items():
            if not math.isclose(float(result[field]), expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ContinualValidationError(f"{name}.{field} summary does not match task metrics")
    if update_counts != {stream_length} or compute_totals != {total_budget}:
        raise ContinualValidationError("controller accounting is not equal")
    return {"status": "INDEPENDENT_CONTINUAL_AGGREGATE_VALIDATED", "report_digest": recorded_digest, "controllers": sorted(CONTROLLERS), "total_compute_budget": total_budget}


def evaluate_v10_controller_success(report: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen V10 controller rule without tuning or ranking tricks.

    The causal monitor must beat every comparator on prior-task retention by at
    least ``0.05`` while remaining within ``0.05`` adaptation of each one.
    This decision is a synthetic readiness check until a reviewed real packet
    is executed.
    """
    validate_matched_continual_report(report)
    controllers = report["controllers"]
    monitor = controllers["causal_monitor"]
    comparisons: dict[str, dict[str, float | bool]] = {}
    for name in sorted(CONTROLLERS - {"causal_monitor"}):
        comparator = controllers[name]
        retention_delta = float(monitor["prior_retention_mean"]) - float(comparator["prior_retention_mean"])
        adaptation_delta = float(monitor["adaptation"]) - float(comparator["adaptation"])
        comparisons[name] = {
            "retention_delta": retention_delta,
            "retention_superiority": retention_delta >= V10_RETENTION_SUPERIORITY_MIN_ABS,
            "adaptation_delta": adaptation_delta,
            "adaptation_noninferiority": adaptation_delta >= -V10_ADAPTATION_NONINFERIORITY_MARGIN,
        }
    all_retention = all(bool(value["retention_superiority"]) for value in comparisons.values())
    all_adaptation = all(bool(value["adaptation_noninferiority"]) for value in comparisons.values())
    return {
        "status": "CANDIDATE" if all_retention and all_adaptation else "NO_CANDIDATE",
        "retention_superiority_min_abs": V10_RETENTION_SUPERIORITY_MIN_ABS,
        "adaptation_noninferiority_margin": V10_ADAPTATION_NONINFERIORITY_MARGIN,
        "comparisons": comparisons,
    }
