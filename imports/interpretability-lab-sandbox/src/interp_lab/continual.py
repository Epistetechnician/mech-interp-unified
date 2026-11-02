"""Deterministic continual-learning controller simulator.

State slice: interpretability-lab-sandbox-v1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .canonical import digest


@dataclass
class Learner:
    weights: list[float]

    def score(self, row: Sequence[float]) -> float:
        return sum(weight * value for weight, value in zip(self.weights, row))

    def update(self, row: Sequence[float], target: float, *, rate: float, penalty: Sequence[float] | None = None) -> None:
        error = target - self.score(row)
        for index, value in enumerate(row):
            regularizer = 0.0 if penalty is None else penalty[index] * self.weights[index]
            self.weights[index] += rate * (error * value - regularizer)


def run_continual_suite() -> dict[str, Any]:
    """Compare fixed controllers on a tiny ordered two-task stream."""
    task_a = (((1.0, 0.0), 1.0), ((0.0, 1.0), -1.0))
    task_b = (((1.0, 1.0), 1.0), ((-1.0, 1.0), -1.0))
    results: dict[str, dict[str, float]] = {}
    for name in ("sgd", "replay", "ewc", "behavioral_only", "causal_monitor"):
        learner = Learner([0.0, 0.0])
        for row, target in task_a:
            learner.update(row, target, rate=0.2)
        retention_before = sum(1.0 for row, target in task_a if learner.score(row) * target > 0.0) / len(task_a)
        for row, target in task_b:
            penalty = (0.15, 0.15) if name in {"ewc", "causal_monitor"} else None
            learner.update(row, target, rate=0.2, penalty=penalty)
            if name == "replay":
                for replay_row, replay_target in task_a:
                    learner.update(replay_row, replay_target, rate=0.05)
        retention_after = sum(1.0 for row, target in task_a if learner.score(row) * target > 0.0) / len(task_a)
        adaptation = sum(1.0 for row, target in task_b if learner.score(row) * target > 0.0) / len(task_b)
        overhead = 1.0 if name in {"ewc", "causal_monitor"} else 0.0
        results[name] = {"retention": retention_after, "adaptation": adaptation, "monitoring_overhead": overhead, "retention_delta": retention_after - retention_before}
    return {"schema":"interp-lab.continual-report.v1","state_slice":"interpretability-lab-sandbox-v1","claim_status":"LOCAL_SYNTHETIC_ONLY","matched_stream":True,"controllers":results,"nonclaims":("synthetic controller comparison only","no real training or model evidence","no general continual-learning claim")}


def run_matched_continual_assessment() -> dict[str, Any]:
    """Run a fixed, conflict-bearing synthetic controller comparison.

    Every controller sees the same ordered stream and update count. A fixed
    compute budget pads cheaper policies so total accounted units match; the
    monitoring units remain visible in the report.
    """
    tasks = (
        (
            ((1.0, 0.0, 0.0), 1.0),
            ((-1.0, 0.0, 0.0), -1.0),
            ((0.0, 1.0, 0.0), 1.0),
            ((0.0, -1.0, 0.0), -1.0),
        ),
        (
            ((1.0, 0.0, 0.0), -1.0),
            ((-1.0, 0.0, 0.0), 1.0),
            ((0.0, 1.0, 0.0), 1.0),
            ((0.0, -1.0, 0.0), -1.0),
        ),
        (
            ((0.0, 0.0, 1.0), 1.0),
            ((0.0, 0.0, -1.0), -1.0),
        ),
    )
    controllers = ("sgd", "ewc", "behavioral_only", "causal_monitor")
    monitor_units = {"sgd": 0, "ewc": 2, "behavioral_only": 3, "causal_monitor": 4}
    base_units = 12
    budget_per_update = base_units + max(monitor_units.values())
    stream_length = sum(len(task) for task in tasks)
    total_budget = stream_length * budget_per_update
    results: dict[str, dict[str, Any]] = {}

    def accuracy(learner: Learner, task: tuple[tuple[tuple[float, ...], float], ...]) -> float:
        return sum(1.0 for row, target in task if learner.score(row) * target > 0.0) / len(task)

    def calibration(learner: Learner, task: tuple[tuple[tuple[float, ...], float], ...]) -> float:
        return sum(abs(learner.score(row) - target) for row, target in task) / len(task)

    for name in controllers:
        learner = Learner([0.0, 0.0, 0.0])
        compute = 0
        updates = 0
        anchor: tuple[float, ...] | None = None
        for task_index, task in enumerate(tasks):
            if task_index == 1 and name in {"ewc", "causal_monitor"}:
                anchor = tuple(learner.weights)
            for row, target in task:
                if name == "ewc":
                    importance = (1.2, 0.0, 0.0)
                    rate_scale = 1.0
                elif name == "causal_monitor":
                    importance = (1.2, 0.0, 0.0)
                    rate_scale = 0.9 if task_index == 1 else 1.0
                elif name == "behavioral_only":
                    importance = (0.4, 0.0, 0.0) if task_index == 1 else (0.0, 0.0, 0.0)
                    rate_scale = 1.0
                else:
                    importance = (0.0, 0.0, 0.0)
                    rate_scale = 1.0
                error = target - learner.score(row)
                for index, value in enumerate(row):
                    anchor_value = 0.0 if anchor is None else anchor[index]
                    regularizer = importance[index] * (learner.weights[index] - anchor_value)
                    learner.weights[index] += 0.15 * rate_scale * (error * value - regularizer)
                updates += 1
                compute += base_units + monitor_units[name]
        compute += total_budget - compute  # fixed-budget padding, no state change
        task_metrics = {
            f"task_{index + 1}": {
                "accuracy": accuracy(learner, task),
                "calibration_error": calibration(learner, task),
            }
            for index, task in enumerate(tasks)
        }
        results[name] = {
            "retention": task_metrics["task_1"]["accuracy"],
            "prior_retention_mean": sum(task_metrics[f"task_{index + 1}"]["accuracy"] for index in range(2)) / 2.0,
            "adaptation": task_metrics["task_3"]["accuracy"],
            "conflict_task_adaptation": task_metrics["task_2"]["accuracy"],
            "calibration_error": task_metrics["task_3"]["calibration_error"],
            "task_metrics": task_metrics,
            "monitoring_overhead_units": monitor_units[name],
            "updates": updates,
            "total_compute_units": compute,
        }
    body = {
        "schema": "interp-lab.matched-continual-assessment.v1",
        "state_slice": "interpretability-lab-sandbox-v1",
        "claim_status": "LOCAL_SYNTHETIC_ONLY",
        "matched_data": True,
        "matched_update_count": len({result["updates"] for result in results.values()}) == 1,
        "matched_total_compute": len({result["total_compute_units"] for result in results.values()}) == 1,
        "stream_length": stream_length,
        "total_compute_budget": total_budget,
        "controllers": results,
        "nonclaims": (
            "synthetic controller comparison only",
            "no real model training or retention evidence",
            "no causal interpretability claim",
            "no alignment or production claim",
        ),
    }
    return {**body, "report_digest": digest(body)}
