"""Reconstruction diagnostics for fresh model-matched feature assets.

State slice: interpretability-lab-sandbox-v1.

The Qwen v3 and v4 packets used a per-row maximum NMSE gate.  This module
adds a separate, predeclared diagnostic for a future packet: pooled,
globally-centered FVU is reported together with per-row NMSE.  It does not
alter either closed packet or authorize model execution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


class ReconstructionError(ValueError):
    """Raised when reconstruction inputs cannot be compared safely."""


def _matrix(values: Sequence[Sequence[float]], field: str) -> tuple[tuple[float, ...], ...]:
    if isinstance(values, (str, bytes)):
        raise ReconstructionError(f"{field} must be a nonempty finite matrix")
    rows = tuple(tuple(float(value) for value in row) for row in values)
    if not rows or not rows[0]:
        raise ReconstructionError(f"{field} must be a nonempty finite matrix")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ReconstructionError(f"{field} rows must have equal width")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ReconstructionError(f"{field} must contain only finite values")
    return rows


@dataclass(frozen=True)
class ReconstructionSummary:
    """Aggregate and row-wise reconstruction metrics for one fixed split."""

    row_count: int
    width: int
    pooled_fvu: float
    max_row_nmse: float
    mean_row_nmse: float

    def __post_init__(self) -> None:
        if type(self.row_count) is not int or self.row_count <= 0:
            raise ReconstructionError("row_count must be positive")
        if type(self.width) is not int or self.width <= 0:
            raise ReconstructionError("width must be positive")
        for name in ("pooled_fvu", "max_row_nmse", "mean_row_nmse"):
            value = float(getattr(self, name))
            if math.isnan(value) or value < 0.0:
                raise ReconstructionError(f"{name} must be nonnegative")

    def as_dict(self) -> dict[str, float | int]:
        return {
            "row_count": self.row_count,
            "width": self.width,
            "pooled_fvu": self.pooled_fvu,
            "max_row_nmse": self.max_row_nmse,
            "mean_row_nmse": self.mean_row_nmse,
        }


def summarize_reconstruction(
    targets: Sequence[Sequence[float]],
    reconstructions: Sequence[Sequence[float]],
) -> ReconstructionSummary:
    """Compute pooled FVU and row NMSE without changing either input matrix.

    FVU is ``SSE / SST`` after centering every target coordinate by the single
    mean over the supplied split.  A constant target matrix is handled
    fail-closed: exact reconstruction yields zero FVU and any error yields
    infinity.  Row NMSE uses the row's squared norm as denominator, so a
    localized collapse cannot be hidden by pooled variance.
    """

    target_rows = _matrix(targets, "targets")
    reconstruction_rows = _matrix(reconstructions, "reconstructions")
    if len(target_rows) != len(reconstruction_rows) or len(target_rows[0]) != len(reconstruction_rows[0]):
        raise ReconstructionError("targets and reconstructions must have the same shape")
    count = len(target_rows) * len(target_rows[0])
    global_mean = sum(value for row in target_rows for value in row) / count
    sse = sum((target - reconstruction) ** 2 for target_row, reconstruction_row in zip(target_rows, reconstruction_rows) for target, reconstruction in zip(target_row, reconstruction_row))
    sst = sum((target - global_mean) ** 2 for row in target_rows for target in row)
    pooled_fvu = sse / sst if sst else (0.0 if sse == 0.0 else math.inf)

    row_nmse: list[float] = []
    for target_row, reconstruction_row in zip(target_rows, reconstruction_rows):
        denominator = sum(value * value for value in target_row)
        row_error = sum((target - reconstruction) ** 2 for target, reconstruction in zip(target_row, reconstruction_row))
        row_nmse.append(row_error / denominator if denominator else (0.0 if row_error == 0.0 else math.inf))
    return ReconstructionSummary(
        row_count=len(target_rows),
        width=len(target_rows[0]),
        pooled_fvu=pooled_fvu,
        max_row_nmse=max(row_nmse),
        mean_row_nmse=sum(row_nmse) / len(row_nmse),
    )
