"""Aggregate metrics used by the fixed synthetic qualification.

State slice: interpretability-lab-sandbox-v1.
"""
from __future__ import annotations

import math
from typing import Sequence


def max_abs_delta(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("metric dimensions differ")
    return max((abs(float(a) - float(b)) for a, b in zip(left, right)), default=0.0)


def nmse(target: Sequence[float], reconstruction: Sequence[float]) -> float:
    if len(target) != len(reconstruction):
        raise ValueError("metric dimensions differ")
    denominator = sum(float(value) ** 2 for value in target)
    if denominator == 0.0:
        return 0.0 if max_abs_delta(target, reconstruction) == 0.0 else math.inf
    return sum((float(a) - float(b)) ** 2 for a, b in zip(target, reconstruction)) / denominator


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("metric dimensions differ")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0 if left_norm == right_norm == 0.0 else 0.0
    return dot / (left_norm * right_norm)
