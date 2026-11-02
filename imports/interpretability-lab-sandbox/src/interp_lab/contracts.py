"""Immutable contracts for sandbox observations and trajectories.

State slice: interpretability-lab-sandbox-v1.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .canonical import digest


def _vector(values: Sequence[float], field: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field} must be a finite vector")
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError(f"{field} must be a finite vector")
    return result


def _effects(values: Mapping[str, float], field: str) -> dict[str, float]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{field} must be a nonempty mapping")
    result = {str(key): float(value) for key, value in values.items()}
    if any(not key or not math.isfinite(value) for key, value in result.items()):
        raise ValueError(f"{field} contains invalid values")
    return dict(sorted(result.items()))


@dataclass(frozen=True)
class StateSnapshot:
    state_id: str
    step: int
    parameters: Sequence[float]
    representation: Sequence[float]
    circuits: Mapping[str, float]
    intervention_effects: Mapping[str, float]
    behavior: Sequence[float]

    def __post_init__(self) -> None:
        if not isinstance(self.state_id, str) or not self.state_id:
            raise ValueError("state_id must be nonempty")
        if type(self.step) is not int or self.step < 0:
            raise ValueError("step must be a nonnegative integer")
        object.__setattr__(self, "parameters", _vector(self.parameters, "parameters"))
        object.__setattr__(self, "representation", _vector(self.representation, "representation"))
        object.__setattr__(self, "circuits", _effects(self.circuits, "circuits"))
        object.__setattr__(self, "intervention_effects", _effects(self.intervention_effects, "intervention_effects"))
        object.__setattr__(self, "behavior", _vector(self.behavior, "behavior"))

    def as_dict(self) -> dict[str, Any]:
        return {"state_id": self.state_id, "step": self.step, "parameters": self.parameters, "representation": self.representation, "circuits": self.circuits, "intervention_effects": self.intervention_effects, "behavior": self.behavior}

    @property
    def state_digest(self) -> str:
        return digest(self.as_dict())


@dataclass(frozen=True)
class Checkpoint:
    """A feature-space checkpoint used for stability and reconstruction."""

    checkpoint_id: str
    features: Mapping[str, Sequence[float]]

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise ValueError("checkpoint_id must be nonempty")
        normal = {str(key): _vector(value, "feature") for key, value in self.features.items()}
        if not normal:
            raise ValueError("features must be nonempty")
        object.__setattr__(self, "features", dict(sorted(normal.items())))

    @property
    def digest(self) -> str:
        return digest({"checkpoint_id": self.checkpoint_id, "features": self.features})


@dataclass(frozen=True)
class LineageEdge:
    source_id: str
    target_id: str
    mass: float

    def __post_init__(self) -> None:
        if not self.source_id or not self.target_id:
            raise ValueError("lineage IDs must be nonempty")
        mass = float(self.mass)
        if not math.isfinite(mass) or not 0.0 < mass <= 1.0:
            raise ValueError("lineage mass must be in (0, 1]")
        object.__setattr__(self, "mass", mass)


@dataclass(frozen=True)
class Transition:
    before_state: str
    after_state: str
    edges: Sequence[LineageEdge]

    def __post_init__(self) -> None:
        edges = tuple(self.edges)
        if not self.before_state or not self.after_state or not edges:
            raise ValueError("transition fields are incomplete")
        identities = [(edge.source_id, edge.target_id) for edge in edges]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate lineage edge")
        source_mass: dict[str, float] = {}
        target_mass: dict[str, float] = {}
        for edge in edges:
            source_mass[edge.source_id] = source_mass.get(edge.source_id, 0.0) + edge.mass
            target_mass[edge.target_id] = target_mass.get(edge.target_id, 0.0) + edge.mass
        if any(value > 1.0 + 1e-12 for value in (*source_mass.values(), *target_mass.values())):
            raise ValueError("lineage mass exceeds one")
        object.__setattr__(self, "edges", edges)

    def as_dict(self) -> dict[str, Any]:
        return {"before_state": self.before_state, "after_state": self.after_state, "edges": tuple(asdict(edge) for edge in self.edges)}


@dataclass(frozen=True)
class Thresholds:
    parity_abs_tol: float = 0.0
    repeatability_abs_tol: float = 0.0
    no_op_abs_tol: float = 0.0
    reconstruction_nmse_max: float = 0.05
    intervention_min_abs_effect: float = 1e-9
    stability_min_cosine: float = 0.8
    scrub_min_abs_effect: float = 1e-9

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values.values()):
            raise ValueError("thresholds must be finite and nonnegative")
        if self.stability_min_cosine > 1.0:
            raise ValueError("stability_min_cosine must be at most one")
