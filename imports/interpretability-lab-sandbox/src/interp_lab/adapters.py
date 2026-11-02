"""Deterministic model and feature ports with in-memory reference adapters.

State slice: interpretability-lab-sandbox-v1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .canonical import digest


class ModelPort(Protocol):
    def forward(self, rows: Sequence[str], *, capture: bool) -> "Observation": ...


class FeaturePort(Protocol):
    def encode(self, activation: Sequence[float]) -> tuple[float, ...]: ...
    def decode(self, features: Sequence[float]) -> tuple[float, ...]: ...
    def intervene(self, activation: Sequence[float], feature_index: int, value: float) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class Observation:
    logits: tuple[float, ...]
    activations: tuple[tuple[float, ...], ...]
    events: tuple[dict[str, object], ...]


class SyntheticModel:
    """Small deterministic stand-in for a model/runtime pair."""

    def __init__(self, weights: Sequence[float] = (0.7, -0.2, 0.4, 0.1)) -> None:
        self.weights = tuple(float(value) for value in weights)

    @staticmethod
    def _activation(row: str) -> tuple[float, ...]:
        values = [sum(ord(char) for char in row) % 97, len(row) % 31, sum(char.isalpha() for char in row) % 29, sum(char.isdigit() for char in row) % 17]
        return tuple(value / 100.0 for value in values)

    def forward(self, rows: Sequence[str], *, capture: bool) -> Observation:
        activations = tuple(self._activation(row) for row in rows)
        logits = tuple(sum(weight * value for weight, value in zip(self.weights, activation)) for activation in activations)
        events = tuple({"row": index, "activation_digest": digest(activation)} for index, activation in enumerate(activations)) if capture else ()
        return Observation(logits=logits, activations=activations if capture else (), events=events)


class NativeAdapter:
    def __init__(self, model: SyntheticModel | None = None) -> None:
        self.model = model or SyntheticModel()

    def forward(self, rows: Sequence[str], *, capture: bool) -> Observation:
        return self.model.forward(rows, capture=capture)


class InstrumentedAdapter:
    def __init__(self, model: SyntheticModel | None = None) -> None:
        self.model = model or SyntheticModel()

    def forward(self, rows: Sequence[str], *, capture: bool) -> Observation:
        observation = self.model.forward(rows, capture=True)
        events = tuple({**event, "instrumented": True} for event in observation.events) if capture else ()
        return Observation(logits=observation.logits, activations=observation.activations if capture else (), events=events)


class SyntheticFeatureAdapter:
    """Identity feature map with a fixed, exactly invertible decoder."""

    def encode(self, activation: Sequence[float]) -> tuple[float, ...]:
        return tuple(float(value) for value in activation)

    def decode(self, features: Sequence[float]) -> tuple[float, ...]:
        return tuple(float(value) for value in features)

    def intervene(self, activation: Sequence[float], feature_index: int, value: float) -> tuple[float, ...]:
        result = list(float(item) for item in activation)
        if not 0 <= feature_index < len(result):
            raise IndexError("feature_index out of range")
        result[feature_index] = float(value)
        return tuple(result)
