"""Shared types: Prereg, Probe, Report, Case, seeding."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

N_DRAWS = 1000


@dataclass(frozen=True)
class Prereg:
    """Frozen criterion the instrument claims to have preregistered. The gate reads the floor from here."""
    metric: str = "margin"
    floor: float = 0.5
    stat: str = "max"
    n_min: int = N_DRAWS


@dataclass(frozen=True)
class Probe:
    """Overrides an auditor may apply when re-executing an instrument."""
    shuffle_readout: bool = False
    dtype: Optional[str] = None            # 'float16' | 'float32' | 'float64'
    transform: Optional[tuple] = None      # (perm, scale): function-preserving unit permutation + rescale
    fresh_assess: bool = False
    control_seed: Optional[int] = None
    use_prereg: bool = False
    need_controls: bool = True
    plain_controls: bool = False           # auditor redraws matched-random controls itself (no instrument-side trimming)


@dataclass
class Report:
    claim: str                              # 'Supported' | 'NoCandidate'
    selected: np.ndarray
    sel: dict                               # {'sing','plur','all'}
    ctrl: dict                              # {'sing','plur','all'} -> (n,) arrays (None if probe skipped controls)
    n_draws: int
    p_reported: float
    floor_applied: float
    known_bad_sel: dict                     # effects of a known-bad (random) selection
    provenance: dict                        # fit_ids, assess_ids, criterion_applied
    baseline_metric: float
    metric_ceiling: Optional[float]


@dataclass
class Case:
    case_id: str
    flaw_class: str
    variant: str
    arm: str                                # 'bad' | 'good'
    seed: int
    flawed: bool
    prereg: Prereg
    run: Callable[[Optional[Probe]], Report]
    supports: frozenset                     # probes the instrument implements meaningfully
    claim_correct: Optional[bool] = None    # defined by construction where possible
    toy: Optional[dict] = None              # class 5: {'W','b','n_feat','S','seed'} for the R5b audit
    meta: dict = field(default_factory=dict)

    def report(self) -> Report:
        if "_report" not in self.meta:
            self.meta["_report"] = self.run(None)
        return self.meta["_report"]


def world_rng(class_id: int, variant_id: int, seed: int, tag: int = 0) -> np.random.Generator:
    """Twins (bad/good) share this stream, so they share the world."""
    return np.random.default_rng(np.random.SeedSequence([class_id, variant_id, seed, tag]))


def perturb_rng(class_id: int, variant_id: int, arm_id: int, seed: int, tag: int = 0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([class_id, variant_id, arm_id, seed, 7919, tag]))


def jaccard(a, b) -> float:
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / max(len(a | b), 1)
