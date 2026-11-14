from __future__ import annotations

from typing import Optional

from ..core import Case, Prereg, Probe, perturb_rng, world_rng
from ..pipeline import Spec, run_pipeline
from ..worlds import NetWorld, WorldCfg

ALL_PROBES = frozenset({"shuffle", "dtype", "transform", "fresh_assess", "control_seed", "prereg"})
ARM_ID = {"bad": 0, "good": 1}


def net_case(class_id: int, flaw_class: str, variant_id: int, variant: str, arm: str, seed: int,
             cfg: WorldCfg, spec: Spec, prereg: Prereg, claim_correct: Optional[bool] = None) -> Case:
    world = NetWorld(cfg, world_rng(class_id, variant_id, seed))
    # nuisance perturbation: good twin gets item-order shuffle + its own control seed (semantic no-ops)
    ctrl_seed = int(world_rng(class_id, variant_id, seed, tag=5).integers(0, 2**31))
    if arm == "good":
        ctrl_seed = int(perturb_rng(class_id, variant_id, 1, seed, tag=5).integers(0, 2**31))
        spec = _replace(spec, shuffle_items=True)
    spec = _replace(spec, control_seed=ctrl_seed)
    prng_seed = (class_id, variant_id, ARM_ID[arm], seed)

    def run(probe: Optional[Probe] = None):
        return run_pipeline(world, spec, probe, perturb_rng=perturb_rng(*prng_seed, tag=9))

    c = Case(case_id=f"{flaw_class}/{variant}/{arm}/{seed}", flaw_class=flaw_class, variant=variant, arm=arm,
             seed=seed, flawed=(arm == "bad"), prereg=prereg, run=run, supports=ALL_PROBES,
             claim_correct=claim_correct)
    c.meta["world"] = world
    return c


def _replace(spec: Spec, **kw) -> Spec:
    from dataclasses import replace
    return replace(spec, **kw)
