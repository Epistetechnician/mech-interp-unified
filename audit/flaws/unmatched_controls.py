"""Class 9 (Phase 2): controls reported as size-matched random draws but drawn or evaluated differently.

World as class 8. claim_correct is set from a rerun with standard controls (the auditor-style plain redraw).
"""
from ..core import Prereg, Probe, world_rng
from ..pipeline import K, Spec
from ..worlds import WorldCfg
from ._common import net_case

CLASS_ID, NAME = 9, "unmatched_controls"
VARIANTS = ["size", "pool", "condition"]
EFFECT_RANGE = (0.02, 0.10)
DEFECT = {"size": dict(ctrl_k=K // 2), "pool": dict(ctrl_pool="lowscore"), "condition": dict(ctrl_cond="shuffled")}


def build(arm: str, variant: str, seed: int):
    pre = Prereg(metric="margin", floor=0.05)
    vid = VARIANTS.index(variant)
    e = float(world_rng(CLASS_ID, vid, seed, tag=3).uniform(*EFFECT_RANGE))
    cfg = WorldCfg(n_P=1, n_N=1, n_inert=30, profile=(e,), inert_d=0.5, inert_norm=1.0)
    spec = Spec(pre, **(DEFECT[variant] if arm == "bad" else {}))
    c = net_case(CLASS_ID, NAME, vid, variant, arm, seed, cfg, spec, pre, None)
    if arm == "bad":
        c.meta["honest_claim"] = c.run(Probe(plain_controls=True)).claim
    return c
