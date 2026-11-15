"""Class 8: control omission. Bad drops the top 5% of control draws and pads back with copies of retained draws.

One weak carrier per class plus noisy inert units; carrier effect size varies by seed (shared by both twins) so some worlds sit between the trimmed and untrimmed
control maxima. claim_correct is set from a rerun with full controls.
"""
from dataclasses import replace

from ..core import Prereg, world_rng
from ..pipeline import Spec, run_pipeline
from ..worlds import WorldCfg
from ._common import net_case

CLASS_ID, NAME = 8, "control_omission"
VARIANTS = ["trim5pct"]
EFFECT_RANGE = (0.02, 0.10)


def build(arm: str, variant: str, seed: int):
    pre = Prereg(metric="margin", floor=0.05)
    e = float(world_rng(CLASS_ID, 0, seed, tag=3).uniform(*EFFECT_RANGE))
    cfg = WorldCfg(n_P=1, n_N=1, n_inert=30, profile=(e,), inert_d=0.5, inert_norm=1.0)
    spec = Spec(pre, trim=0.05 if arm == "bad" else 0.0)
    c = net_case(CLASS_ID, NAME, 0, variant, arm, seed, cfg, spec, pre, None)
    if arm == "bad":
        honest = run_pipeline(c.meta["world"], replace(spec, trim=0.0, control_seed=spec.control_seed))
        c.meta["honest_claim"] = honest.claim
    return c
