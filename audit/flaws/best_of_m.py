"""Class 11 (Phase 2): the instrument searches up to M item subsets and reports the first Supported one.

World scaled so the full-data effect sits just under the floor; SCALE is calibrated on development seeds
(PROTOCOL_PHASE2.md section 7). claim_correct is set from a rerun without the search.
"""
from dataclasses import replace

from ..core import Prereg
from ..pipeline import Spec, run_pipeline
from ..worlds import STRONG, WorldCfg
from ._common import net_case

CLASS_ID, NAME = 11, "best_of_m"
VARIANTS = ["subset_search"]
M, SUBSET_N = 20, 64
SCALE = 0.075


def build(arm: str, variant: str, seed: int):
    pre = Prereg(metric="margin", floor=0.5)
    cfg = WorldCfg(profile=tuple(SCALE * x for x in STRONG))
    spec = Spec(pre, best_of_m=M, subset_n=SUBSET_N) if arm == "bad" else Spec(pre)
    c = net_case(CLASS_ID, NAME, 0, variant, arm, seed, cfg, spec, pre, None)
    if arm == "bad":
        c.meta["honest_claim"] = run_pipeline(c.meta["world"], replace(c.meta["spec"], best_of_m=0)).claim
    return c
