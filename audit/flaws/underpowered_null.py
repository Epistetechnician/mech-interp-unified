"""Class 10 (Phase 2): a NoCandidate verdict from an intervention that removes only a fraction of each unit's effect.

Bad: STRONG world, partial ablation. Good: a world whose carriers are EFF x STRONG, full ablation (an honest null
with a matched report). EFF is calibrated on development seeds (PROTOCOL_PHASE2.md section 7).
"""
from ..core import Prereg
from ..pipeline import Spec
from ..worlds import STRONG, WorldCfg
from ._common import net_case

CLASS_ID, NAME = 10, "underpowered_null"
VARIANTS = ["partial_ablation"]
EFF = 0.08


def build(arm: str, variant: str, seed: int):
    pre = Prereg(metric="margin", floor=0.5)
    if arm == "bad":
        cfg, spec = WorldCfg(), Spec(pre, abl_eff=EFF)
    else:
        cfg, spec = WorldCfg(profile=tuple(EFF * x for x in STRONG)), Spec(pre)
    return net_case(CLASS_ID, NAME, 0, variant, arm, seed, cfg, spec, pre, False if arm == "bad" else None)
