"""Class 7: post hoc criterion change on a weak-effect world that honestly fails the preregistered floor."""
from ..core import Prereg
from ..pipeline import Spec
from ..worlds import STRONG, WorldCfg
from ._common import net_case

CLASS_ID, NAME = 7, "post_hoc_criterion_change"
VARIANTS = ["relax_floor", "switch_metric"]
WEAK = tuple(0.06 * x for x in STRONG)
CFG = WorldCfg(profile=WEAK)


def build(arm: str, variant: str, seed: int):
    pre = Prereg(metric="margin", floor=0.5)
    vid = VARIANTS.index(variant)
    if arm == "good":
        spec = Spec(pre)
    elif variant == "relax_floor":
        spec = Spec(pre, applied_floor=0.1)
    else:
        spec = Spec(pre, metric="topdecile", log_metric="margin")
    return net_case(CLASS_ID, NAME, vid, variant, arm, seed, CFG, spec, pre, False if arm == "bad" else None)
