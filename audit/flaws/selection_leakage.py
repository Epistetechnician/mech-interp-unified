"""Class 2: selection leakage. Assess items equal fit items on a null model with few items."""
from ..core import Prereg
from ..pipeline import Spec
from ..worlds import WorldCfg
from ._common import net_case

CLASS_ID, NAME = 2, "selection_leakage"
VARIANTS = ["visible", "hidden"]
CFG = WorldCfg(n_P=0, n_N=0, n_decoy=0, n_inert=32, null=True, n_fit=16, n_assess=16)


def build(arm: str, variant: str, seed: int):
    pre = Prereg(metric="margin")
    vid = VARIANTS.index(variant)
    if arm == "bad":
        spec = Spec(pre, assess="fit", leak_log="honest" if variant == "visible" else "hidden")
        cc = False
    else:
        spec = Spec(pre)
        cc = None
    return net_case(CLASS_ID, NAME, vid, variant, arm, seed, CFG, spec, pre, cc)
