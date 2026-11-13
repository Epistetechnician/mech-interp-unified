"""Class 1: label-independent effect. Bad metric/selection use label-free output shift; decoys look causal."""
from ..core import Prereg
from ..pipeline import Spec
from ..worlds import WorldCfg
from ._common import net_case

CLASS_ID, NAME = 1, "label_independent_effect"
VARIANTS = ["decoy"]
CFG = WorldCfg(n_decoy=8, n_inert=14, decoy_d=8.0)


def build(arm: str, variant: str, seed: int):
    if arm == "bad":
        pre = Prereg(metric="labelfree")
        spec = Spec(pre, metric="labelfree", select_score="labelfree")
        cc = False
    else:
        pre = Prereg(metric="margin")
        spec = Spec(pre)
        cc = None
    return net_case(CLASS_ID, NAME, 0, variant, arm, seed, CFG, spec, pre, cc)
