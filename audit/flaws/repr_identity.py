"""Class 4: representation-relative identity. Bad selects by incoming weight norm (not rescale invariant)."""
from ..core import Prereg
from ..pipeline import Spec
from ..worlds import WorldCfg
from ._common import net_case

CLASS_ID, NAME = 4, "representation_relative_identity"
VARIANTS = ["weight_norm"]


def build(arm: str, variant: str, seed: int):
    pre = Prereg(metric="margin")
    spec = Spec(pre, select_score="weightnorm" if arm == "bad" else "ablation")
    return net_case(CLASS_ID, NAME, 0, variant, arm, seed, WorldCfg(), spec, pre, None)
