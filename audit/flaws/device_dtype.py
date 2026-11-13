"""Class 3: device/dtype-induced selection change. Near-tied unit effects flip selection at reduced precision."""
from ..core import Prereg
from ..pipeline import Spec
from ..worlds import WorldCfg
from ._common import net_case

CLASS_ID, NAME = 3, "device_dtype_selection"
VARIANTS = ["near_ties"]


def build(arm: str, variant: str, seed: int):
    pre = Prereg(metric="margin")
    cfg = WorldCfg(tie="near" if arm == "bad" else "distinct")
    return net_case(CLASS_ID, NAME, 0, variant, arm, seed, cfg, Spec(pre), pre, None)
