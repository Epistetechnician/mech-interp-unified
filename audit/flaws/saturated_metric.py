"""Class 6: saturated behaviour metric. Bad = accuracy metric on a rank-1 task (baseline accuracy exactly 1.0)."""
from ..core import Prereg
from ..pipeline import Spec
from ..worlds import WorldCfg
from ._common import net_case

CLASS_ID, NAME = 6, "saturated_behavior_metric"
VARIANTS = ["accuracy_ceiling"]
SIGMA_GOOD = 1.2


def build(arm: str, variant: str, seed: int):
    pre = Prereg(metric="accuracy", floor=0.05)
    cfg = WorldCfg(sigma=0.0 if arm == "bad" else SIGMA_GOOD, ablate="zero")
    spec = Spec(pre, metric="accuracy")
    return net_case(CLASS_ID, NAME, 0, variant, arm, seed, cfg, spec, pre, False if arm == "bad" else None)
