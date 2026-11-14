from .base import FLAG, NA, PASS, Auditor
from .nulls import AlwaysFlag, AlwaysPass, CoinFlip
from .probe_battery import ProbeBattery
from .r5b import R5bAudit
from .rules_gate import RulesGate


def default_auditors():
    return [RulesGate(), R5bAudit(), ProbeBattery()]


def null_auditors():
    return [AlwaysPass(), AlwaysFlag(), CoinFlip()]
