"""A2: the existing R5b causal share audit, `experiments/r5b_causal_audit.py::run_model`, imported unchanged.

Applicable only to cases that expose a toy privileged-basis model (class 5). FLAG iff the share-audit
multi-job fraction over used neurons exceeds R5b's own H3 bound (0.20) at the primary share cutoff (0.2).
"""
import importlib.util
from pathlib import Path

import numpy as np

from .base import FLAG, NA, PASS, Auditor

_MOD = None
H3_BOUND = 0.20
PRIMARY = 0.2


def _r5b():
    global _MOD
    if _MOD is None:
        path = Path(__file__).resolve().parents[2] / "experiments" / "r5b_causal_audit.py"
        spec = importlib.util.spec_from_file_location("r5b_causal_audit", path)
        _MOD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_MOD)
    return _MOD


def multi_fraction(toy: dict) -> float:
    m = _r5b()
    r = m.run_model(toy["W"], toy["b"], toy["n_feat"], toy["S"], toy["seed"])
    jobs = [j for j, u in zip(r["audit_jobs"][str(PRIMARY)], r["used"]) if u]
    return float(np.mean([j >= 2 for j in jobs])) if jobs else float("nan")


class R5bAudit(Auditor):
    name = "r5b_audit"

    def explain(self, case):
        return [] if case.toy is None else [f"multi_fraction={multi_fraction(case.toy):.3f}"]

    def audit(self, case):
        if case.toy is None:
            return NA
        f = multi_fraction(case.toy)
        return FLAG if f > H3_BOUND else PASS
