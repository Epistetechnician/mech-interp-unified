"""A1: rules-based gate reimplementing the neuron-gate decision logic (INDEX.md section B; Gate.lean).

Support needs strict exceedance of the matched-random maxima in both classes plus a floor; the permutation-p
floor is 1/(1+n). Uses the frozen *preregistered* floor, not the floor the report claims to have used.
Checks decision arithmetic only (the Lean package proves harness logic, not instrument validity).
"""
import numpy as np

from ..core import N_DRAWS
from .base import FLAG, PASS, Auditor


def recompute_verdict(sel: dict, ctrl: dict, floor: float) -> str:
    ok = sel["sing"] > np.max(ctrl["sing"]) and sel["plur"] > np.max(ctrl["plur"]) and sel["all"] >= floor
    return "Supported" if ok else "NoCandidate"


def gate_checks(report, prereg) -> list:
    fired = []
    ctrl = report.ctrl
    if ctrl is None or report.n_draws < N_DRAWS or any(len(ctrl[k]) != report.n_draws for k in ("sing", "plur", "all")):
        return ["controls_missing_or_short"]
    n = report.n_draws
    p_exp = (1 + int((ctrl["all"] >= report.sel["all"]).sum())) / (1 + n)
    if report.p_reported < 1.0 / (1 + n) - 1e-15 or abs(report.p_reported - p_exp) > 1e-12:
        fired.append("p_value_mismatch")
    kb = report.known_bad_sel
    if recompute_verdict(kb, ctrl, prereg.floor) == "Supported":
        fired.append("known_bad_control_supported")
    if recompute_verdict(report.sel, ctrl, prereg.floor) != report.claim:
        fired.append("verdict_mismatch")
    return fired


class RulesGate(Auditor):
    name = "rules_gate"

    def explain(self, case):
        if case.run is None:
            return []
        return gate_checks(case.report(), case.prereg)

    def audit(self, case):
        if case.run is None:
            return "NA"
        return FLAG if gate_checks(case.report(), case.prereg) else PASS
