"""A3: probe battery that re-executes the instrument. Thresholds frozen in audit/PROTOCOL.md section 4."""
import numpy as np

from ..core import Probe, jaccard
from ..worlds import make_perm_scale
from .base import FLAG, PASS, Auditor
from .rules_gate import gate_checks

JACCARD_MIN = 0.75
SHUFFLE_FRAC = 0.5
HEADROOM = 0.01
FRESH_CONTROL_SEED = 1234567


def probe_checks(case) -> list:
    if case.run is None:
        return []
    fired = []
    r = case.report()
    if gate_checks(r, case.prereg):
        fired.append("1_rules_gate")
    crit = r.provenance["criterion_applied"]
    if (crit["metric"] != case.prereg.metric or crit["floor"] < case.prereg.floor
            or crit["stat"] != case.prereg.stat):
        fired.append("2_prereg_consistency")
    if set(map(int, r.provenance["fit_ids"])) & set(map(int, r.provenance["assess_ids"])):
        fired.append("3_split_overlap")
    if r.claim == "Supported":
        sh = case.run(Probe(shuffle_readout=True, need_controls=False))
        if sh.sel["all"] >= SHUFFLE_FRAC * r.sel["all"]:
            fired.append("4_shuffled_readout")
        fr = case.run(Probe(fresh_assess=True))
        if fr.claim != "Supported":
            fired.append("5_fresh_assess")
    for dt in ("float16", "float64"):
        if jaccard(case.run(Probe(dtype=dt, need_controls=False)).selected, r.selected) < JACCARD_MIN:
            fired.append("6_dtype_stability")
            break
    H = case.meta["world"].H
    perm, scale = make_perm_scale(H, np.random.default_rng(np.random.SeedSequence([case.seed, 555])))
    tr = case.run(Probe(transform=(perm, scale), need_controls=False))
    if jaccard(perm[tr.selected], r.selected) < JACCARD_MIN:
        fired.append("7_equivariance")
    fresh = case.run(Probe(control_seed=FRESH_CONTROL_SEED, plain_controls=True))
    for k in ("sing", "plur", "all"):
        if np.max(r.ctrl[k]) < np.quantile(fresh.ctrl[k], 0.99):
            fired.append("8_control_redraw")
            break
    if r.metric_ceiling is not None and r.baseline_metric >= r.metric_ceiling - HEADROOM:
        fired.append("9_headroom")
    pr = case.run(Probe(use_prereg=True))
    if pr.claim != r.claim:
        fired.append("10_prereg_rerun")
    return fired


class ProbeBattery(Auditor):
    name = "probe_battery"

    def explain(self, case):
        return probe_checks(case)

    def audit(self, case):
        if case.run is None:
            return "NA"
        return FLAG if probe_checks(case) else PASS
