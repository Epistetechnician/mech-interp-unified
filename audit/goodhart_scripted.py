"""Scripted adversaries for audit/GOODHART.md condition C0. CPU only, no model calls.

python3 -m audit.goodhart_scripted [--out audit/results/f2_goodhart_scripted.json]
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

from .auditors.probe_battery import probe_checks
from .auditors.rules_gate import gate_checks
from .core import Prereg
from .flaws._common import net_case
from .harness import dumps
from .pipeline import Spec
from .stats import boot_ci, wilson
from .worlds import WorldCfg

SEEDS = range(900, 950)
Q = 200
CFG = WorldCfg(n_decoy=8, n_inert=14, decoy_d=8.0)
METRICS = ["margin", "labelfree", "topdecile", "accuracy"]
SELECT = ["ablation", "labelfree", "weightnorm"]
KNOBS = ["p_metric", "p_floor", "metric", "log_metric", "select", "assess", "leak_log", "n_fit_use", "trim",
         "applied_floor", "control_seed"]


def sample_knob(name, rng):
    return {
        "p_metric": lambda: METRICS[rng.integers(4)],
        "p_floor": lambda: float(rng.uniform(0, 1)),
        "metric": lambda: METRICS[rng.integers(4)],
        "log_metric": lambda: [None, "margin", "labelfree", "topdecile", "accuracy"][rng.integers(5)],
        "select": lambda: SELECT[rng.integers(3)],
        "assess": lambda: ["heldout", "fit"][rng.integers(2)],
        "leak_log": lambda: ["honest", "hidden"][rng.integers(2)],
        "n_fit_use": lambda: [None, 6, 2][rng.integers(3)],
        "trim": lambda: float(rng.choice([0.0, 0.05, 0.1, 0.2])),
        "applied_floor": lambda: [None, float(rng.uniform(0, 1))][rng.integers(2)],
        "control_seed": lambda: int(rng.integers(0, 2**31)),
    }[name]()


def build_case(theta, seed):
    pre = Prereg(metric=theta["p_metric"], floor=theta["p_floor"])
    spec = Spec(pre, metric=theta["metric"], select_score=theta["select"], assess=theta["assess"],
                leak_log=theta["leak_log"], trim=theta["trim"], applied_floor=theta["applied_floor"],
                log_metric=theta["log_metric"], n_fit_use=theta["n_fit_use"], control_seed=theta["control_seed"])
    return net_case(9, "goodhart", 0, "knobs", "bad", seed, CFG, spec, pre, None)


def is_wrong(case) -> bool:
    r = case.report()
    roles = case.meta["world"].roles[r.selected]
    return r.claim == "Supported" and np.mean(np.isin(roles, ["P", "N"])) < 0.5


def n_fired(auditor: str, case) -> int:
    return len(gate_checks(case.report(), case.prereg)) if auditor == "rules_gate" else len(probe_checks(case))


def evaluate_theta(auditor, theta, seed):
    """-> (wrong, fired_count) ; crash => (False, 99)."""
    try:
        c = build_case(theta, seed)
        return is_wrong(c), n_fired(auditor, c)
    except Exception:
        return False, 99


def search(auditor: str, mode: str, seed: int):
    rng = np.random.default_rng(np.random.SeedSequence([zlib.crc32(f"{auditor}{mode}".encode()), seed]))
    theta = {k: sample_knob(k, rng) for k in KNOBS}
    best = None
    best_obj = None
    for q in range(1, Q + 1):
        if mode == "random" or q == 1:
            cand = {k: sample_knob(k, rng) for k in KNOBS} if q > 1 else theta
        else:
            cand = dict(best)
            k = KNOBS[rng.integers(len(KNOBS))]
            cand[k] = sample_knob(k, rng)
        wrong, fired = evaluate_theta(auditor, cand, seed)
        if wrong and fired == 0:
            return {"success": True, "queries": q, "theta": cand}
        obj = fired if wrong else 100 + fired      # gray-box objective; random search ignores it
        if best_obj is None or obj <= best_obj:
            best, best_obj = cand, obj
    return {"success": False, "queries": Q, "theta": best}


def run():
    out = {"seeds": [SEEDS.start, SEEDS.stop - 1], "queries": Q, "runs": {}}
    thetas = {}
    for auditor in ("rules_gate", "probe_battery"):
        for mode in ("random", "es"):
            res = [search(auditor, mode, s) for s in SEEDS]
            k = sum(r["success"] for r in res)
            qs = [r["queries"] for r in res if r["success"]]
            key = f"{auditor}/{mode}"
            out["runs"][key] = {"successes": k, "n": len(res), "rate": k / len(res),
                                "boot_ci": boot_ci([(k, len(res))], tag=zlib.crc32(key.encode())),
                                "wilson_ci": wilson(k, len(res)), "median_queries_to_success": float(np.median(qs)) if qs else None}
            thetas[key] = [(s, r["theta"]) for s, r in zip(SEEDS, res) if r["success"]]
    # transfer: configurations that beat one auditor, judged by the other
    other = {"rules_gate": "probe_battery", "probe_battery": "rules_gate"}
    for key, lst in thetas.items():
        aud = key.split("/")[0]
        flagged = [n_fired(other[aud], build_case(t, s)) > 0 for s, t in lst]
        out["runs"][key]["transfer_flag_rate_on_" + other[aud]] = (float(np.mean(flagged)) if flagged else None)
        out["runs"][key]["example_theta"] = lst[0][1] if lst else None
    for key, r in out["runs"].items():
        lo, hi = r["boot_ci"]
        r["decision_component"] = "robust" if hi <= 0.10 else ("broken" if lo >= 0.50 else "partial")
    dec = {}
    for aud in ("rules_gate", "probe_battery"):
        comps = [out["runs"][f"{aud}/{m}"]["decision_component"] for m in ("random", "es")]
        dec[aud] = "robust_within_space" if all(c == "robust" for c in comps) else (
            "broken" if "broken" in comps else "partially_robust")
    out["decision"] = dec
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results" / "f2_goodhart_scripted.json")
    a = ap.parse_args(argv)
    res = run()
    a.out.write_text(dumps(res))
    print(json.dumps({k: (v["successes"], v["n"], v["decision_component"]) for k, v in res["runs"].items()}), res["decision"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
