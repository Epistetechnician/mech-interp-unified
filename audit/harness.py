"""Audit harness: build cases, run pluggable auditors, aggregate TPR/FPR with intervals."""
from __future__ import annotations

import hashlib
import json
import zlib
from collections import Counter, defaultdict

from .auditors import FLAG, NA, default_auditors, null_auditors
from .auditors.probe_battery import ProbeBattery
from .auditors.rules_gate import RulesGate
from .flaws import ALL_CLASSES
from .stats import label, rate, wilson

PROTOCOL_N = 200


def iter_cases(n: int, seed0: int = 0, classes=None):
    for mod in (classes or ALL_CLASSES):
        for v in mod.VARIANTS:
            for s in range(seed0, seed0 + n):
                for arm in ("bad", "good"):
                    yield mod, v, arm, s


def evaluate(n: int = PROTOCOL_N, seed0: int = 0, auditors=None, nulls=True, classes=None) -> dict:
    auds = list(auditors or default_auditors()) + (null_auditors() if nulls else [])
    # dec[aud][class][variant][arm] -> Counter(FLAG/PASS/NA); plus claim_correct split and digests
    dec = {a.name: defaultdict(lambda: defaultdict(lambda: defaultdict(Counter))) for a in auds}
    cc_false = {a.name: defaultdict(Counter) for a in auds}   # class -> Counter over flawed cases whose claim is wrong
    lines = {a.name: [] for a in auds}
    claims = defaultdict(Counter)
    fires = defaultdict(Counter)
    diag_aud = [a for a in auds if isinstance(a, (RulesGate, ProbeBattery))]
    for mod, v, arm, s in iter_cases(n, seed0, classes):
        case = mod.build(arm, v, s)
        key = (mod.NAME, v, arm)
        if case.run is not None:
            claims[key][case.report().claim] += 1
        wrong = None
        if arm == "bad" and case.run is not None:
            wrong = (case.claim_correct is False) or (case.meta.get("honest_claim") not in (None, case.report().claim))
        for a in auds:
            d = a.audit(case)
            lines[a.name].append(f"{case.case_id}:{d}")
            dec[a.name][mod.NAME][v][arm][d] += 1
            if wrong:
                cc_false[a.name][mod.NAME][d] += 1
        for a in diag_aud:
            for f in a.explain(case):
                fires[(a.name, mod.NAME, v, arm)][f] += 1
    out = {"n_seeds": n, "seed0": seed0, "auditors": {}, "instrument_claims": {}, "fired_checks": {}}
    for aud, per_class in dec.items():
        res = {}
        for cname, per_var in per_class.items():
            tstr, fstr, pv = [], [], {}
            na = 0
            for v, arms in per_var.items():
                bk = arms["bad"][FLAG]; bn = arms["bad"][FLAG] + arms["bad"]["PASS"]
                gk = arms["good"][FLAG]; gn = arms["good"][FLAG] + arms["good"]["PASS"]
                na += arms["bad"][NA] + arms["good"][NA]
                tstr.append((bk, bn)); fstr.append((gk, gn))
                pv[v] = {"tpr": bk / bn if bn else None, "fpr": gk / gn if gn else None, "n_bad": bn, "n_good": gn}
            tag = zlib.crc32((aud + cname).encode())
            tpr, fpr = rate(tstr, tag), rate(fstr, tag + 1)
            ccf = cc_false[aud][cname]
            nw = ccf[FLAG] + ccf["PASS"]
            res[cname] = {"tpr": tpr, "fpr": fpr, "label": label(tpr, fpr), "n_na": na, "per_variant": pv,
                          "tpr_claim_wrong": {"k": ccf[FLAG], "n": nw, "rate": ccf[FLAG] / nw if nw else None}}
        out["auditors"][aud] = res
        out["auditors"][aud]["_digest"] = hashlib.sha256("\n".join(lines[aud]).encode()).hexdigest()
    for (cname, v, arm), c in claims.items():
        out["instrument_claims"][f"{cname}/{v}/{arm}"] = dict(c)
    for (an, cname, v, arm), c in fires.items():
        out["fired_checks"][f"{an}/{cname}/{v}/{arm}"] = dict(c)
    out["harness_validity"] = harness_validity(out)
    out["decisions"] = decisions(out)
    return out


def harness_validity(out: dict) -> dict:
    a = out["auditors"]
    v = {}
    ap, af, cf = a["always_pass"], a["always_flag"], a["coin_flip"]
    classes = [c for c in ap if not c.startswith("_")]
    v["always_pass_zero"] = all(ap[c]["tpr"]["k"] == 0 and ap[c]["fpr"]["k"] == 0 for c in classes)
    v["always_flag_one"] = all(af[c]["tpr"]["k"] == af[c]["tpr"]["n"] and af[c]["fpr"]["k"] == af[c]["fpr"]["n"]
                               for c in classes)
    # pooled over classes (16 per-class CIs would fail by multiplicity alone); per-class point estimates bounded
    pool = lambda r: (sum(cf[c][r]["k"] for c in classes), sum(cf[c][r]["n"] for c in classes))
    ok = True
    for r in ("tpr", "fpr"):
        k, n = pool(r)
        lo, hi = wilson(k, n)
        ok &= lo <= 0.5 <= hi
        ok &= all(abs(cf[c][r]["rate"] - 0.5) <= 0.15 for c in classes)
    v["coin_flip_contains_half"] = bool(ok)
    v["valid"] = all(v.values())
    return v


def decisions(out: dict) -> dict:
    a = out["auditors"]
    cls = [c for c in a["rules_gate"] if not c.startswith("_")]
    det = lambda aud: [c for c in cls if a[aud].get(c, {}).get("label") == "DETECTS"]
    d = {"rules_gate_detects": det("rules_gate"), "probe_battery_detects": det("probe_battery"),
         "r5b_detects": det("r5b_audit")}
    d["rules_gate_negative"] = len(d["rules_gate_detects"]) < 6
    d["r5b_negative"] = "planted_superposition" not in d["r5b_detects"]
    d["no_single_auditor_covers_all_8"] = not any(len(det(x)) == 8 for x in ("rules_gate", "r5b_audit", "probe_battery"))
    return d


def dumps(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n"
