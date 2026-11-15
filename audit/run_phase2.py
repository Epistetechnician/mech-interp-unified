"""python3 -m audit.run_phase2 [--n 200] : Phase-2 evaluation (audit/PROTOCOL_PHASE2.md). --check re-derives the saved file."""
import argparse
import json
import sys
import time
from pathlib import Path

from .flaws import PHASE2_CLASSES
from .harness import PROTOCOL_N, dumps, evaluate

OUT = Path(__file__).resolve().parent / "results" / "f2_phase2.json"
MIN_CLAIM_WRONG = 0.2


def run(n: int, seed0: int = 0) -> dict:
    res = evaluate(n, seed0, classes=PHASE2_CLASSES)
    res.pop("decisions")            # Phase-1 rules do not apply
    a = res["auditors"]
    names = [m.NAME for m in PHASE2_CLASSES]
    det = lambda aud: [c for c in names if a[aud][c]["label"] == "DETECTS"]
    wrong = {c: a["always_pass"][c]["tpr_claim_wrong"]["n"] / a["always_pass"][c]["tpr"]["n"] for c in names}
    res["decisions_phase2"] = {
        "probe_battery_detects": det("probe_battery"), "rules_gate_detects": det("rules_gate"),
        "a3_generalises": len(det("probe_battery")) >= 2, "a1_negative": len(det("rules_gate")) < 2,
        "claim_wrong_fraction": wrong,
        "defect_rarely_changes_claim": [c for c in names if wrong[c] < MIN_CLAIM_WRONG],
    }
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=PROTOCOL_N)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        saved = json.loads(OUT.read_text())
        new = json.loads(dumps(run(saved["n_seeds"], saved["seed0"])))
        saved.pop("runtime_s", None)
        ok = saved == new
        print("VERIFIED" if ok else "MISMATCH", OUT)
        return 0 if ok else 1
    t = time.time()
    res = run(a.n)
    res["runtime_s"] = round(time.time() - t, 1)
    OUT.write_text(dumps(res))
    print(f"wrote {OUT} in {res['runtime_s']}s; harness valid={res['harness_validity']['valid']}")
    return 0 if res["harness_validity"]["valid"] else 2


if __name__ == "__main__":
    sys.exit(main())
