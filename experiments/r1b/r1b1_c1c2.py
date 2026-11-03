#!/usr/bin/env python3
"""R1b-1: C1 (rule neurons) and C2 (shuffled descriptions) only. Protocol: docs/protocols/R1b-1-protocol.md"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
import mlx_backend as mb  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = "/Users/shaanp/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit"
CTX_LEN, N_FIT, N_ASSESS = 64, 600, 600
FIT_IDX, ASSESS_IDX = (2800, 3400), (3400, 4000)
LOG = ROOT / "results" / "r1b1_poles.jsonl"
OUT = ROOT / "results" / "r1b1_c1c2.json"


def pools():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet", repo_type="dataset")
    lines = [t.strip() for t in pq.read_table(p).column("text").to_pylist()]
    lines = [t for t in lines if len(t) >= 200]

    def build(sl):
        out = []
        for t in lines[sl[0]:sl[1]]:
            ids = tok(t, add_special_tokens=False)["input_ids"][:CTX_LEN]
            if len(ids) >= 32:
                toks = [tok.decode([i]) for i in ids]
                out.append(lib.Ctx(toks, np.zeros(len(toks))))
        return out
    return build(FIT_IDX), build(ASSESS_IDX)


def main():
    fit, assess = pools()
    print("pools", len(fit), len(assess), flush=True)
    poles = lib.make_rule_poles(fit, assess, np.random.default_rng(100), n_each=20)
    for i, p in enumerate(poles):
        p.assess = lib.sample_assess(p.assess, np.random.default_rng(1000 + i))
    m = mb.MLXModel(MODEL)
    lab, sim = mb.MLXLabeller(m), mb.MLXSimulator(m)
    done = {}
    if LOG.exists():
        for ln in LOG.read_text().splitlines():
            r = json.loads(ln)
            done[r["i"]] = r
    LOG.parent.mkdir(exist_ok=True)
    for i, p in enumerate(poles):
        if i in done:
            continue
        r = lib.audit_pole(p, lab, sim, np.random.default_rng(i))
        rec = {"i": i, "truth": p.truth, "meta": p.meta, "jobs": r["jobs"], "r1": r["r1"], "r2": r["r2"],
               "r3": r["r3"], "cls": r["cls"]}
        done[i] = rec
        with LOG.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(i, p.truth, p.meta["kind"], [round(rec[k], 2) for k in ("r1", "r2", "r3")], rec["cls"]["0.5"], flush=True)
    res = [done[i] for i in range(len(poles))]
    c1 = lib.c1_metrics(res, poles)
    out = {"c1": c1, "n_poles": len(poles)}
    if c1["pass"]:
        perm = lib.derangement(len(poles), np.random.default_rng(7))
        r3 = []
        for i, j in enumerate(perm):
            r3.append(lib.score_jobs(sim, done[int(j)]["jobs"], poles[i].assess)["r3"])
            print("c2", i, round(r3[-1], 2), flush=True)
        out["c2"] = lib.c2_metrics(r3)
        out["c2_r3"] = r3
    else:
        out["c2"] = "not run: C1 failed (frozen rule: R1b closes as labeller_too_weak)"
    out["by_kind"] = {k: {"n": sum(1 for r in res if r["meta"]["kind"] == k),
                          "mean_r3": float(np.mean([r["r3"] for r in res if r["meta"]["kind"] == k])),
                          "classes": {c: sum(1 for r in res if r["meta"]["kind"] == k and r["cls"]["0.5"] == c)
                                      for c in ("one", "two", "many", "unexplained")}}
                      for k in ("set", "after", "union")}
    out["verdict_so_far"] = "labeller_too_weak" if not c1["pass"] else ("c1_c2_pass" if out["c2"]["pass"] else "c2_failed")
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
    print(json.dumps(out["by_kind"], indent=1), c1, out["c2"], out["verdict_so_far"])


if __name__ == "__main__":
    main()
