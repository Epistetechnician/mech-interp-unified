#!/usr/bin/env python3
"""v2 DEV diagnostic: how well does the simulator score TRUE descriptions? Uses dev slices only. Not a result."""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import lib  # noqa: E402
import mlx_backend as mb  # noqa: E402

MODEL = "/Users/shaanp/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit"
OUT = Path(__file__).resolve().parents[2] / "results" / "r1b2_dev_diag.json"


def oracle_job(p):
    k, spec = p.meta["kind"], p.meta["spec"]
    if k == "set":
        return "the token is exactly one of: " + ", ".join(repr(t) for t in spec)
    if k == "after":
        return f"the token that comes immediately after the token {spec!r}"
    return ("either the token is exactly one of: " + ", ".join(repr(t) for t in spec["set"]) +
            f"; or the token comes immediately after the token {spec['after']!r}")


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    lines = data.load_lines()
    fit, assess = data.pool("dev_fit", lines, tok), data.pool("dev_assess", lines, tok)
    poles = lib.make_rule_poles(fit, assess, np.random.default_rng(200), n_each=6)
    sel = [p for p in poles if p.meta["kind"] in ("set", "after")][:12:1]
    sel = [p for p in poles if p.meta["kind"] == "set"][:6] + [p for p in poles if p.meta["kind"] == "after"][:6]
    for i, p in enumerate(sel):
        p.assess = lib.sample_assess(p.assess, np.random.default_rng(300 + i))
    m = mb.MLXModel(MODEL)
    sim = mb.MLXSimulator(m)
    res = []
    for p in sel:
        job = oracle_job(p)
        t = time.time()
        s = lib.job_scores(sim, job, p.assess)
        x = lib.flat_x(p.assess)
        rec = {"kind": p.meta["kind"], "job": job, "r": lib.pearson(s, x), "s_on": float(s[x > 0].mean()),
               "s_off": float(s[x == 0].mean()), "n_on": int((x > 0).sum()), "sec": round(time.time() - t, 1)}
        res.append(rec)
        print(rec, flush=True)
    OUT.write_text(json.dumps(res, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
