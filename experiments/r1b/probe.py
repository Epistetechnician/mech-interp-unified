#!/usr/bin/env python3
"""R1b-0 timing and load probe. Not a result: synthetic text only, no real poles. Writes results/r1b0_probe.json."""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
import mlx_backend as mb  # noqa: E402

PATHS = {
    "qwen3.6-35b-a3b-4bit": "/Users/shaanp/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit",
}
OUT = Path(__file__).resolve().parents[2] / "results" / "r1b0_probe.json"


def probe(name, path):
    r = {"model": name, "path": path}
    t = time.time()
    m = mb.MLXModel(path)
    r["load_s"] = round(time.time() - t, 2)
    rng = np.random.default_rng(0)
    vocab = ["the", "of", "and", "in", "was", "a", "to", "Colorado", "river", ",", ".", "1998", "city", "he"]
    ctx = lambda: lib.Ctx([vocab[i] for i in rng.integers(0, len(vocab), 64)], np.zeros(64))
    fit = []
    for _ in range(40):
        c = ctx()
        c.x[rng.integers(0, 64, 3)] = rng.random(3) + 0.5
        fit.append(c)
    prompt = lib.explainer_prompt(lib.format_evidence(fit, rng))
    lab = mb.MLXLabeller(m)
    t = time.time()
    jobs = lab.explain(prompt)
    r["explain_s"] = round(time.time() - t, 2)
    r["explain_prompt_chars"] = len(prompt)
    r["explain_jobs_parsed"] = len(jobs)
    sim = mb.MLXSimulator(m)
    toks = ["the", "Colorado", ",", "river", "1998", "was"] * 10 + ["of"] * 4
    sim.score("the token \"Colorado\"", toks)  # warmup
    sim.tokens_seen = 0
    t = time.time()
    n = 5
    for _ in range(n):
        s = sim.score("the token \"Colorado\"", toks)
    dt = (time.time() - t) / n
    r["sim_s_per_64tok_context"] = round(dt, 3)
    r["sim_prompt_tokens_per_context"] = sim.tokens_seen // n
    r["sim_tokens_per_s"] = round(sim.tokens_seen / n / dt, 1)
    r["sim_sanity_colorado_mean"] = float(np.mean([s[i] for i, t_ in enumerate(toks) if t_ == "Colorado"]))
    r["sim_sanity_other_mean"] = float(np.mean([s[i] for i, t_ in enumerate(toks) if t_ != "Colorado"]))
    return r


if __name__ == "__main__":
    out = {"note": "timing/load probe on synthetic text; not a result", "probes": []}
    for k, p in PATHS.items():
        out["probes"].append(probe(k, p))
        print(out["probes"][-1], flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
