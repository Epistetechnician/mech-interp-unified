#!/usr/bin/env python3
"""R5: can the neuron-audit pipeline see planted superposition? Protocol: docs/protocols/R5-planted-superposition.md"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

N_FEAT, D_HID = 20, 10
SPARSITIES = [0.0, 0.5, 0.9, 0.95]
SEEDS = list(range(10))
TAUS = [0.1, 0.2, 0.3, 0.4]
PRIMARY_TAU = 0.2
STEPS, BATCH, LR, EVAL_N = 6000, 1024, 1e-3, 20000


def sample(n, s, gen):
    active = (torch.rand(n, N_FEAT, generator=gen) > s).float()
    return active * torch.rand(n, N_FEAT, generator=gen)


def train(s, seed):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(D_HID, N_FEAT) * 0.3)
    b = torch.nn.Parameter(torch.zeros(N_FEAT))
    opt = torch.optim.Adam([W, b], lr=LR)
    for _ in range(STEPS):
        x = sample(BATCH, s, gen)
        y = torch.relu(x @ W.T @ W + b)
        loss = ((y - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W.detach(), float(loss)


def audit(W, s, seed):
    gen = torch.Generator().manual_seed(10_000 + seed)
    x = sample(EVAL_N, s, gen)
    h = (x @ W.T).numpy()
    xs = x.numpy()
    hc = (h - h.mean(0)) / (h.std(0) + 1e-12)
    xc = (xs - xs.mean(0)) / (xs.std(0) + 1e-12)
    corr = hc.T @ xc / EVAL_N  # (D_HID, N_FEAT)
    out = {}
    for tau in TAUS:
        jobs = (corr >= tau).sum(1)
        out[tau] = {
            "one": float((jobs == 1).mean()),
            "multi": float((jobs >= 2).mean()),
            "unexplained": float((jobs == 0).mean()),
        }
    return out


def main():
    res = {"trained": {}, "random": {}}
    for s in SPARSITIES:
        res["trained"][s], res["random"][s] = [], []
        for seed in SEEDS:
            W, loss = train(s, seed)
            a = audit(W, s, seed)
            res["trained"][s].append({"seed": seed, "final_loss": loss, "audit": a})
            rg = torch.Generator().manual_seed(500 + seed)
            Wr = torch.randn(D_HID, N_FEAT, generator=rg)
            Wr = Wr * (W.norm(dim=0, keepdim=True) / Wr.norm(dim=0, keepdim=True))
            res["random"][s].append({"seed": seed, "audit": audit(Wr, s, seed)})
        print("S", s, "done", flush=True)

    def mean_multi(arm, s, tau=PRIMARY_TAU):
        return float(np.mean([r["audit"][tau]["multi"] for r in res[arm][s]]))

    m = {s: mean_multi("trained", s) for s in SPARSITIES}
    mr = {s: mean_multi("random", s) for s in SPARSITIES}
    h1 = m[0.95] - m[0.0] >= 0.30
    h2 = all(m[SPARSITIES[i + 1]] >= m[SPARSITIES[i]] - 0.02 for i in range(3))
    h3 = abs(mr[0.0] - m[0.0]) <= 0.15
    verdict = {
        "mean_multi_trained": m,
        "mean_multi_random": mr,
        "H1_power": bool(h1),
        "H2_monotone": bool(h2),
        "H3_random_arm_specific": bool(h3),
        "detects_planted_superposition": bool(h1 and h2),
        "sensitivity": {
            str(t): {str(s): mean_multi("trained", s, t) for s in SPARSITIES} for t in TAUS
        },
    }
    res["verdict"] = verdict
    out = Path(__file__).resolve().parent.parent / "results" / "r5_planted_superposition.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    sys.exit(main())
