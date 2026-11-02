#!/usr/bin/env python3
"""R5b: causal share-based audit on a privileged-basis toy model. Protocol: docs/protocols/R5b-causal-audit-privileged-basis.md"""
import json
from pathlib import Path

import numpy as np
import torch

CONDS = {
    "C1_n10d10_S0.0": (10, 10, 0.0), "C2_n10d10_S0.9": (10, 10, 0.9),
    "C3_n20d10_S0.0": (20, 10, 0.0), "C4_n20d10_S0.5": (20, 10, 0.5),
    "C5_n20d10_S0.9": (20, 10, 0.9), "C6_n20d10_S0.95": (20, 10, 0.95),
}
SEEDS = range(10)
STEPS, BATCH, LR, EVAL_N = 6000, 1024, 1e-3, 20000
SHARES = [0.1, 0.2, 0.3]
PRIMARY = 0.2
GT_Q2 = 0.15
OUT = Path(__file__).resolve().parent.parent / "results" / "r5b_causal_audit.json"


def sample(n_feat, n, s, gen):
    return (torch.rand(n, n_feat, generator=gen) > s).float() * torch.rand(n, n_feat, generator=gen)


def train(n_feat, d, s, seed):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(d, n_feat) * 0.3)
    b = torch.nn.Parameter(torch.zeros(n_feat))
    opt = torch.optim.Adam([W, b], lr=LR)
    for _ in range(STEPS):
        x = sample(n_feat, BATCH, s, gen)
        y = torch.relu(torch.relu(x @ W.T) @ W + b)
        loss = ((y - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W.detach(), b.detach()


def run_model(W, b, n_feat, s, seed):
    gen = torch.Generator().manual_seed(10_000 + seed)
    x = sample(n_feat, EVAL_N, s, gen)
    pre = x @ W.T
    h = torch.relu(pre)
    base_err = ((torch.relu(h @ W + b) - x) ** 2).mean(0)  # per feature
    d = W.shape[0]
    audits = {sh: [] for sh in SHARES}
    gt, used = [], []
    corr_multi = []
    xc = (x - x.mean(0)) / (x.std(0) + 1e-12)
    for j in range(d):
        ha = h.clone()
        ha[:, j] = 0
        D = (((torch.relu(ha @ W + b) - x) ** 2).mean(0) - base_err).clamp(min=0)
        tot = float(D.sum())
        used.append(tot >= 1e-6)
        for sh in SHARES:
            audits[sh].append(int(((D / max(tot, 1e-12)) >= sh).sum()) if tot >= 1e-6 else 0)
        q = W[j] ** 2
        q = (q / q.sum()).sort(descending=True).values
        gt.append(bool(q[1] >= GT_Q2))
        pj = pre[:, j]
        pc = (pj - pj.mean()) / (pj.std() + 1e-12)
        corr = (pc[:, None] * xc).mean(0)
        corr_multi.append(int((corr >= 0.2).sum()) >= 2)
    return {"audit_jobs": {str(k): v for k, v in audits.items()}, "gt_multi": gt, "used": used,
            "corr_multi": corr_multi}


def main():
    res = {c: [] for c in CONDS}
    for c, (n_feat, d, s) in CONDS.items():
        for seed in SEEDS:
            W, b = train(n_feat, d, s, seed)
            res[c].append(run_model(W, b, n_feat, s, seed))
        print(c, "done", flush=True)

    def frac(c, sh):
        v = [(jobs >= 2) for r in res[c] for jobs, u in zip(r["audit_jobs"][str(sh)], r["used"]) if u]
        return float(np.mean(v)) if v else float("nan")

    primary = {c: frac(c, PRIMARY) for c in CONDS}
    tp = tn = fp = fn = 0
    for c in CONDS:
        for r in res[c]:
            for jobs, g, u in zip(r["audit_jobs"][str(PRIMARY)], r["gt_multi"], r["used"]):
                if not u:
                    continue
                a = jobs >= 2
                tp += a and g
                tn += (not a) and (not g)
                fp += a and (not g)
                fn += (not a) and g
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    bal = (tpr + tnr) / 2
    corr_frac = {c: float(np.mean([m for r in res[c] for m in r["corr_multi"]])) for c in CONDS}
    gt_frac = {c: float(np.mean([g for r in res[c] for g, u in zip(r["gt_multi"], r["used"]) if u])) for c in CONDS}
    h1 = primary["C6_n20d10_S0.95"] - primary["C1_n10d10_S0.0"] >= 0.30
    h2 = bal >= 0.75
    h3 = primary["C1_n10d10_S0.0"] <= 0.20
    verdict = {
        "audit_multi_fraction": primary, "ground_truth_multi_fraction": gt_frac,
        "corr_audit_multi_fraction": corr_frac,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "tpr": tpr, "tnr": tnr},
        "balanced_accuracy": bal,
        "H1_power": bool(h1), "H2_accuracy": bool(h2), "H3_null_control": bool(h3),
        "audit_validated_for_R1": bool(h1 and h2 and h3),
        "share_sensitivity": {str(sh): {c: frac(c, sh) for c in CONDS} for sh in SHARES},
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"verdict": verdict, "runs": res}, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
