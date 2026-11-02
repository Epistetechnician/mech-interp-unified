#!/usr/bin/env python3
"""R4b: directed growth vs stable coherence. Protocol: docs/protocols/R4b-growth-vs-stable-coherence.md"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r4_offshoot_trajectory as r4  # noqa: E402

THETA = 0.0941653847694397  # frozen from R4
PERMS_A, PERMS_C = 1000, 10000
OUT = Path(__file__).resolve().parent.parent / "results" / "r4b_growth_vs_coherence.json"


def main():
    hashes = {}
    fit, assess = r4.get_text(hashes)
    tok = AutoTokenizer.from_pretrained(r4.REPO, revision=r4.FINAL)
    tok.pad_token = tok.eos_token
    final = r4.load_model(r4.FINAL, hashes)
    st = r4.collect(final, tok, fit)
    sel = {}
    for L in r4.LAYERS:
        s, s2, n, mx = st[L]
        mean = s / n
        std = (s2 / n - mean ** 2).clamp(min=1e-12).sqrt()
        sel[L] = torch.topk((mx - mean) / std, r4.N_SEL).indices
    E = final.gpt_neox.embed_in.weight.detach()
    En = (E / E.norm(dim=1, keepdim=True)).numpy()
    sets = {}
    for rev in r4.REVS:
        model = final if rev == r4.FINAL else r4.load_model(rev, hashes)
        stt = r4.collect(model, tok, assess, sel)
        sets[rev] = {L: r4.token_sets(*stt[L]) for L in r4.LAYERS}
        print("sets", rev, flush=True)
    c = r4.REVS  # c[0..4]

    def related(t, S):
        return bool(S) and float((En[S] @ En[t]).max()) >= THETA

    rng = np.random.default_rng(1)
    out = {"hashes": hashes, "theta": THETA, "layers": {}}
    for L in r4.LAYERS:
        # Test A
        ev = []  # (group id, persistent, related)
        gid = 0
        for k in range(3):
            prev_c, next_c = c[k], c[k + 1]
            for j in range(r4.N_SEL):
                P, N = sets[prev_c][L][j], sets[next_c][L][j]
                if not P:
                    continue
                added = [t for t in N if t not in P]
                if not added:
                    continue
                F = set(sets[c[4]][L][j])
                for t in added:
                    ev.append((gid, t in F, related(t, P)))
                gid += 1
        g = np.array([e[0] for e in ev])
        pers = np.array([e[1] for e in ev])
        rel = np.array([e[2] for e in ev], float)
        nP, nT = int(pers.sum()), int((~pers).sum())
        a = {"events": len(ev), "n_persistent": nP, "n_transient": nT}
        if nP >= 50 and nT >= 50:
            obs = rel[pers].mean() - rel[~pers].mean()
            groups = [np.where(g == u)[0] for u in np.unique(g)]
            null = np.zeros(PERMS_A)
            for p in range(PERMS_A):
                pp = pers.copy()
                for idx in groups:
                    pp[idx] = rng.permutation(pers[idx])
                null[p] = rel[pp].mean() - rel[~pp].mean()
            pv = (1 + int((null >= obs).sum())) / (PERMS_A + 1)
            a.update(gap=float(obs), null_mean=float(null.mean()), p=pv,
                     pass_=bool(obs >= 0.10 and pv <= 0.01))
        else:
            a["pass_"] = False
            a["note"] = "insufficient events in a class"
        # Test C
        Dn = []
        for j in range(r4.N_SEL):
            core = sets[c[0]][L][j]
            ds = []
            for k in range(2, 5):
                prevprev, prev, nxt = sets[c[k - 2]][L][j], sets[c[k - 1]][L][j], sets[c[k]][L][j]
                frontier = [t for t in prev if t not in prevprev]
                if not frontier or not core:
                    continue
                for t in nxt:
                    if t in prev:
                        continue
                    ds.append(float(related(t, frontier)) - float(related(t, core)))
            if ds:
                Dn.append(np.mean(ds))
        Dn = np.array(Dn)
        cres = {"neurons": int(len(Dn))}
        if len(Dn) >= 30:
            obs = float(Dn.mean())
            signs = rng.choice([-1, 1], size=(PERMS_C, len(Dn)))
            nul = (signs * Dn).mean(1)
            pv = (1 + int((np.abs(nul) >= abs(obs)).sum())) / (PERMS_C + 1)
            cres.update(mean_d=obs, p=pv, growth_pass=bool(obs >= 0.05 and pv <= 0.01),
                        stable_pass=bool(obs <= -0.05 and pv <= 0.01))
        else:
            cres.update(growth_pass=False, stable_pass=False, note="fewer than 30 neurons")
        out["layers"][str(L)] = {"A": a, "C": cres}
        print("layer", L, out["layers"][str(L)], flush=True)
    both = sum(1 for v in out["layers"].values() if v["A"].get("pass_") and v["C"]["growth_pass"])
    stable = sum(1 for v in out["layers"].values() if v["C"]["stable_pass"])
    out["verdict"] = ("directed_growth_supported" if both >= 2 else
                      "stable_coherence_favored" if stable >= 2 else "inconclusive")
    out["layers_A_and_C_growth"] = both
    out["layers_C_stable"] = stable
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
    print("VERDICT", out["verdict"])


if __name__ == "__main__":
    main()
