#!/usr/bin/env python3
"""R4c: birth-order proximity vs token similarity. Protocol: docs/protocols/R4c-birth-order-similarity.md"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r4_offshoot_trajectory as r4  # noqa: E402

REVS = ["step1000", "step2000", "step4000", "step8000", "step16000", "step24000", "step32000",
        "step48000", "step64000", "step80000", "step100000", "step120000", "step143000"]
PERMS = 1000
MIN_TOK, MIN_BIRTHS, MIN_NEURONS = 5, 3, 30
GROWTH_RHO, STABLE_RHO = -0.05, 0.03
OUT = Path(__file__).resolve().parent.parent / "results" / "r4c_birth_order_similarity.json"


def spearman(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    r4.REVS = REVS
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
    E_in = final.gpt_neox.embed_in.weight.detach()
    E_out = final.embed_out.weight.detach()
    En_in = (E_in / E_in.norm(dim=1, keepdim=True)).numpy()
    En_out = (E_out / E_out.norm(dim=1, keepdim=True)).numpy()
    counts = torch.zeros(E_in.shape[0])
    for ids, mask in r4.batches(tok, fit):
        counts.index_add_(0, ids.reshape(-1)[mask.bool().reshape(-1)], torch.ones(int(mask.sum())))
    counts = counts.numpy()
    seen = counts[counts >= 1]
    e1, e2 = np.quantile(seen, [1 / 3, 2 / 3])
    fbin = np.where(counts >= e2, 2, np.where(counts >= e1, 1, 0))

    sets = {}
    for rev in REVS:
        model = final if rev == r4.FINAL else r4.load_model(rev, hashes)
        stt = r4.collect(model, tok, assess, sel)
        sets[rev] = {L: r4.token_sets(*stt[L]) for L in r4.LAYERS}
        if rev != r4.FINAL:
            del model
        print("sets", rev, flush=True)

    rng = np.random.default_rng(3)
    result = {"hashes": hashes, "layers": {}}
    for L in r4.LAYERS:
        neurons = []
        for j in range(r4.N_SEL):
            F = sets[r4.FINAL][L][j]
            if len(F) < MIN_TOK:
                continue
            birth = []
            for t in F:
                b = next((i for i, rv in enumerate(REVS) if t in sets[rv][L][j]), None)
                birth.append(b)
            birth = np.array(birth)
            if len(set(birth.tolist())) < MIN_BIRTHS:
                continue
            F = np.array(F)
            iu = np.triu_indices(len(F), 1)
            cos_in = (En_in[F] @ En_in[F].T)[iu]
            cos_out = (En_out[F] @ En_out[F].T)[iu]
            neurons.append((F, birth, iu, cos_in, cos_out))

        def layer_rho(births_per_neuron, which):
            rhos = []
            for (F, _b, iu, ci, co), b in zip(neurons, births_per_neuron):
                d = np.abs(b[iu[0]] - b[iu[1]])
                m = d > 0
                if m.sum() < 3:
                    continue
                r = spearman((ci if which == "in" else co)[m], d[m])
                if not np.isnan(r):
                    rhos.append(r)
            return rhos

        obs_list = layer_rho([n[1] for n in neurons], "in")
        obs_out = layer_rho([n[1] for n in neurons], "out")
        info = {"qualifying_neurons": len(obs_list)}
        if len(obs_list) >= MIN_NEURONS:
            obs = float(np.mean(obs_list))
            null = np.zeros(PERMS)
            strata = [[np.where(fbin[F] == k)[0] for k in range(3)] for F, *_ in neurons]
            for p in range(PERMS):
                perm_b = []
                for (F, b, *_), st_ in zip(neurons, strata):
                    nb = b.copy()
                    for idx in st_:
                        if len(idx) > 1:
                            nb[idx] = b[idx][rng.permutation(len(idx))]
                    perm_b.append(nb)
                r = layer_rho(perm_b, "in")
                null[p] = np.mean(r) if r else np.nan
            null = null[~np.isnan(null)]
            pv = (1 + int((null <= obs).sum())) / (len(null) + 1)
            info.update(mean_rho=obs, null_mean=float(null.mean()), null_sd=float(null.std()), p=pv,
                        mean_rho_output_embedding=float(np.mean(obs_out)) if obs_out else None,
                        growth_pass=bool(obs <= GROWTH_RHO and pv <= 0.01),
                        stable_pass=bool(abs(obs) < STABLE_RHO and pv > 0.05))
        else:
            info.update(growth_pass=False, stable_pass=False, note="fewer than 30 qualifying neurons")
        result["layers"][str(L)] = info
        print("layer", L, info, flush=True)
    g = sum(v["growth_pass"] for v in result["layers"].values())
    s = sum(v["stable_pass"] for v in result["layers"].values())
    result["verdict"] = ("directed_growth_supported" if g >= 2 else
                         "stable_coherence_supported" if s >= 2 else "inconclusive")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True, default=float) + "\n")
    print("VERDICT", result["verdict"])


if __name__ == "__main__":
    main()
