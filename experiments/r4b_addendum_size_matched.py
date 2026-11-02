#!/usr/bin/env python3
"""R4b addendum (POST-HOC, descriptive, not decision-bearing): size-matched frontier-vs-core.
The frozen Test C compares relatedness to a small frontier set vs a larger core set, which favors the core by construction
(max-cosine over more tokens). Here the core is subsampled to the frontier's size (200 draws per event)."""
import json, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r4_offshoot_trajectory as r4
THETA = 0.0941653847694397
rng = np.random.default_rng(2)
hashes = {}
fit, assess = r4.get_text(hashes)
tok = AutoTokenizer.from_pretrained(r4.REPO, revision=r4.FINAL); tok.pad_token = tok.eos_token
final = r4.load_model(r4.FINAL, hashes)
st = r4.collect(final, tok, fit); sel = {}
for L in r4.LAYERS:
    s, s2, n, mx = st[L]; m = s / n
    sel[L] = torch.topk((mx - m) / (s2 / n - m ** 2).clamp(min=1e-12).sqrt(), r4.N_SEL).indices
E = final.gpt_neox.embed_in.weight.detach(); En = (E / E.norm(dim=1, keepdim=True)).numpy()
sets = {}
for rev in r4.REVS:
    model = final if rev == r4.FINAL else r4.load_model(rev, hashes)
    stt = r4.collect(model, tok, assess, sel); sets[rev] = {L: r4.token_sets(*stt[L]) for L in r4.LAYERS}
c = r4.REVS
rel = lambda t, S: bool(S) and float((En[S] @ En[t]).max()) >= THETA
out = {}
for L in r4.LAYERS:
    Dn = []
    for j in range(r4.N_SEL):
        core = sets[c[0]][L][j]; ds = []
        for k in range(2, 5):
            pp, pv, nx = sets[c[k-2]][L][j], sets[c[k-1]][L][j], sets[c[k]][L][j]
            fr = [t for t in pv if t not in pp]
            if not fr or not core: continue
            for t in nx:
                if t in pv: continue
                sub = [rel(t, list(rng.choice(core, size=min(len(fr), len(core)), replace=False))) for _ in range(200)]
                ds.append(float(rel(t, fr)) - float(np.mean(sub)))
        if ds: Dn.append(np.mean(ds))
    Dn = np.array(Dn)
    signs = rng.choice([-1, 1], size=(10000, len(Dn))); nul = (signs * Dn).mean(1)
    out[str(L)] = {"neurons": len(Dn), "mean_d": float(Dn.mean()),
                   "p_two_sided": (1 + int((np.abs(nul) >= abs(Dn.mean())).sum())) / 10001}
    print(L, out[str(L)], flush=True)
p = Path(__file__).resolve().parent.parent / "results" / "r4b_addendum_size_matched.json"
p.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
