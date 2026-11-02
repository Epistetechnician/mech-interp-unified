#!/usr/bin/env python3
"""R4: offshoot growth across Pythia-160m checkpoints. Protocol: docs/protocols/R4-offshoot-trajectory.md"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, GPTNeoXForCausalLM

REPO = "EleutherAI/pythia-160m"
REVS = ["step16000", "step32000", "step64000", "step100000", "step143000"]
FINAL = "step143000"
LAYERS = [3, 6, 9]
N_SEL, MAX_T, MIN_COUNT, REL_FRAC = 100, 15, 5, 0.5
N_FIT, N_ASSESS, SEQ = 400, 400, 128
PERMS, MIN_EVENTS = 1000, 200
NEED_GAP, NEED_P = 0.10, 0.01
THETA_PCT, THETA_TOP = 95, 2000
OUT = Path(__file__).resolve().parent.parent / "results" / "r4_offshoot_trajectory.json"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def load_model(rev, hashes):
    local = None
    for name in ("model.safetensors", "pytorch_model.bin"):
        try:
            local = hf_hub_download(REPO, name, revision=rev)
            hashes[f"{rev}/{name}"] = sha(local)
            break
        except Exception:
            continue
    assert local, rev
    d = str(Path(local).parent)
    for cfg in ("config.json",):
        hf_hub_download(REPO, cfg, revision=rev)
    return GPTNeoXForCausalLM.from_pretrained(d, torch_dtype=torch.float32).eval()


def get_text(hashes):
    p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet",
                        repo_type="dataset")
    hashes["wikitext-2-raw-v1/train"] = sha(p)
    lines = [t.strip() for t in pq.read_table(p).column("text").to_pylist()]
    lines = [t for t in lines if len(t) >= 200]
    return lines[:N_FIT], lines[N_FIT:N_FIT + N_ASSESS]


def batches(tok, texts, bs=16):
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=SEQ, padding=True, return_tensors="pt")
        yield enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def collect(model, tok, texts, sel=None):
    """If sel is None: return per-layer (sum,sumsq,count,max) over all neurons.
    Else: return per-layer sum_by_type[V,100], count_by_type[V] over selected neurons."""
    acts = {}
    hooks = []
    for L in LAYERS:
        hooks.append(model.gpt_neox.layers[L].mlp.act.register_forward_hook(
            lambda m, i, o, L=L: acts.__setitem__(L, o)))
    V = model.config.vocab_size
    st = {L: None for L in LAYERS}
    for ids, mask in batches(tok, texts):
        model(input_ids=ids, attention_mask=mask)
        m = mask.bool().reshape(-1)
        flat_ids = ids.reshape(-1)[m]
        for L in LAYERS:
            a = acts[L].reshape(-1, acts[L].shape[-1])[m]
            if sel is None:
                if st[L] is None:
                    D = a.shape[1]
                    st[L] = [torch.zeros(D), torch.zeros(D), 0, torch.full((D,), -1e9)]
                st[L][0] += a.sum(0)
                st[L][1] += (a ** 2).sum(0)
                st[L][2] += a.shape[0]
                st[L][3] = torch.maximum(st[L][3], a.max(0).values)
            else:
                if st[L] is None:
                    st[L] = [torch.zeros(V, N_SEL), torch.zeros(V)]
                st[L][0].index_add_(0, flat_ids, a[:, sel[L]])
                st[L][1].index_add_(0, flat_ids, torch.ones(a.shape[0]))
    for h in hooks:
        h.remove()
    return st


def token_sets(sum_by, cnt):
    """Per neuron: up to MAX_T token ids with count>=MIN_COUNT and type-mean >= REL_FRAC*max type-mean."""
    mean = sum_by / cnt.clamp(min=1).unsqueeze(1)
    mean[cnt < MIN_COUNT] = -1e9
    sets = []
    for j in range(mean.shape[1]):
        col = mean[:, j]
        mx = col.max().item()
        if mx <= 0:
            sets.append([])
            continue
        top = torch.topk(col, MAX_T)
        sets.append([int(i) for v, i in zip(top.values.tolist(), top.indices.tolist())
                     if v >= REL_FRAC * mx and v > 0])
    return sets


def main():
    hashes = {}
    fit, assess = get_text(hashes)
    tok = AutoTokenizer.from_pretrained(REPO, revision=FINAL)
    tok.pad_token = tok.eos_token

    final = load_model(FINAL, hashes)
    # step 1: neuron selection (fit text, final checkpoint)
    st = collect(final, tok, fit)
    sel = {}
    for L in LAYERS:
        s, s2, n, mx = st[L]
        mean = s / n
        std = (s2 / n - mean ** 2).clamp(min=1e-12).sqrt()
        z = (mx - mean) / std
        sel[L] = torch.topk(z, N_SEL).indices
    # embedding similarity threshold
    E = final.gpt_neox.embed_in.weight.detach()
    En = E / E.norm(dim=1, keepdim=True)
    counts = torch.zeros(E.shape[0])
    for ids, mask in batches(tok, fit):
        counts.index_add_(0, ids.reshape(-1)[mask.bool().reshape(-1)],
                          torch.ones(int(mask.sum())))
    top = torch.topk(counts, THETA_TOP).indices
    C = En[top] @ En[top].T
    iu = torch.triu_indices(THETA_TOP, THETA_TOP, offset=1)
    theta = float(np.percentile(C[iu[0], iu[1]].numpy(), THETA_PCT))
    En = En.numpy()

    # step 2: token sets per checkpoint
    sets = {}
    for rev in REVS:
        model = final if rev == FINAL else load_model(rev, hashes)
        st = collect(model, tok, assess, sel)
        sets[rev] = {L: token_sets(*st[L]) for L in LAYERS}
        if rev != FINAL:
            del model
        print("sets", rev, flush=True)

    dec = {}

    def text(t):
        if t not in dec:
            dec[t] = tok.decode([t]).strip().lower()
        return dec[t]

    rng = np.random.default_rng(0)
    result = {"hashes": hashes, "theta": theta, "layers": {}}
    examples = []
    for L in LAYERS:
        obs_any = obs_sub = obs_emb = 0
        n_events = 0
        null_any = np.zeros(PERMS)
        null_sub = np.zeros(PERMS)
        null_emb = np.zeros(PERMS)
        for a, b in zip(REVS[:-1], REVS[1:]):
            prev, nxt = sets[a][L], sets[b][L]
            valid = np.array([j for j in range(N_SEL) if prev[j]])
            ev_tok, ev_n = [], []
            for j in range(N_SEL):
                if not prev[j]:
                    continue
                for t in nxt[j]:
                    if t not in prev[j]:
                        ev_tok.append(t)
                        ev_n.append(j)
            if not ev_tok or len(valid) < 2:
                continue
            ev_tok = np.array(ev_tok)
            ev_n = np.array(ev_n)
            # relatedness of each event token to each neuron's prior set
            sub = np.zeros((len(ev_tok), N_SEL), bool)
            emb = np.zeros((len(ev_tok), N_SEL), bool)
            for j in valid:
                P = prev[j]
                sims = En[ev_tok] @ En[P].T
                emb[:, j] = sims.max(1) >= theta
                ps = [text(p) for p in P]
                for i, t in enumerate(ev_tok):
                    s = text(int(t))
                    if len(s) >= 2:
                        sub[i, j] = any(len(p) >= 2 and (s in p or p in s) for p in ps)
            anyr = sub | emb
            idx = np.arange(len(ev_tok))
            obs_any += anyr[idx, ev_n].sum()
            obs_sub += sub[idx, ev_n].sum()
            obs_emb += emb[idx, ev_n].sum()
            n_events += len(ev_tok)
            for k in range(PERMS):
                m = valid[rng.integers(0, len(valid), size=len(ev_tok))]
                same = m == ev_n
                while same.any():
                    m[same] = valid[rng.integers(0, len(valid), size=int(same.sum()))]
                    same = m == ev_n
                null_any[k] += anyr[idx, m].sum()
                null_sub[k] += sub[idx, m].sum()
                null_emb[k] += emb[idx, m].sum()
            if L == 6:
                for i in np.where(sub[idx, ev_n])[0][:5]:
                    prior = [text(p) for p in prev[ev_n[i]]]
                    examples.append({"layer": L, "transition": f"{a}->{b}", "neuron": int(sel[L][ev_n[i]]),
                                     "new": text(int(ev_tok[i])), "prior": prior})
        if n_events == 0:
            result["layers"][str(L)] = {"events": 0, "status": "uninformative"}
            continue
        R = obs_any / n_events
        nm = null_any / n_events
        p = (1 + int((nm >= R).sum())) / (PERMS + 1)
        gap = R - nm.mean()
        status = ("uninformative" if n_events < MIN_EVENTS else
                  "supported" if (gap >= NEED_GAP and p <= NEED_P) else "not_supported")
        result["layers"][str(L)] = {
            "events": int(n_events), "R_any": R, "R_null_mean": float(nm.mean()), "gap": float(gap), "p": p,
            "R_sub": obs_sub / n_events, "R_sub_null_mean": float((null_sub / n_events).mean()),
            "R_emb": obs_emb / n_events, "R_emb_null_mean": float((null_emb / n_events).mean()),
            "status": status}
        print("layer", L, result["layers"][str(L)], flush=True)
    sup = sum(1 for v in result["layers"].values() if v["status"] == "supported")
    result["offshoot_growth_supported"] = bool(sup >= 2)
    result["descriptive_examples_layer6"] = examples[:20]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print("supported layers", sup, "->", result["offshoot_growth_supported"])


if __name__ == "__main__":
    main()
