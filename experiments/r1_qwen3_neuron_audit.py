#!/usr/bin/env python3
"""R1: calibrated token-level neuron-pole audit on Qwen3-1.7B-Base. Protocol: docs/protocols/R1-qwen3-neuron-audit.md"""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from scipy.cluster.hierarchy import fcluster, linkage
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "Qwen/Qwen3-1.7B-Base"
LAYERS = [4, 10, 16, 22, 27]
N_PER_LAYER, SEQ, BS = 20, 128, 8
FIT_IDX, ASSESS_IDX = (1600, 2200), (2200, 2800)
Z_MIN, TOP_FRAC, MASS, MAX_TYPES = 4.0, 0.01, 0.8, 30
R_TOK_MIN, MIN_COUNT = 0.3, 3
PCTS = [90, 95, 98]
PRIMARY = 95
OUT = Path(__file__).resolve().parent.parent / "results" / "r1_qwen3_neuron_audit.json"
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def get_text():
    p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet", repo_type="dataset")
    lines = [t.strip() for t in pq.read_table(p).column("text").to_pylist()]
    lines = [t for t in lines if len(t) >= 200]
    return lines[FIT_IDX[0]:FIT_IDX[1]], lines[ASSESS_IDX[0]:ASSESS_IDX[1]]


@torch.no_grad()
def collect(model, tok, texts, sel):
    store = {L: [] for L in LAYERS}
    ids_all = []
    cur = {}
    hooks = [model.model.layers[L].mlp.down_proj.register_forward_pre_hook(
        lambda m, i, L=L: cur.__setitem__(L, i[0])) for L in LAYERS]
    for i in range(0, len(texts), BS):
        enc = tok(texts[i:i + BS], truncation=True, max_length=SEQ, padding=True, return_tensors="pt")
        model(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev))
        m = enc["attention_mask"].bool().reshape(-1)
        ids_all.append(enc["input_ids"].reshape(-1)[m].numpy())
        for L in LAYERS:
            a = cur[L].float().reshape(-1, cur[L].shape[-1]).cpu()[m]
            store[L].append(a[:, sel[L]].numpy())
    for h in hooks:
        h.remove()
    return np.concatenate(ids_all), {L: np.concatenate(store[L]) for L in LAYERS}


def classify(x_derive, ids_derive, x_score, ids_score, En, theta):
    n = len(x_derive)
    k_top = max(200, int(TOP_FRAC * n))
    top = np.argsort(-x_derive)[:k_top]
    mass = {}
    for p in top:
        mass[ids_derive[p]] = mass.get(ids_derive[p], 0.0) + float(x_derive[p])
    items = sorted(mass.items(), key=lambda kv: -kv[1])
    tot = sum(v for _, v in items)
    types, acc = [], 0.0
    for t, v in items:
        types.append(t)
        acc += v
        if acc >= MASS * tot or len(types) >= MAX_TYPES:
            break
    if len(types) == 1:
        k = 1
    else:
        Z = linkage(En[types], method="average", metric="cosine")
        k = int(fcluster(Z, t=1 - theta, criterion="distance").max())
    # token-lookup simulator
    V = En.shape[0]
    s = np.zeros(V)
    c = np.zeros(V)
    np.add.at(s, ids_derive, x_derive)
    np.add.at(c, ids_derive, 1)
    g = x_derive.mean()
    lookup = np.where(c >= MIN_COUNT, s / np.maximum(c, 1), g)
    pred = lookup[ids_score]
    r = float(np.corrcoef(pred, x_score)[0, 1]) if x_score.std() > 0 and pred.std() > 0 else 0.0
    if r < R_TOK_MIN:
        cls = "token_unexplained"
    else:
        cls = "one" if k == 1 else "two" if k == 2 else "many"
    return cls, k, r, types


def main():
    fit, assess = get_text()
    tok = AutoTokenizer.from_pretrained(REPO)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16).to(dev).eval()
    inter = model.config.intermediate_size
    rng = np.random.default_rng(0)
    sel = {L: np.sort(rng.choice(inter, N_PER_LAYER, replace=False)) for L in LAYERS}
    E = model.get_input_embeddings().weight.detach().float().cpu()
    En = (E / E.norm(dim=1, keepdim=True)).numpy()
    ids_f, A_f = collect(model, tok, fit, sel)
    ids_a, A_a = collect(model, tok, assess, sel)
    del model
    print("collected", len(ids_f), len(ids_a), flush=True)

    freq = np.bincount(ids_f, minlength=En.shape[0])
    top = np.argsort(-freq)[:2000]
    C = En[top] @ En[top].T
    iu = np.triu_indices(2000, 1)
    thetas = {P: float(np.percentile(C[iu], P)) for P in PCTS}

    poles = []
    for L in LAYERS:
        for jj, j in enumerate(sel[L]):
            af, aa = A_f[L][:, jj], A_a[L][:, jj]
            mu, sd = af.mean(), af.std() + 1e-12
            for sign in (1, -1):
                z = (sign * (af - mu)) / sd
                if z.max() < Z_MIN:
                    continue
                poles.append({"layer": L, "neuron": int(j), "sign": sign,
                              "xf": np.maximum(sign * af, 0), "xa": np.maximum(sign * aa, 0)})
    print("valid poles", len(poles), "of", 2 * len(LAYERS) * N_PER_LAYER, flush=True)

    res = {"n_valid_poles": len(poles), "n_possible_poles": 2 * len(LAYERS) * N_PER_LAYER,
           "thetas": thetas, "by_percentile": {}, "model": REPO}
    cls_primary = None
    for P in PCTS:
        th = thetas[P]
        rows = []
        for p in poles:
            c1, k1, r1, T1 = classify(p["xf"], ids_f, p["xa"], ids_a, En, th)
            c2, _, _, _ = classify(p["xa"], ids_a, p["xf"], ids_f, En, th)
            rows.append({"layer": p["layer"], "neuron": p["neuron"], "sign": p["sign"], "class": c1,
                         "clusters": k1, "r_tok": r1, "retest_class": c2, "types": T1})
        fr = {c: float(np.mean([r["class"] == c for r in rows])) for c in ("one", "two", "many", "token_unexplained")}
        agree = float(np.mean([r["class"] == r["retest_class"] for r in rows]))
        # composites among one-poles in same layer with distinct jobs
        ones = [(i, r) for i, r in enumerate(rows) if r["class"] == "one"]
        pairs = []
        for a in range(len(ones)):
            for b in range(a + 1, len(ones)):
                (ia, ra), (ib, rb) = ones[a], ones[b]
                if ra["layer"] != rb["layer"]:
                    continue
                if float((En[ra["types"]] @ En[rb["types"]].T).max()) < th:
                    pairs.append((ia, ib))
        rng2 = np.random.default_rng(1)
        rng2.shuffle(pairs)
        pairs = pairs[:100]
        det = []
        for ia, ib in pairs:
            xa_c = np.maximum(poles[ia]["xf"], poles[ib]["xf"])
            xb_c = np.maximum(poles[ia]["xa"], poles[ib]["xa"])
            c, _, _, _ = classify(xa_c, ids_f, xb_c, ids_a, En, th)
            det.append(c in ("two", "many"))
        sens = float(np.mean(det)) if det else None
        valid = bool(len(pairs) >= 30 and sens is not None and sens >= 0.8 and agree >= 0.8)
        per_layer = {str(L): {c: float(np.mean([r["class"] == c for r in rows if r["layer"] == L] or [np.nan]))
                              for c in ("one", "two", "many", "token_unexplained")} for L in LAYERS}
        res["by_percentile"][str(P)] = {
            "fractions": fr, "test_retest_agreement": agree, "composite_pairs": len(pairs),
            "sensitivity": sens, "validated": valid, "per_layer": per_layer,
            "essay_consistent": bool(valid and 0.80 <= fr["one"] <= 0.97 and fr["token_unexplained"] <= 0.08)}
        print("P", P, res["by_percentile"][str(P)]["fractions"], "retest", agree, "sens", sens, "pairs", len(pairs), flush=True)
        if P == PRIMARY:
            cls_primary = rows
    res["primary_rows"] = [{k: v for k, v in r.items() if k != "types"} for r in cls_primary]
    prim = res["by_percentile"][str(PRIMARY)]
    res["verdict"] = {"validated": prim["validated"], "essay_consistent": prim["essay_consistent"]}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + "\n")
    print("VERDICT", res["verdict"])


if __name__ == "__main__":
    main()
