"""R1b harness: pure numpy/stdlib (no model imports). Design: docs/protocols/R1b-design.md.

Interfaces
  labeller.explain(evidence_text) -> list[str]            (up to MAX_JOBS job descriptions)
  simulator.score(job, tokens)    -> np.ndarray in [0, 9]  (one score per token string)
"""
import json
import re
from dataclasses import dataclass, field

import numpy as np

THRESHOLDS = (0.3, 0.5, 0.7)
PRIMARY = 0.5
DELTA = 0.05
N_TOP, N_MID, N_RAND = 12, 6, 6
MAX_JOBS = 3
# calibration bars (design section 4)
C1_R3, C1_ONE_RECALL, C1_TWO_DETECT = 0.7, 0.8, 0.8
C2_MEAN_R3, C2_EXPLAINED = 0.1, 0.05
C3_AGREE, C3_CORR = 0.8, 0.8
C4_DETECT, C4_MIN_PAIRS = 0.8, 30


@dataclass
class Ctx:
    tokens: list
    x: np.ndarray


@dataclass
class Pole:
    layer: int
    neuron: int
    sign: int
    fit: list
    assess: list
    top_tokens: frozenset = frozenset()
    truth: str = ""
    meta: dict = field(default_factory=dict)


def pearson(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.size != b.size or a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def classify(r1, r2, r3, thr):
    if r3 < thr:
        return "unexplained"
    if r1 >= thr and r3 - r1 < DELTA:
        return "one"
    if r2 >= thr and r3 - r2 < DELTA:
        return "two"
    return "many"


def clean_tok(t):
    t = t.replace("\n", "⏎").replace("\t", "⇥")
    return t if t else "␣"


def top_tokens_of(ctxs, frac=0.01, min_n=20):
    pairs = [(c.x[i], c.tokens[i]) for c in ctxs for i in range(len(c.tokens))]
    pairs.sort(key=lambda p: -p[0])
    k = max(min_n, int(frac * len(pairs)))
    return frozenset(t for v, t in pairs[:k] if v > 0)


def select_evidence(ctxs, rng):
    """12 top, 6 mid, 6 random contexts (by context max activation)."""
    gmax = max(float(c.x.max()) for c in ctxs)
    mx = np.array([float(c.x.max()) for c in ctxs])
    order = list(np.argsort(-mx))
    top = order[:N_TOP]
    rest = [i for i in order if i not in top]
    mid_pool = [i for i in rest if 0.25 * gmax <= mx[i] <= 0.6 * gmax]
    mid = [int(i) for i in rng.permutation(mid_pool)[:N_MID]] if mid_pool else []
    rest2 = [i for i in rest if i not in mid]
    rand = [int(i) for i in rng.permutation(rest2)[:N_RAND]]
    return [(c, kind) for kind, idx in (("top", top), ("mid", mid), ("random", rand))
            for c in (ctxs[i] for i in idx)], gmax


def format_evidence(ctxs, rng):
    sel, gmax = select_evidence(ctxs, rng)
    lines = []
    for n, (c, kind) in enumerate(sel):
        parts = []
        for t, v in zip(c.tokens, c.x):
            s = int(round(10 * float(v) / gmax)) if gmax > 0 else 0
            parts.append(f"«{clean_tok(t)}|{s}»" if s >= 1 else clean_tok(t))
        lines.append(f"[{kind} {n}] " + " ".join(parts))
    return "\n".join(lines)


EXPLAIN_HEADER = (
    "A neuron in a language model fires on certain tokens. Below are text contexts. Tokens the neuron "
    "fires on are marked «token|strength» with strength 1-10; unmarked tokens are near zero.\n"
    "Write up to 3 SHORT, SEPARATE jobs (conditions on the token or its context) that together explain "
    "when the neuron fires. If one job is enough, write one. Reply with JSON only: "
    '{"jobs": ["...", "..."]}\n\n')


def explainer_prompt(evidence_text):
    return EXPLAIN_HEADER + evidence_text


def parse_jobs(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    jobs = []
    if m:
        try:
            obj = json.loads(m.group(0))
            raw = obj.get("jobs", [])
            jobs = [j.strip() for j in raw if isinstance(j, str) and j.strip()] if isinstance(raw, list) else []
        except (json.JSONDecodeError, AttributeError):
            jobs = []
    return jobs[:MAX_JOBS]


def flat_x(ctxs):
    return np.concatenate([c.x for c in ctxs])


def job_scores(simulator, job, ctxs):
    return np.concatenate([np.asarray(simulator.score(job, c.tokens), float) for c in ctxs])


def score_jobs(simulator, jobs, ctxs):
    """r1, r2, r3 and the per-job score matrix; absent jobs reuse the last available signal."""
    x = flat_x(ctxs)
    if not jobs:
        return {"r1": 0.0, "r2": 0.0, "r3": 0.0, "scores": np.zeros((0, x.size))}
    S = np.stack([job_scores(simulator, j, ctxs) for j in jobs[:MAX_JOBS]])
    r = [pearson(S[:k].max(0), x) for k in range(1, S.shape[0] + 1)]
    while len(r) < 3:
        r.append(r[-1])
    return {"r1": r[0], "r2": r[1], "r3": r[2], "scores": S}


def audit_pole(pole, labeller, simulator, rng):
    jobs = labeller.explain(explainer_prompt(format_evidence(pole.fit, rng)))
    out = score_jobs(simulator, jobs, pole.assess)
    out["jobs"] = jobs
    out["cls"] = {str(t): classify(out["r1"], out["r2"], out["r3"], t) for t in THRESHOLDS}
    return out


def derangement(n, rng):
    if n < 2:
        raise ValueError("need at least 2 poles")
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


# ---- calibration ---------------------------------------------------------------------------------
def _count(pool):
    cnt = {}
    for c in pool:
        for t in c.tokens:
            cnt[t] = cnt.get(t, 0) + 1
    return cnt


def _rule_x(tokens, kind, spec):
    if kind == "set":
        return np.array([1.0 if t in spec else 0.0 for t in tokens])
    prev = [None] + list(tokens[:-1])
    return np.array([1.0 if p == spec else 0.0 for p in prev])


def make_rule_poles(fit_pool, assess_pool, rng, n_each=20, set_size=6, min_cover=30):
    """Synthetic poles with a known rule: 'set' and 'after' (one job) and 'union' of set+after (two)."""
    cnt = _count(fit_pool)
    cnt_a = _count(assess_pool)
    common = [t for t, c in cnt.items() if c >= 8 and cnt_a.get(t, 0) >= 3]
    if len(common) < 4 * set_size:
        raise ValueError("not enough common tokens for rule poles")
    poles = []

    def apply(pool, kind, spec):
        return [Ctx(c.tokens, _rule_x(c.tokens, kind, spec)) for c in pool]

    def pick(kind):
        for _ in range(200):
            spec = (frozenset(rng.choice(common, set_size, replace=False).tolist()) if kind == "set"
                    else str(rng.choice(common)))
            if sum(c.x.sum() for c in apply(fit_pool, kind, spec)) >= min_cover:
                return spec
        raise ValueError("could not find a covering rule")

    for truth, kinds in (("one", ["set"] * n_each), ("one", ["after"] * n_each), ("two", ["union"] * n_each)):
        for kind in kinds:
            if kind == "union":
                s, a = pick("set"), pick("after")
                mk = lambda pool: [Ctx(c.tokens, np.maximum(_rule_x(c.tokens, "set", s), _rule_x(c.tokens, "after", a)))
                                   for c in pool]
                fit, assess, spec = mk(fit_pool), mk(assess_pool), {"set": sorted(s), "after": a}
            else:
                spec = pick(kind)
                fit, assess = apply(fit_pool, kind, spec), apply(assess_pool, kind, spec)
                spec = sorted(spec) if kind == "set" else spec
            poles.append(Pole(-1, len(poles), +1, fit, assess, top_tokens_of(fit), truth, {"kind": kind, "spec": spec}))
    return poles


def composite(a, b):
    mk = lambda ca, cb: [Ctx(x.tokens, np.maximum(x.x, y.x)) for x, y in zip(ca, cb)]
    fit, assess = mk(a.fit, b.fit), mk(a.assess, b.assess)
    return Pole(a.layer, -1, +1, fit, assess, a.top_tokens | b.top_tokens, "two", {"parts": [a.neuron, b.neuron]})


def make_composites(one_poles, rng, max_pairs=60):
    pairs = []
    by_layer = {}
    for p in one_poles:
        by_layer.setdefault(p.layer, []).append(p)
    for pl in by_layer.values():
        idx = [(i, j) for i in range(len(pl)) for j in range(i + 1, len(pl)) if not (pl[i].top_tokens & pl[j].top_tokens)]
        for k in rng.permutation(len(idx)):
            pairs.append(composite(pl[idx[k][0]], pl[idx[k][1]]))
    return [pairs[i] for i in rng.permutation(len(pairs))[:max_pairs]]


def c1_metrics(results, poles):
    t = str(PRIMARY)
    singles = [(r, p) for r, p in zip(results, poles) if p.truth == "one"]
    unions = [(r, p) for r, p in zip(results, poles) if p.truth == "two"]
    m = {"mean_r3_singles": float(np.mean([r["r3"] for r, _ in singles])),
         "one_recall": float(np.mean([r["cls"][t] == "one" for r, _ in singles])),
         "two_detect": float(np.mean([r["cls"][t] == "two" for r, _ in unions]))}
    m["pass"] = bool(m["mean_r3_singles"] >= C1_R3 and m["one_recall"] >= C1_ONE_RECALL and m["two_detect"] >= C1_TWO_DETECT)
    return m


def c2_metrics(r3_shuffled):
    r3 = np.asarray(r3_shuffled, float)
    m = {"mean_r3": float(r3.mean()), "frac_explained": float((r3 >= PRIMARY).mean())}
    m["pass"] = bool(m["mean_r3"] <= C2_MEAN_R3 and m["frac_explained"] < C2_EXPLAINED)
    return m


def c3_metrics(cls_a, cls_b, scores_primary, scores_second):
    agree = float(np.mean([a == b for a, b in zip(cls_a, cls_b)]))
    corr = pearson(scores_primary, scores_second)
    return {"class_agreement": agree, "simulator_corr": corr, "pass": bool(agree >= C3_AGREE and corr >= C3_CORR)}


def c4_metrics(classes):
    if len(classes) < C4_MIN_PAIRS:
        return {"pairs": len(classes), "detect": None, "pass": False, "note": "insufficient pairs"}
    d = float(np.mean([c in ("two", "many") for c in classes]))
    return {"pairs": len(classes), "detect": d, "pass": bool(d >= C4_DETECT)}


def verdict(c1, c2, c3, c4):
    if not c1["pass"]:
        return "labeller_too_weak"
    return "validated" if all(c["pass"] for c in (c1, c2, c3, c4)) else "calibration_failed"
