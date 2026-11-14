"""Baselines. Rules are fixed here BEFORE they are run; no tuning on the holdout.

Features used by the metadata rules (never label, label_rule, label_reason, label_inputs,
evidence, notes, transfer.verdict, transfer.arm_status):
  transfer.run_id (is there a transfer arm at all), base.verdict_class,
  transfer.instrument_gate_status, transfer.axes_changed, base size_b, transfer size_b.

Caveat: `transfer.run_id is None` and a rejected target gate are the same facts that rules
L0 and L4 key on, so Meta-A/Meta-B recover part of the label *construction*, not of
scientific replicability. The tested-only view (replicated vs failed) removes that part.

majority  : most frequent label in the train split (fit on train only).
meta_a    : not_tested if no transfer arm; not_tested if base verdict_class != supported;
            not_tested if the target gate status mentions "NotValid"; else replicated.
meta_b    : meta_a, then failed if "scale" is an axis changed and both sizes are known
            and the target is smaller than the base.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

LABELS = ("replicated", "failed", "not_tested")


def majority_label(labels):
    c = Counter(labels)
    # deterministic tie-break by LABELS order
    return max(LABELS, key=lambda l: (c.get(l, 0), -LABELS.index(l)))


def meta_a(item) -> str:
    t = item["transfer"]
    if t["run_id"] is None:
        return "not_tested"
    if item["base"]["verdict_class"] != "supported":
        return "not_tested"
    if "NotValid" in t["instrument_gate_status"]:
        return "not_tested"
    return "replicated"


def meta_b(item) -> str:
    p = meta_a(item)
    if p != "replicated":
        return p
    t = item["transfer"]
    if "scale" in t["axes_changed"] and item["size_b"] is not None and t["size_b"] is not None \
            and t["size_b"] < item["size_b"]:
        return "failed"
    return "replicated"


RULES = {"meta_a": meta_a, "meta_b": meta_b}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def confusion(truth, pred, labels=LABELS):
    m = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(truth, pred):
        m[t][p] += 1
    return m


def macro_f1(truth, pred, labels=LABELS):
    """Macro-F1 over labels that occur in truth or prediction (absent labels are skipped)."""
    present = [l for l in labels if l in set(truth) | set(pred)]
    f1s = []
    for l in present:
        tp = sum(1 for t, p in zip(truth, pred) if t == l and p == l)
        fp = sum(1 for t, p in zip(truth, pred) if t != l and p == l)
        fn = sum(1 for t, p in zip(truth, pred) if t == l and p != l)
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s) if f1s else float("nan")


def score(truth, pred):
    n = len(truth)
    k = sum(1 for t, p in zip(truth, pred) if t == p)
    lo, hi = wilson(k, n)
    return {"n": n, "correct": k, "accuracy": (k / n) if n else float("nan"),
            "accuracy_wilson95": [round(lo, 4), round(hi, 4)],
            "macro_f1": macro_f1(truth, pred), "confusion": confusion(truth, pred)}


def tested_only(items, preds):
    """Binary replicated-vs-failed on items whose truth is not not_tested. A not_tested
    prediction on such an item counts as wrong."""
    rows = [(i["label"], p) for i, p in zip(items, preds) if i["label"] != "not_tested"]
    return score([t for t, _ in rows], [p for _, p in rows])


def evaluate(items, train_ids, holdout_ids):
    by = {i["id"]: i for i in items}
    train = [by[i] for i in sorted(train_ids)]
    hold = [by[i] for i in sorted(holdout_ids)]
    maj = majority_label([i["label"] for i in train])
    out = {"majority_label_train": maj, "methods": {}}

    def run(name, predict):
        res = {}
        for split, rows in (("train", train), ("holdout", hold), ("all", train + hold)):
            preds = [predict(i) for i in rows]
            res[split] = score([i["label"] for i in rows], preds)
            res[split]["tested_only"] = tested_only(rows, preds)
        return res

    out["methods"]["majority"] = run("majority", lambda i: maj)
    for name, fn in RULES.items():
        out["methods"][name] = run(name, fn)
    out["methods"]["majority"]["leave_one_cluster_out"] = loco_majority(items)
    return out


def loco_majority(items):
    """Majority label fitted on all other clusters, scored on the held-out cluster (cluster-grouped
    sensitivity to the leakage that a hash-of-id split allows)."""
    clusters = defaultdict(list)
    for i in items:
        clusters[i["cluster_id"]].append(i)
    truth, pred = [], []
    for c, rows in clusters.items():
        rest = [i["label"] for k, v in clusters.items() if k != c for i in v]
        m = majority_label(rest)
        truth += [i["label"] for i in rows]
        pred += [m] * len(rows)
    s = score(truth, pred)
    s["tested_only"] = None
    return s
