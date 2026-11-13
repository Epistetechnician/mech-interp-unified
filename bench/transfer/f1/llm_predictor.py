"""LLM predictor interface for the F1 bench. Nothing here calls a network or a paid API.

Backends:
  dry   : write the prompts (no label-bearing field) to a JSONL file; nothing is predicted.
  file  : read predictions from a JSONL ({"id":..., "label":...}) produced elsewhere and score them.
A live backend (for example the Anthropic Messages API) is deliberately not implemented here: it
needs a provider call, which AGENTS.md rule 4 forbids without an owner-stated USD ceiling in chat,
a recorded provider identity and an allocation receipt. See bench/transfer/README.md.
"""
from __future__ import annotations

import json

from .baselines import LABELS, score

# Fields that carry or imply the answer; the prompt must never include them.
FORBIDDEN_KEYS = ("label", "label_rule", "label_reason", "label_inputs", "evidence", "notes",
                  "arm_status", "verdict", "confounds", "cluster_id", "id")

SYSTEM = (
    "You judge whether a small-scale causal claim from an interpretability experiment survived a "
    "change of scale, device or checkpoint. Use only the information given. Answer with exactly one "
    "word: replicated, failed, or not_tested."
)

DEFINITIONS = (
    "replicated: the transfer run applied the same frozen procedure and its own preregistered rule "
    "was met. failed: the transfer run was completed and its own preregistered rule was not met. "
    "not_tested: no valid test of the transfer exists (no run, run closed or inconclusive, instrument "
    "invalid, or the original claim was never established)."
)


def _fmt_cond(c, size_b):
    return (f"model={c['model']}; size={c['size']}; device={c['device']}; dtype={c['dtype']}; "
            f"checkpoint={c['checkpoint']}; corpus={c['corpus']}; instrument gate={c['instrument_gate_status']}")


def build_prompt(item) -> dict:
    """Return {'system', 'user'} for one item, using only label-free metadata."""
    base = {k: item[k] for k in ("model", "size", "device", "dtype", "checkpoint", "corpus", "instrument_gate_status")}
    t = item["transfer"]
    tr = {k: t[k] for k in ("model", "size", "device", "dtype", "checkpoint", "corpus", "instrument_gate_status")}
    has_arm = t["run_id"] is not None
    user = [
        f"Claim: {item['claim']['text']}",
        f"Claim family: {item['claim']['family']}",
        f"Established in the base run under: {_fmt_cond(base, item['size_b'])}",
        f"Base run verdict class: {item['base']['verdict_class']}",
        ("Transfer arm conditions: " + _fmt_cond(tr, t["size_b"])) if has_arm else "Transfer arm: none recorded",
        f"Axes changed in the transfer: {', '.join(t['axes_changed']) or 'none'}",
        "", DEFINITIONS, "", "Answer (one word):",
    ]
    return {"system": SYSTEM, "user": "\n".join(user)}


def prompt_has_no_forbidden_values(item, prompt) -> bool:
    """Guard used in tests: the rule id and reason code must not appear in the prompt."""
    text = prompt["user"] + prompt["system"]
    banned = [item["label_rule"], item["label_reason"]]
    return not any(b in text for b in banned)


def write_prompts(items, path):
    with open(path, "w") as f:
        for i in items:
            f.write(json.dumps({"id": i["id"], **build_prompt(i)}, sort_keys=True) + "\n")


def parse_label(text):
    t = text.strip().lower().split()[0].strip(".,:;\"'`") if text.strip() else ""
    return t if t in LABELS else None


def score_predictions(items, preds_by_id):
    """Score predictions. A missing or unparseable prediction counts as wrong."""
    parsed = [parse_label(preds_by_id.get(i["id"], "")) for i in items]
    truth = [i["label"] for i in items]
    k = sum(1 for t, p in zip(truth, parsed) if t == p)
    out = {"n": len(items), "correct": k, "accuracy": k / len(items) if items else float("nan"),
           "unparsed": sum(1 for p in parsed if p is None)}
    if all(p is not None for p in parsed):
        out["detail"] = score(truth, parsed)
    return out
