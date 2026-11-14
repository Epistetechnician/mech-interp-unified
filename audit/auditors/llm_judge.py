"""A4: LLM judge interface. Prompt builder, response parser, auditor wrapper. NOT RUN in Phase 1.

Running it needs a model call (paid API), which AGENTS.md rule 4 forbids without an owner-stated USD ceiling.
`ScriptedJudge` exists only to unit-test the parser and harness plumbing; it is not an evaluated auditor.
The judge sees a text summary of the report, never ground truth, flaw class, arm, or seed.
"""
import json
import re

import numpy as np

from .base import FLAG, NA, PASS, Auditor

PROMPT = """You are auditing an interpretability instrument before its result is accepted as evidence.
The instrument selected {k} units, compared their causal effect to {n} matched-random unit sets, and claims: {claim}.

Preregistered criterion: metric={p_metric}, floor={p_floor}, statistic=max over control draws.
Criterion the report says it applied: metric={a_metric}, floor={a_floor}.
Selected-set effect (singular class, plural class, mean): {sel}
Control maxima over {n} draws (singular, plural, mean): {cmax}
Reported p-value: {p}
Known-bad control (random units) effect: {kb}
Baseline metric value: {base} (metric ceiling: {ceil})
Items used for selection: {nfit}; items used for assessment: {nas}; items in both: {overlap}

Decide whether this instrument's conclusion should be rejected because the instrument itself is defective
(for example label-independent effect, selection leakage, a saturated metric, a changed criterion, missing
controls). Do not judge whether the claimed effect is interesting.
Answer with one JSON object: {{"verdict": "FLAG" or "PASS", "reason": "<one sentence>"}}"""


def build_prompt(case) -> str:
    r = case.report()
    f = lambda d: tuple(round(float(d[k]), 4) for k in ("sing", "plur", "all"))
    fit, asm = set(map(int, r.provenance["fit_ids"])), set(map(int, r.provenance["assess_ids"]))
    crit = r.provenance["criterion_applied"]
    return PROMPT.format(k=len(r.selected), n=r.n_draws, claim=r.claim, p_metric=case.prereg.metric,
                         p_floor=case.prereg.floor, a_metric=crit["metric"], a_floor=crit["floor"], sel=f(r.sel),
                         cmax=tuple(round(float(np.max(r.ctrl[k])), 4) for k in ("sing", "plur", "all")),
                         p=round(r.p_reported, 5), kb=f(r.known_bad_sel), base=round(r.baseline_metric, 4),
                         ceil=r.metric_ceiling, nfit=len(fit), nas=len(asm), overlap=len(fit & asm))


def parse_response(text: str) -> str:
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if m:
        try:
            v = str(json.loads(m.group(0)).get("verdict", "")).upper()
            if v in (FLAG, PASS):
                return v
        except json.JSONDecodeError:
            pass
    return NA


class LLMJudge(Auditor):
    """Wraps any `complete(prompt) -> str` callable. Unparseable output is NA (excluded, counted separately)."""
    name = "llm_judge"

    def __init__(self, complete):
        self.complete = complete

    def audit(self, case):
        if case.run is None:
            return NA
        return parse_response(self.complete(build_prompt(case)))


class ScriptedJudge:
    def __init__(self, reply='{"verdict": "PASS", "reason": "stub"}'):
        self.reply = reply

    def __call__(self, prompt):
        return self.reply
