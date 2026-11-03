"""Deterministic fake labeller/simulator for hermetic tests (no model)."""
import re

import numpy as np

_MARK = re.compile(r"«([^|»]+)\|(\d+)»")


class FakeLabeller:
    """Reads marked tokens from the evidence text; jobs are 'tokens: a, b, c, d' by marked frequency."""

    def __init__(self, per_job=4):
        self.per_job = per_job
        self.calls = []

    def explain(self, prompt):
        self.calls.append(prompt)
        cnt = {}
        for tok, _ in _MARK.findall(prompt):
            cnt[tok] = cnt.get(tok, 0) + 1
        order = [t for t, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))]
        jobs = []
        for i in range(0, min(len(order), 3 * self.per_job), self.per_job):
            jobs.append("tokens: " + ", ".join(order[i:i + self.per_job]))
        return jobs


class RandomLabeller:
    """Known-bad labeller: ignores evidence, emits random tokens from a fixed vocabulary."""

    def __init__(self, vocab, seed=0):
        self.vocab, self.rng = list(vocab), np.random.default_rng(seed)

    def explain(self, prompt):
        return ["tokens: " + ", ".join(self.rng.choice(self.vocab, 4, replace=False).tolist()) for _ in range(2)]


class FakeSimulator:
    """Scores 9 for tokens listed in a 'tokens: ...' job, else 0. `noise` adds deterministic jitter."""

    def __init__(self, noise=0.0, seed=0):
        self.noise, self.seed = noise, seed

    def score(self, job, tokens):
        listed = {t.strip() for t in job.split(":", 1)[-1].split(",")}
        s = np.array([9.0 if t in listed else 0.0 for t in tokens])
        if self.noise:
            rng = np.random.default_rng(abs(hash((job, tuple(tokens)))) % 2**32 ^ self.seed)
            s = np.clip(s + rng.normal(0, self.noise, s.size), 0, 9)
        return s
