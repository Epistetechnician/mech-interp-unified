"""MLX labeller/simulator for R1b. Imported only when a model is used (tests never import this)."""
import numpy as np

import lib

SIM_HEADER = (
    "We're studying neurons in a neural network. Each neuron looks for one particular thing in a text. "
    "Given an explanation of what the neuron does, predict its activation on every token on a scale 0-9 "
    "(0 = no match, 9 = strong match).\n\n"
    "Neuron explanation: the token \"the\"\nActivations:\n<start>\nthe\t9\ncat\t0\nsat\t0\non\t0\nthe\t9\nmat\t0\n<end>\n\n"
    "Neuron explanation: a number or digit\nActivations:\n<start>\nIn\t0\n1998\t9\nthere\t0\nwere\t0\n42\t9\nwords\t0\n<end>\n\n")


class MLXModel:
    def __init__(self, path):
        from mlx_lm import load
        self.model, self.tok = load(path)


class MLXLabeller:
    def __init__(self, m, max_tokens=300):
        self.m, self.max_tokens = m, max_tokens

    def explain(self, prompt):
        from mlx_lm import generate
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = self.m.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            text = self.m.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        out = generate(self.m.model, self.m.tok, prompt=text, max_tokens=self.max_tokens, verbose=False)
        return lib.parse_jobs(out)


class MLXSimulator:
    """Expected digit under the model's next-token distribution at each token's tab position (one pass)."""

    def __init__(self, m):
        import mlx.core as mx
        self.mx, self.m = mx, m
        enc = m.tok.encode
        self.digit_ids = [enc(str(d), add_special_tokens=False)[-1] for d in range(10)]
        self.tab = enc("\t", add_special_tokens=False)[-1]
        self.unk = enc("unknown\n", add_special_tokens=False)
        self.header = enc(SIM_HEADER, add_special_tokens=False)
        self.tokens_seen = 0

    def build(self, job, tokens):
        enc = self.m.tok.encode
        ids = list(self.header) + enc(f"Neuron explanation: {job}\nActivations:\n<start>\n", add_special_tokens=False)
        pos = []
        for t in tokens:
            ids += enc(lib.clean_tok(t), add_special_tokens=False) + [self.tab]
            pos.append(len(ids) - 1)
            ids += self.unk
        return ids, pos

    def score(self, job, tokens):
        mx = self.mx
        ids, pos = self.build(job, tokens)
        self.tokens_seen += len(ids)
        logits = self.m.model(mx.array([ids]))[0]
        sel = logits[mx.array(pos)][:, mx.array(self.digit_ids)].astype(mx.float32)
        p = mx.softmax(sel, axis=-1)
        ev = (p * mx.arange(10)).sum(-1)
        mx.eval(ev)
        return np.array(ev)
