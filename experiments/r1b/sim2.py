"""Simulator variants for R1b-v2 development. mode: digit | digit_prev | binary_prev."""
import numpy as np

import lib

FEW = {
    "digit": [("the token \"the\"", [("the", 9), ("cat", 0), ("sat", 0), ("on", 0), ("the", 9), ("mat", 0)]),
              ("a number or digit", [("In", 0), ("1998", 9), ("there", 0), ("were", 0), ("42", 9), ("words", 0)])],
    "pairs": [("the token \"the\"", [("the", 9), ("cat", 0), ("sat", 0), ("on", 0), ("the", 9), ("mat", 0)]),
              ("the token that comes immediately after the token \"of\"",
               [("The", 0), ("end", 0), ("of", 0), ("the", 9), ("war", 0), ("of", 0), ("1812", 9), ("began", 0)])],
}
HEAD = ("We're studying neurons in a neural network. Each neuron fires on one particular pattern in a text. Given an "
        "explanation of the neuron, predict its activation on every token: 0 = does not fire, 9 = fires strongly. "
        "Read the explanation literally and exactly; most tokens do not fire.\n\n")


class Sim2:
    def __init__(self, m, mode):
        import mlx.core as mx
        self.mx, self.m, self.mode = mx, m, mode
        enc = lambda s: m.tok.encode(s, add_special_tokens=False)
        self.enc = enc
        self.d = [enc(str(i))[-1] for i in range(10)]
        self.tab = enc("\t")[-1]
        self.unk = enc("unknown\n")
        self.prev = mode != "digit"
        self.binary = mode == "binary_prev"
        ex = FEW["pairs" if self.prev else "digit"]
        txt = HEAD
        for job, items in ex:
            txt += f"Neuron explanation: {job}\nActivations:\n<start>\n"
            for i, (t, v) in enumerate(items):
                v = (1 if v else 0) if self.binary else v
                txt += (f"{items[i-1][0] if i else '<s>'} | {t}\t{v}\n" if self.prev else f"{t}\t{v}\n")
            txt += "<end>\n\n"
        self.header = enc(txt)
        self.tokens_seen = 0

    def score(self, job, tokens):
        mx, enc = self.mx, self.enc
        ids = list(self.header) + enc(f"Neuron explanation: {job}\nActivations:\n<start>\n")
        pos = []
        for i, t in enumerate(tokens):
            line = f"{lib.clean_tok(tokens[i-1]) if i else '<s>'} | {lib.clean_tok(t)}" if self.prev else lib.clean_tok(t)
            ids += enc(line) + [self.tab]
            pos.append(len(ids) - 1)
            ids += self.unk
        self.tokens_seen += len(ids)
        logits = self.m.model(mx.array([ids]))[0]
        if self.binary:
            sel = logits[mx.array(pos)][:, mx.array([self.d[0], self.d[1]])].astype(mx.float32)
            ev = mx.softmax(sel, axis=-1)[:, 1] * 9.0
        else:
            sel = logits[mx.array(pos)][:, mx.array(self.d)].astype(mx.float32)
            ev = (mx.softmax(sel, axis=-1) * mx.arange(10)).sum(-1)
        mx.eval(ev)
        return np.array(ev)
