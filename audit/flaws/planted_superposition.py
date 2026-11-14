"""Class 5: planted superposition in hand-built privileged-basis nets (R5b interface: W (d,n_feat), b).

Ground truth is by construction: a neuron is multi-job iff it reads >= 2 features. Good twin = exact
monosemantic net (scaled permutation, n = d). Bad variants: `pairs` (2 features/neuron, n = 2d) and `diffuse`
(8 features/neuron, damage share about 1/8 each, below R5b's 0.2 job cutoff).
The instrument's implicit claim is "each hidden unit has one job" (clean basis).
"""
import numpy as np
import torch

from ..core import Case, Prereg, world_rng

CLASS_ID, NAME = 5, "planted_superposition"
VARIANTS = ["pairs", "diffuse"]
D, S = 10, 0.5


def _build_W(variant: str, arm: str, rng: np.random.Generator):
    if arm == "good":
        n = D
        W = np.zeros((D, n))
        W[np.arange(D), rng.permutation(n)] = 1.0
        return W, n
    if variant == "pairs":
        n = 2 * D
        perm = rng.permutation(n)
        W = np.zeros((D, n))
        for j in range(D):
            W[j, perm[2 * j]] = rng.uniform(0.6, 0.9)
            W[j, perm[2 * j + 1]] = rng.uniform(0.6, 0.9)
        return W, n
    n = 20
    W = np.zeros((D, n))
    for j in range(D):
        idx = rng.choice(n, size=8, replace=False)
        W[j, idx] = rng.uniform(0.3, 0.45, size=8)
    return W, n


def _calibrate(W: np.ndarray, n_feat: int, rng: np.random.Generator):
    """Global scale alpha and bias b so the hand-built net reconstructs its input as well as it can (ablating a
    neuron then hurts reconstruction, as in a trained model). Structure (which feature each neuron reads) is untouched."""
    x = (rng.random((4000, n_feat)) > S) * rng.random((4000, n_feat))
    best = None
    for a in np.linspace(0.4, 3.0, 27):
        Wa = a * W
        y = np.maximum(np.maximum(x @ Wa.T, 0) @ Wa, 0)
        b = (x - y).mean(0)
        err = ((np.maximum(y + b, 0) - x) ** 2).mean()
        if best is None or err < best[0]:
            best = (err, a, b)
    return best[1] * W, best[2]


def build(arm: str, variant: str, seed: int) -> Case:
    vid = VARIANTS.index(variant)
    rng = world_rng(CLASS_ID, vid, seed)
    W, n_feat = _build_W(variant, arm, rng)
    W, b = _calibrate(W, n_feat, world_rng(CLASS_ID, vid, seed, tag=11))
    gt_multi = (W > 0).sum(1) >= 2
    toy = {"W": torch.tensor(W, dtype=torch.float32), "b": torch.tensor(b, dtype=torch.float32),
           "n_feat": n_feat, "S": S, "seed": seed}
    c = Case(case_id=f"{NAME}/{variant}/{arm}/{seed}", flaw_class=NAME, variant=variant, arm=arm, seed=seed,
             flawed=(arm == "bad"), prereg=Prereg(), run=None, supports=frozenset(), claim_correct=None, toy=toy)
    c.meta["gt_multi_fraction"] = float(gt_multi.mean())
    return c
