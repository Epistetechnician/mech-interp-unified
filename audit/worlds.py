"""Calibrated one-hidden-layer ReLU nets with known per-unit effects.

Net: h = relu(W1 x), margin_i = s_i * sum_u d_u h_iu (s_i = +1 for class 0 'sing', -1 for class 1 'plur').
Ablating a unit set U removes sum_{u in U} C_iu from the margin, C_iu = s_i d_u h_iu, so effects are additive
and ground truth is analytic. Roles: P carriers serve class 0, N carriers serve class 1, decoys are large but
label-orthogonal, inert units are small.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

D_IN = 16
STRONG = (0.9, 0.77, 0.65, 0.55, 0.47, 0.40)


@dataclass(frozen=True)
class WorldCfg:
    n_P: int = 6
    n_N: int = 6
    n_decoy: int = 0
    n_inert: int = 20
    profile: tuple = STRONG
    sigma: float = 0.5              # orthogonal noise on carrier directions (0 => rank-1, accuracy saturates)
    n_fit: int = 1000
    n_assess: int = 1000
    decoy_d: float = 3.0
    inert_d: float = 0.02
    carrier_norm: float = 1.0
    inert_norm: float = 0.3
    tie: str | None = None          # None | 'near' | 'distinct'
    tie_eps: float = 1e-4
    tie_gap: float = 0.06
    null: bool = False              # no label-aligned unit at all
    null_d: tuple = (0.8, 1.5)
    ablate: str = 'mean'            # 'mean' (replace h by its reference mean) | 'zero'


class NetWorld:
    def __init__(self, cfg: WorldCfg, rng: np.random.Generator):
        self.cfg = cfg
        self._rng = rng
        self.H = cfg.n_P + cfg.n_N + cfg.n_decoy + cfg.n_inert
        a = rng.normal(size=D_IN)
        self.a = a / np.linalg.norm(a)
        self._fresh_ss = np.random.SeedSequence(rng.integers(0, 2**31, size=4).tolist())
        self.fit = self._items(cfg.n_fit, rng)
        self.assess = self._items(cfg.n_assess, rng)
        self._build(rng)

    # ---- items -------------------------------------------------------------------------------------------------
    def _items(self, n: int, rng: np.random.Generator):
        """Exactly n/2 items per class: reflect across the teacher hyperplane when class counts must be fixed."""
        X = rng.normal(size=(n, D_IN))
        proj = X @ self.a
        want = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        flip = np.sign(proj) != want
        X[flip] = X[flip] - 2 * proj[flip, None] * self.a[None, :]
        order = rng.permutation(n)
        X, want = X[order], want[order]
        return {"X": X, "s": want, "cls": (want < 0).astype(int)}

    def fresh_items(self, n: int, tag: int):
        r = np.random.default_rng(np.random.SeedSequence(list(self._fresh_ss.generate_state(2)) + [tag]))
        return self._items(n, r)

    # ---- net ---------------------------------------------------------------------------------------------------
    def _orth(self, rng):
        v = rng.normal(size=D_IN)
        v -= (v @ self.a) * self.a
        return v / np.linalg.norm(v)

    def _build(self, rng):
        c = self.cfg
        roles = (["P"] * c.n_P) + (["N"] * c.n_N) + (["decoy"] * c.n_decoy) + (["inert"] * c.n_inert)
        roles = [roles[i] for i in rng.permutation(len(roles))]
        self.roles = np.array(roles)
        H = len(roles)
        W1 = np.zeros((H, D_IN))
        d = np.zeros(H)
        target = np.zeros(H)
        if c.null:
            for u in range(H):
                W1[u] = self._orth(rng)
                d[u] = rng.choice([-1.0, 1.0]) * rng.uniform(*c.null_d)
            self.W1, self.d, self.target = W1, d, target
            self.mu = self._ref_mu(rng)
            return
        shared = {"P": None, "N": None}
        if c.tie:
            for role, sgn in (("P", 1.0), ("N", -1.0)):
                shared[role] = (sgn * self.a + c.sigma * self._orth(rng)) * c.carrier_norm / np.sqrt(1 + c.sigma**2)
        rank = {"P": 0, "N": 0}
        for u, role in enumerate(roles):
            if role in ("P", "N"):
                sgn = 1.0 if role == "P" else -1.0
                if c.tie:
                    W1[u] = shared[role]
                else:
                    W1[u] = (sgn * self.a + c.sigma * self._orth(rng)) * c.carrier_norm / np.sqrt(1 + c.sigma**2)
                target[u] = c.profile[rank[role]] if not c.tie else 0.6
                rank[role] += 1
            elif role == "decoy":
                W1[u] = self._orth(rng)
                d[u] = c.decoy_d * rng.choice([-1.0, 1.0])
            else:
                W1[u] = self._orth(rng) * c.inert_norm
                d[u] = c.inert_d * rng.choice([-1.0, 1.0])
        # calibrate carrier d so that the class-own mean effect equals target
        cal = self.fit if c.tie else self._items(20000, rng)
        Hc = np.maximum(cal["X"] @ W1.T, 0.0)
        for u, role in enumerate(roles):
            if role in ("P", "N"):
                own = cal["cls"] == (0 if role == "P" else 1)
                g = Hc[own, u].mean() - (Hc[:, u].mean() if c.ablate == "mean" else 0.0)
                d[u] = (1.0 if role == "P" else -1.0) * target[u] / g
        if c.tie:
            idx = np.array([u for u, r in enumerate(roles) if r in ("P", "N")])
            j = np.arange(len(idx))
            if c.tie == "near":
                d[idx] *= 1.0 + c.tie_eps * rng.uniform(-1, 1, size=len(idx))
            else:
                d[idx] *= (1.0 + c.tie_gap * rng.permutation(len(idx)))
        self.W1, self.d, self.target = W1, d, target
        self.mu = self._ref_mu(rng)

    def _ref_mu(self, rng):
        ref = self._items(20000, np.random.default_rng(np.random.SeedSequence(self._fresh_ss.generate_state(2).tolist() + [99])))
        return np.maximum(ref["X"] @ self.W1.T, 0.0).mean(0)

    # ---- views -------------------------------------------------------------------------------------------------
    def transformed(self, perm: np.ndarray, scale: np.ndarray):
        """Function-preserving: new unit j = old unit perm[j] with ReLU rescale (W1 row * scale, d / scale)."""
        w = object.__new__(NetWorld)
        w.__dict__.update(self.__dict__)
        w.W1 = scale[:, None] * self.W1[perm]
        w.d = self.d[perm] / scale
        w.roles = self.roles[perm]
        w.target = self.target[perm]
        w.mu = scale * self.mu[perm]
        return w

    def cast(self, dt):
        """Emulate running the net at precision dt: parameters and inputs rounded through dt."""
        w = object.__new__(NetWorld)
        w.__dict__.update(self.__dict__)
        r = lambda a: np.asarray(a).astype(dt).astype(np.float64)
        w.W1, w.d, w.mu = r(self.W1), r(self.d), r(self.mu)
        w._cast_items = r
        return w

    def _h(self, items):
        if hasattr(self, "_cast_items"):
            items = {**items, "X": self._cast_items(items["X"])}
        return np.maximum(items["X"] @ self.W1.T, 0.0)

    def contrib(self, items: dict, s: np.ndarray | None = None) -> np.ndarray:
        """Margin removed by ablating each unit (mean- or zero-ablation)."""
        H = self._h(items)
        if self.cfg.ablate == "mean":
            H = H - self.mu[None, :]
        s = items["s"] if s is None else s
        return s[:, None] * self.d[None, :] * H

    def margin(self, items: dict, s: np.ndarray | None = None) -> np.ndarray:
        s = items["s"] if s is None else s
        return s * (self._h(items) @ self.d)


def make_perm_scale(H: int, rng: np.random.Generator):
    perm = rng.permutation(H)
    scale = np.exp(rng.uniform(np.log(0.25), np.log(4.0), size=H))
    return perm, scale
