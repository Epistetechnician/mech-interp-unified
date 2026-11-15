"""Shared instrument pipeline: select top-k units on fit items, assess against matched-random unit sets.

A Spec selects which stage is defective. `run_pipeline` returns a Report; probes re-execute with overrides.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .core import N_DRAWS, Prereg, Probe, Report

K = 8
DT = {"float16": np.float16, "float32": np.float32, "float64": np.float64}


@dataclass(frozen=True)
class Spec:
    prereg: Prereg
    k: int = K
    n_draws: int = N_DRAWS
    metric: str = "margin"             # margin | accuracy | labelfree | topdecile
    select_score: str = "ablation"     # ablation | labelfree | weightnorm
    assess: str = "heldout"            # heldout | fit
    leak_log: str = "honest"           # honest | hidden
    trim: float = 0.0
    applied_floor: Optional[float] = None
    log_metric: Optional[str] = None   # what provenance claims the metric was (None => truthful)
    control_seed: int = 0
    shuffle_items: bool = False        # semantic no-op perturbation for good twins
    n_fit_use: Optional[int] = None    # use only the first n items of the fit split for selection
    # --- Phase 2 (audit/PROTOCOL_PHASE2.md); defaults leave Phase-1 cases unchanged ---
    ctrl_k: Optional[int] = None       # class 9 `size`: control sets of this size instead of k
    ctrl_pool: str = "all"             # class 9 `pool`: 'lowscore' draws controls from the low-fit-score half
    ctrl_cond: str = "same"            # class 9 `condition`: 'shuffled' computes control effects under a shuffled readout
    abl_eff: float = 1.0               # class 10: fraction of each unit's contribution the intervention removes
    best_of_m: int = 0                 # class 11: search up to m item subsets for a Supported verdict
    subset_n: int = 64


def _effect(metric: str, m: np.ndarray, A: np.ndarray, cls: np.ndarray) -> dict:
    """Per-class effects. A: (n,) or (n, draws)."""
    out = {}
    for name, c in (("sing", 0), ("plur", 1)):
        sel = cls == c
        a = A[sel]
        if metric == "margin":
            e = a.mean(0)
        elif metric == "labelfree":
            e = np.abs(a).mean(0)
        elif metric == "accuracy":
            mm = m[sel]
            mm = mm[:, None] if a.ndim == 2 else mm
            e = (mm > 0).mean(0) - ((mm - a) > 0).mean(0)
        elif metric == "topdecile":
            q = max(1, int(round(0.1 * a.shape[0])))
            e = np.sort(a, axis=0)[-q:].mean(0)
        else:
            raise ValueError(metric)
        out[name] = e
    out["all"] = 0.5 * (out["sing"] + out["plur"])
    return out


def _masks(H: int, sel: np.ndarray, k: int, n: int, rng, pool=None) -> np.ndarray:
    if pool is None:
        pool = np.setdiff1d(np.arange(H), sel)
    M = np.zeros((H, n))
    for j in range(n):
        M[rng.choice(pool, size=k, replace=False), j] = 1.0
    return M


def verdict(sel: dict, ctrl: dict, floor: float) -> str:
    ok = sel["sing"] > np.max(ctrl["sing"]) and sel["plur"] > np.max(ctrl["plur"]) and sel["all"] >= floor
    return "Supported" if ok else "NoCandidate"


def run_pipeline(world, spec: Spec, probe: Optional[Probe] = None, perturb_rng=None) -> Report:
    probe = probe or Probe()
    cfg = world.cfg
    net = world
    if probe.transform is not None:
        net = world.transformed(*probe.transform)
    dt = DT[probe.dtype or "float32"]
    if dt is not np.float64:
        net = net.cast(dt)
    fit = world.fit
    if spec.n_fit_use:
        fit = {k_: v[: spec.n_fit_use] for k_, v in fit.items()}
    n_fit = len(fit["s"])
    if probe.fresh_assess:
        asm = world.fresh_items(cfg.n_assess, tag=17)
        assess_ids = 10**6 + np.arange(len(asm["s"]))
    elif spec.assess == "fit":
        asm = fit
        assess_ids = np.arange(n_fit) if spec.leak_log == "honest" else 10**5 + np.arange(n_fit)
    else:
        asm = world.assess
        assess_ids = 10**5 + np.arange(len(asm["s"]))
    if spec.shuffle_items and perturb_rng is not None:
        p = perturb_rng.permutation(len(asm["s"]))
        asm = {k_: v[p] for k_, v in asm.items()}

    metric = spec.prereg.metric if probe.use_prereg else spec.metric
    floor = spec.prereg.floor if probe.use_prereg else (spec.applied_floor or spec.prereg.floor)

    # --- selection on fit items (always true labels) ---
    Cf = net.contrib(fit)
    if spec.select_score == "weightnorm":
        score = np.linalg.norm(net.W1, axis=1)
        selected = np.argsort(-score, kind="stable")[: spec.k]
    else:
        # quota selection: k/2 units by class-0 score, then k/2 more by class-1 score (excluding chosen)
        Cs = np.abs(Cf) if spec.select_score == "labelfree" else Cf
        if spec.select_score not in ("ablation", "labelfree"):
            raise ValueError(spec.select_score)
        chosen = []
        for c in (0, 1):
            sc = np.mean(Cs[fit["cls"] == c], axis=0)
            sc[chosen] = -np.inf
            chosen += list(np.argsort(-sc, kind="stable")[: spec.k // 2])
        selected = np.array(chosen)

    # --- assessment ---
    cseed = spec.control_seed if probe.control_seed is None else probe.control_seed
    H = net.W1.shape[0]
    defect = not probe.plain_controls      # the auditor's plain redraw never reproduces a control-side defect

    def assess(items):
        s_eff = items["s"]
        if probe.shuffle_readout:
            s_eff = np.random.default_rng(991).permutation(items["s"])
        Ca = net.contrib(items, s_eff)
        m = net.margin(items, s_eff)
        if spec.abl_eff != 1.0:
            Ca = spec.abl_eff * Ca
        cls = items["cls"]
        sel = {k_: float(v) for k_, v in _effect(metric, m, Ca[:, selected].sum(1), cls).items()}
        rng = np.random.default_rng(np.random.SeedSequence([cseed, 31337]))
        # author-supplied known-bad control: a random size-k set drawn from the non-selected pool (same law as a control draw)
        kb_set = rng.choice(np.setdiff1d(np.arange(H), selected), size=spec.k, replace=False)
        kb = {k_: float(v) for k_, v in _effect(metric, m, Ca[:, kb_set].sum(1), cls).items()}
        ctrl, p_rep, claim = None, float("nan"), "NoCandidate"
        if probe.need_controls:
            k_c, pool, Cc, mc = spec.k, None, Ca, m
            if defect and spec.ctrl_k:
                k_c = spec.ctrl_k
            if defect and spec.ctrl_pool == "lowscore":
                rest = np.setdiff1d(np.arange(H), selected)
                pool = rest[np.argsort(np.abs(Cf[:, rest]).mean(0), kind="stable")[: len(rest) // 2]]
            if defect and spec.ctrl_cond == "shuffled":
                s_sh = np.random.default_rng(np.random.SeedSequence([cseed, 4242])).permutation(items["s"])
                Cc, mc = spec.abl_eff * net.contrib(items, s_sh), net.margin(items, s_sh)
            M = _masks(H, selected, k_c, spec.n_draws, rng, pool)
            ctrl = {k_: np.asarray(v) for k_, v in _effect(metric, mc, Cc @ M, cls).items()}
            if spec.trim > 0 and defect:
                n_drop = int(round(spec.trim * spec.n_draws))
                order = np.argsort(-ctrl["all"], kind="stable")
                drop, keep = order[:n_drop], order[n_drop:]
                src = rng.choice(keep, size=n_drop, replace=True)
                for k_ in ctrl:
                    ctrl[k_] = ctrl[k_].copy()
                    ctrl[k_][drop] = ctrl[k_][src]
            p_rep = (1 + int((ctrl["all"] >= sel["all"]).sum())) / (1 + len(ctrl["all"]))
            claim = verdict(sel, ctrl, floor)
        return sel, ctrl, kb, p_rep, claim, m

    if spec.best_of_m and probe.need_controls:
        rs = np.random.default_rng(np.random.SeedSequence([cseed, 777]))
        by_cls = [np.flatnonzero(asm["cls"] == c) for c in (0, 1)]
        for _ in range(spec.best_of_m):
            idx = np.concatenate([rs.choice(ix, size=spec.subset_n // 2, replace=False) for ix in by_cls])
            sub = {k_: v[idx] for k_, v in asm.items()}
            if assess(sub)[4] == "Supported":
                asm, assess_ids = sub, np.asarray(assess_ids)[idx]
                break
    sel, ctrl, kb, p_rep, claim, m = assess(asm)
    base = float((m > 0).mean())
    prov = {
        "fit_ids": np.arange(n_fit),
        "assess_ids": assess_ids,
        "criterion_applied": {"metric": spec.log_metric or metric, "floor": floor, "stat": "max"},
    }
    return Report(claim=claim, selected=selected, sel=sel, ctrl=ctrl, n_draws=spec.n_draws,
                  p_reported=p_rep, floor_applied=floor, known_bad_sel=kb, provenance=prov,
                  baseline_metric=base, metric_ceiling=1.0 if metric == "accuracy" else None)
