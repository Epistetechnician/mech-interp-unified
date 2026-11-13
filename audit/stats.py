"""Rates with bootstrap and Wilson 95% intervals, and the preregistered labels."""
import math

import numpy as np

B = 10_000
BOOT_SEED = 20261113


def wilson(k: int, n: int, z: float = 1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def boot_ci(strata: list, tag: int = 0):
    """strata: list of (k, n) per variant. Percentile bootstrap resampling cases within variant."""
    n_tot = sum(n for _, n in strata)
    if n_tot == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(np.random.SeedSequence([BOOT_SEED, tag]))
    tot = np.zeros(B)
    for k, n in strata:
        if n:
            tot += rng.binomial(n, k / n, size=B)  # resampling n binary cases = Binomial(n, k/n)
    r = tot / n_tot
    return (float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975)))


def rate(strata: list, tag: int = 0) -> dict:
    k = sum(k for k, _ in strata)
    n = sum(n for _, n in strata)
    return {"k": k, "n": n, "rate": (k / n if n else float("nan")), "boot_ci": boot_ci(strata, tag),
            "wilson_ci": wilson(k, n)}


def label(tpr: dict, fpr: dict) -> str:
    if tpr["n"] == 0:
        return "N/A"
    if tpr["rate"] < 0.30:
        lab = "BLIND"
    elif tpr["boot_ci"][0] >= 0.80 and fpr["boot_ci"][1] <= 0.10:
        lab = "DETECTS"
    else:
        lab = "PARTIAL"
    if lab != "BLIND" and fpr["n"] and fpr["boot_ci"][1] > 0.10:
        lab += "+NOISY"
    return lab
