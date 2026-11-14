"""Frozen holdout split by hash of item id. The rule is fixed here and never re-salted.

holdout iff int(sha256(SALT + ":" + id).hexdigest()[:8], 16) % MOD == 0   (about 1 in 4)
"""
from __future__ import annotations

import hashlib

SALT = "f1-transfer-bench-v1-split"
MOD = 4


def bucket(item_id: str) -> int:
    return int(hashlib.sha256(f"{SALT}:{item_id}".encode()).hexdigest()[:8], 16) % MOD


def assign(item_id: str) -> str:
    return "holdout" if bucket(item_id) == 0 else "train"


def digest(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
