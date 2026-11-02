"""Canonical identity and hash-chain helpers.

State slice: interpretability-lab-sandbox-v1.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def chain_digest(previous: str, payload: Mapping[str, Any]) -> str:
    return digest({"previous": previous, "payload": payload})
