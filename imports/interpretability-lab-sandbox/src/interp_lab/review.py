"""Verification of an external packet-bound Ed25519 review receipt.

State slice: interpretability-lab-sandbox-v1.

The verifier authenticates bytes; it cannot establish the reviewer's real-world
independence. That identity and conflict-of-interest assertion must arrive
through a separate trusted channel and remain outside this repository.
"""
from __future__ import annotations

import base64
from typing import Mapping, Any

from .canonical import canonical_bytes, digest


class ReviewVerificationError(ValueError):
    """Raised when an external review receipt is malformed or unauthenticated."""


STATE_SLICE = "interpretability-lab-sandbox-v1"
REVIEW_SCHEMA = "interp-lab.external-review-receipt.v1"
REGISTRY_SCHEMA = "interp-lab.reviewer-registry-entry.v1"


def _b64(value: Any, field: str, *, size: int) -> bytes:
    if not isinstance(value, str) or not value:
        raise ReviewVerificationError(f"{field} is missing")
    try:
        decoded = base64.b64decode(value, validate=True)
    except Exception as error:
        raise ReviewVerificationError(f"{field} is not strict base64") from error
    if len(decoded) != size:
        raise ReviewVerificationError(f"{field} has the wrong length")
    return decoded


def verify_external_accept(receipt: Mapping[str, Any], expected_packet_digest: str) -> dict[str, Any]:
    """Verify an exact external ACCEPT receipt and return its bound identity."""
    if not isinstance(expected_packet_digest, str) or not expected_packet_digest:
        raise ReviewVerificationError("expected packet digest is missing")
    if receipt.get("schema") != REVIEW_SCHEMA:
        raise ReviewVerificationError("review receipt schema mismatch")
    if receipt.get("state_slice") != STATE_SLICE:
        raise ReviewVerificationError("review receipt state slice mismatch")
    if receipt.get("source") != "external" or receipt.get("decision") != "ACCEPT":
        raise ReviewVerificationError("review receipt is not an external ACCEPT")
    if receipt.get("algorithm") != "Ed25519":
        raise ReviewVerificationError("review receipt algorithm is not Ed25519")
    reviewer_id = receipt.get("reviewer_id")
    public_key = receipt.get("public_key")
    coi = receipt.get("conflict_of_interest_statement")
    if not isinstance(reviewer_id, str) or not reviewer_id:
        raise ReviewVerificationError("reviewer identity is missing")
    if not isinstance(coi, str) or not coi:
        raise ReviewVerificationError("reviewer conflict-of-interest statement is missing")
    registry = receipt.get("registry_entry")
    if not isinstance(registry, Mapping):
        raise ReviewVerificationError("reviewer registry entry is missing")
    expected_registry = {
        "schema": REGISTRY_SCHEMA,
        "state_slice": STATE_SLICE,
        "reviewer_id": reviewer_id,
        "public_key": public_key,
        "conflict_of_interest_statement": coi,
    }
    if dict(registry) != expected_registry:
        raise ReviewVerificationError("reviewer registry entry is not bound to receipt identity")
    expected_payload = {
        "schema": REVIEW_SCHEMA,
        "state_slice": STATE_SLICE,
        "decision": "ACCEPT",
        "packet_digest": expected_packet_digest,
        "reviewer_id": reviewer_id,
        "public_key": public_key,
        "conflict_of_interest_statement": coi,
        "registry_entry_digest": digest(expected_registry),
    }
    if receipt.get("packet_digest") != expected_packet_digest:
        raise ReviewVerificationError("review receipt packet digest mismatch")
    if receipt.get("signed_payload") != expected_payload:
        raise ReviewVerificationError("signed payload does not bind receipt fields")
    key_bytes = _b64(public_key, "public_key", size=32)
    signature = _b64(receipt.get("signature"), "signature", size=64)
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey  # type: ignore
    except ImportError as error:  # pragma: no cover - dependency-only failure
        raise ReviewVerificationError("cryptographic Ed25519 verifier is unavailable") from error
    try:
        Ed25519PublicKey.from_public_bytes(key_bytes).verify(signature, canonical_bytes(expected_payload))
    except Exception as error:
        raise ReviewVerificationError("Ed25519 signature verification failed") from error
    return {
        "reviewer_id": reviewer_id,
        "public_key": public_key,
        "registry_entry_digest": expected_payload["registry_entry_digest"],
        "packet_digest": expected_packet_digest,
    }
