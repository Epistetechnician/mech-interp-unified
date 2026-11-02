"""External review receipt verifier tests.

State slice: interpretability-lab-sandbox-v1.

The key is generated in memory and is never an execution authorization.
"""
from __future__ import annotations

import base64
import unittest

from interp_lab.canonical import canonical_bytes, digest
from interp_lab.review import ReviewVerificationError, verify_external_accept


class ReviewTests(unittest.TestCase):
    def _receipt(self) -> tuple[dict[str, object], str]:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        packet_digest = "a" * 64
        private = Ed25519PrivateKey.generate()
        public_key = base64.b64encode(private.public_key().public_bytes_raw()).decode("ascii")
        registry = {
            "schema": "interp-lab.reviewer-registry-entry.v1",
            "state_slice": "interpretability-lab-sandbox-v1",
            "reviewer_id": "reviewer.example/independent-1",
            "public_key": public_key,
            "conflict_of_interest_statement": "No financial or supervisory relationship.",
        }
        payload = {
            "schema": "interp-lab.external-review-receipt.v1",
            "state_slice": "interpretability-lab-sandbox-v1",
            "decision": "ACCEPT",
            "packet_digest": packet_digest,
            "reviewer_id": registry["reviewer_id"],
            "public_key": public_key,
            "conflict_of_interest_statement": registry["conflict_of_interest_statement"],
            "registry_entry_digest": digest(registry),
        }
        receipt = {
            **payload,
            "source": "external",
            "algorithm": "Ed25519",
            "registry_entry": registry,
            "signed_payload": payload,
            "signature": base64.b64encode(private.sign(canonical_bytes(payload))).decode("ascii"),
        }
        return receipt, packet_digest

    def test_valid_receipt_verifies(self):
        receipt, packet_digest = self._receipt()
        result = verify_external_accept(receipt, packet_digest)
        self.assertEqual(result["reviewer_id"], "reviewer.example/independent-1")

    def test_signature_tampering_is_rejected(self):
        receipt, packet_digest = self._receipt()
        receipt["signature"] = base64.b64encode(b"0" * 64).decode("ascii")
        with self.assertRaisesRegex(ReviewVerificationError, "signature verification failed"):
            verify_external_accept(receipt, packet_digest)

    def test_registry_swap_is_rejected(self):
        receipt, packet_digest = self._receipt()
        receipt["registry_entry"] = {**receipt["registry_entry"], "reviewer_id": "operator"}  # type: ignore[index]
        with self.assertRaisesRegex(ReviewVerificationError, "not bound"):
            verify_external_accept(receipt, packet_digest)
