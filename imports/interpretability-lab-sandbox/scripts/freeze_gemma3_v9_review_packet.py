#!/usr/bin/env python3
"""Freeze a review packet for the Gemma 3 V9 design.

State slice: interpretability-lab-sandbox-v1.

The output requests an independent review. It contains no custody material,
private inputs, model bytes, or acceptance signature.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interp_lab.canonical import digest  # noqa: E402


STATE_SLICE = "interpretability-lab-sandbox-v1"


def freeze(design_path: Path, output_path: Path) -> dict[str, Any]:
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if not isinstance(design, dict) or design.get("state_slice") != STATE_SLICE:
        raise ValueError("Gemma V9 design identity mismatch")
    protocol_digest = digest(design)
    packet: dict[str, Any] = {
        "schema": "interp-lab.independent-review-packet.v1",
        "state_slice": STATE_SLICE,
        "review_status": "PENDING_INDEPENDENT_REVIEW",
        "protocol_id": "gemma3-reconstruction-v9",
        "protocol_digest": protocol_digest,
        "review_scope": [
            "model and feature identity",
            "SAE orientation, thresholds, and normalization",
            "layer provenance and dimensionality",
            "fresh corpus and split disjointness",
            "runtime, determinism, and H100 attestation requirements",
            "qualification gate order and reconstruction thresholds",
            "custody, retention, and claim ceilings",
            "downstream Weco, causal, scrubbing, and continual-learning locks",
        ],
        "required_return": {
            "decision": "ACCEPT or REJECT",
            "packet_digest_binding": True,
            "reviewer_controlled_ed25519_registry": True,
            "reviewer_identity": True,
            "conflict_of_interest_statement": True,
            "signed_receipt": True,
        },
        "acceptance": {
            "operator_may_self_sign": False,
            "local_signature_is_valid": False,
            "assessment_unlocks_only_after_signed_accept": True,
        },
        "claim_ceiling_before_accept": "LOCAL_DESIGN_AUDIT_ONLY",
        "nonclaims": ["no model execution authorization", "no independent proof", "no causal or continual-learning result"],
    }
    packet["packet_digest"] = digest(packet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(output_path, 0o644)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, default=ROOT / "configs/gemma3-27b-reconstruction-v9-design-only.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/review/gemma3-v9-review-packet.json")
    args = parser.parse_args()
    print(json.dumps(freeze(args.design, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
