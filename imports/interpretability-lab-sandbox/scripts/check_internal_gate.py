#!/usr/bin/env python3
"""Validate an internal exploratory gate without loading a model.

State slice: interpretability-lab-sandbox-v1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interp_lab.execution import require_execution


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--runtime-attestation", type=Path, required=True)
    parser.add_argument("--custody-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    runtime_attestation = json.loads(args.runtime_attestation.read_text(encoding="utf-8"))
    require_execution(
        config=config,
        packet=packet,
        packet_digest=packet.get("packet_digest"),
        custody_root=args.custody_root,
        authorization=authorization,
        runtime_attestation=runtime_attestation,
    )
    print(json.dumps({"status": "INTERNAL_AUTHORIZED", "claim_ceiling": authorization["claim_ceiling"], "packet_digest": packet["packet_digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
