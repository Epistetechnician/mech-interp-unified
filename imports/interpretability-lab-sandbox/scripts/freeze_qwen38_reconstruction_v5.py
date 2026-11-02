#!/usr/bin/env python3
"""Compile a custody-bound Qwen3.8 V5 packet without executing a model.

State slice: interpretability-lab-sandbox-v1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interp_lab.reconstruction_packet import freeze_reconstruction_packet  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--custody-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--sae-weights", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-identity", required=True)
    parser.add_argument("--transformers-version", required=True)
    parser.add_argument("--transformers-source-revision", required=True)
    parser.add_argument("--packet-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze_reconstruction_packet(
        design_path=args.design,
        custody_root=args.custody_root,
        model_root=args.model_root,
        sae_weights=args.sae_weights,
        corpus_path=args.corpus,
        corpus_identity=args.corpus_identity,
        transformers_version=args.transformers_version,
        transformers_source_revision=args.transformers_source_revision,
        packet_id=args.packet_id,
        output_path=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
