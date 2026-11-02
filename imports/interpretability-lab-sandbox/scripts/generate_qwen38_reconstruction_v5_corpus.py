#!/usr/bin/env python3
"""Generate the fresh deterministic V5 reconstruction corpus.

State slice: interpretability-lab-sandbox-v1.

The corpus contains no model outputs or activations.  It is a fixed roster of
document-disjoint prompt families and is intended to be generated inside the
owner-only custody root before packet compilation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


STATE_SLICE = "interpretability-lab-sandbox-v1"
SCHEMA = "interp-lab.assessment-corpus.v1"
SPLITS = ("fit", "tune", "assessment")
COUNT = 48


def _row(split: str, index: int) -> dict[str, object]:
    row_id = f"{split}-{index:03d}"
    document_id = f"{split}-document-{index:03d}"
    family = f"{split}-family-{index:03d}"
    text = (
        f"V5 fresh family {index:03d} in the {split} split. "
        "Explain one precise engineering tradeoff in a small typed program, "
        "then give a concise invariant and a counterexample."
    )
    donor = f"{split}-{(index - 1) % COUNT:03d}"
    shuffled = f"{split}-{(index + 17) % COUNT:03d}"
    return {
        "row_id": row_id,
        "split": split,
        "family": family,
        "document_id": document_id,
        "text": text,
        "target_token_id": 1,
        "contrast_token_id": 2,
        "donor_row_id": donor,
        "shuffled_donor_row_id": shuffled,
    }


def generate(output: Path) -> dict[str, object]:
    rows = [_row(split, index) for split in SPLITS for index in range(COUNT)]
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "state_slice": STATE_SLICE,
        "fresh": True,
        "generator": "deterministic-v5-prompt-roster",
        "seed": 20260916,
        "splits": {split: COUNT for split in SPLITS},
        "document_disjoint": True,
        "family_disjoint": True,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(output, 0o600)
    return {
        "path": str(output),
        "rows": len(rows),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
