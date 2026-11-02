#!/usr/bin/env python3
"""Generate the fresh deterministic Gemma 3 V9 corpus.

State slice: interpretability-lab-sandbox-v1.

The generator writes prompt metadata only. It never calls a model and keeps
the corpus outside Git when used by a real run.
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
    document_id = f"gemma-v9-{split}-document-{index:03d}"
    family = f"gemma-v9-{split}-family-{index:03d}"
    text = (
        f"Gemma V9 family {index:03d} in the {split} split. "
        "State one concrete consequence of a model update, identify the "
        "behavior that must be retained, and give a falsifying example."
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
        "generator": "deterministic-gemma3-v9-prompt-roster",
        "seed": 20260916,
        "splits": {split: COUNT for split in SPLITS},
        "document_disjoint": True,
        "family_disjoint": True,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(output, 0o600)
    return {"path": str(output), "rows": len(rows), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
