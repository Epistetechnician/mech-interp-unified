#!/usr/bin/env python3
"""Bind the Qwen3.8 internal packet to custody inputs.

State slice: interpretability-lab-sandbox-v1.

This command hashes already acquired inputs.  It does not download, execute,
or sign anything, and it never modifies the design packet.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interp_lab.canonical import digest  # noqa: E402
from interp_lab.custody import file_manifest, require_owner_only  # noqa: E402


STATE_SLICE = "interpretability-lab-sandbox-v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"file is missing or a symlink: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside custody root: {path}") from exc


def freeze(
    *,
    design_packet: Path,
    custody_root: Path,
    model_root: Path,
    sae_weights: Path,
    corpus: Path,
    transformers_version: str,
    packet_id: str,
    output_packet: Path,
) -> dict[str, str]:
    require_owner_only(custody_root)
    for path in (model_root, sae_weights, corpus, output_packet):
        _relative(custody_root, path)
    if output_packet.resolve() == design_packet.resolve():
        raise ValueError("design packet is immutable; output must be a new packet")
    design = _read(design_packet)
    if design.get("state_slice") != STATE_SLICE or design.get("schema") != "interp-lab.assessment-packet.v1":
        raise ValueError("internal design packet identity mismatch")
    design_digest = design.get("packet_digest")
    if not isinstance(design_digest, str) or digest({key: value for key, value in design.items() if key != "packet_digest"}) != design_digest:
        raise ValueError("internal design packet digest mismatch")
    if design.get("status") != "FROZEN_INTERNAL_DESIGN_REQUIRES_CUSTODY":
        raise ValueError("input is not the frozen internal design packet")
    if not transformers_version or any(char.isspace() for char in transformers_version):
        raise ValueError("a concrete Transformers version is required")
    model_manifest = file_manifest(model_root)
    packet = copy.deepcopy(design)
    if not packet_id or any(char.isspace() for char in packet_id):
        raise ValueError("a concrete packet id is required")
    packet["packet_id"] = packet_id
    packet["status"] = "FROZEN_INTERNAL_CUSTODY_BOUND"
    packet["model"]["checkpoints"]["base"]["sha256"] = model_manifest["manifest_digest"]
    packet["feature_asset"]["weights_sha256"] = _sha256(sae_weights)
    packet["corpus"]["sha256"] = _sha256(corpus)
    packet["runtime"]["transformers"] = transformers_version
    packet["custody"]["input_paths"] = {
        "model_root": _relative(custody_root, model_root),
        "sae_weights": _relative(custody_root, sae_weights),
        "corpus": _relative(custody_root, corpus),
        "manifest": "custody-manifest.json",
    }
    packet["custody"]["model_manifest_digest"] = model_manifest["manifest_digest"]
    packet["packet_digest"] = digest({key: value for key, value in packet.items() if key != "packet_digest"})
    output_packet.parent.mkdir(parents=True, exist_ok=True)
    output_packet.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(output_packet, 0o600)
    return {"packet_digest": packet["packet_digest"], "design_digest": design_digest, "model_manifest_digest": model_manifest["manifest_digest"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-packet", type=Path, required=True)
    parser.add_argument("--custody-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--sae-weights", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--transformers-version", required=True)
    parser.add_argument("--packet-id", required=True)
    parser.add_argument("--output-packet", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(
        design_packet=args.design_packet,
        custody_root=args.custody_root,
        model_root=args.model_root,
        sae_weights=args.sae_weights,
        corpus=args.corpus,
        transformers_version=args.transformers_version,
        packet_id=args.packet_id,
        output_packet=args.output_packet,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
