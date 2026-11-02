#!/usr/bin/env python3
"""Bind the Gemma 3 V9 design to already acquired custody inputs.

State slice: interpretability-lab-sandbox-v1.

This command never downloads, executes, or signs anything. It hashes an
owner-only custody root and emits a new packet, leaving the design JSON
immutable.
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
DESIGN_SCHEMA = "interp-lab.gemma3-reconstruction-protocol.v1"


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


def freeze(*, design: Path, custody_root: Path, model_root: Path, feature_root: Path, corpus: Path, packet_id: str, output: Path) -> dict[str, str]:
    require_owner_only(custody_root)
    if not model_root.is_dir() or not feature_root.is_dir():
        raise ValueError("model and feature custody roots must be directories")
    for path in (model_root, feature_root, corpus, output):
        _relative(custody_root, path)
    if output.resolve() == design.resolve():
        raise ValueError("design packet is immutable; output must be a new packet")
    payload = _read(design)
    if payload.get("state_slice") != STATE_SLICE or payload.get("schema") != DESIGN_SCHEMA:
        raise ValueError("Gemma V9 design identity mismatch")
    if payload.get("status") != "DESIGN_ONLY_NO_MODEL_EXECUTION":
        raise ValueError("input is not the frozen design packet")
    if not packet_id or any(char.isspace() for char in packet_id):
        raise ValueError("a concrete packet id is required")
    model_manifest = file_manifest(model_root)
    feature_manifest = file_manifest(feature_root)
    packet = copy.deepcopy(payload)
    packet["packet_id"] = packet_id
    packet["status"] = "FROZEN_INTERNAL_CUSTODY_BOUND"
    # The design is intentionally closed; a custody-bound packet is the
    # explicit transition into the separately authorized internal-real lane.
    execution = packet.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("Gemma V9 execution policy is missing")
    execution["real_model_enabled"] = True
    packet["model"]["checkpoints"]= {"base": {"sha256": model_manifest["manifest_digest"]}}
    packet["model"]["model_manifest_digest"] = model_manifest["manifest_digest"]
    packet["feature_asset"]["asset_digest"] = feature_manifest["manifest_digest"]
    packet["feature_asset"]["weights_digest"] = feature_manifest["manifest_digest"]
    packet["feature_asset"]["weights_sha256"] = feature_manifest["manifest_digest"]
    packet["feature_asset"]["params_digest"] = _sha256(feature_root / "params.safetensors")
    packet["feature_asset"]["config_digest"] = _sha256(feature_root / "config.json")
    packet["corpus"]["corpus_digest"] = _sha256(corpus)
    packet["corpus"]["sha256"] = _sha256(corpus)
    packet["custody"] = {
        "root": "external-owner-only",
        "input_paths": {
            "model_root": _relative(custody_root, model_root),
            "feature_root": _relative(custody_root, feature_root),
            "corpus": _relative(custody_root, corpus),
            "manifest": "custody-manifest.json",
        },
        "model_manifest_digest": model_manifest["manifest_digest"],
        "feature_manifest_digest": feature_manifest["manifest_digest"],
    }
    packet["packet_digest"] = digest({key: value for key, value in packet.items() if key != "packet_digest"})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(output, 0o600)
    return {"packet_digest": packet["packet_digest"], "model_manifest_digest": str(model_manifest["manifest_digest"]), "feature_manifest_digest": str(feature_manifest["manifest_digest"])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--custody-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--packet-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(design=args.design, custody_root=args.custody_root, model_root=args.model_root, feature_root=args.feature_root, corpus=args.corpus, packet_id=args.packet_id, output=args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
