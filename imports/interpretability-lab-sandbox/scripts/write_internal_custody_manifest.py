#!/usr/bin/env python3
"""Write the packet-bound input manifest for an internal run.

State slice: interpretability-lab-sandbox-v1.

The manifest contains hashes only.  It never copies model, feature, corpus, or
runtime bytes and does not authorize execution by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interp_lab.canonical import digest  # noqa: E402
from interp_lab.custody import file_manifest, require_owner_only, verify_packet_binding  # noqa: E402


STATE_SLICE = "interpretability-lab-sandbox-v1"


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


def write_manifest(*, root: Path, packet: Path, model_root: Path, sae_weights: Path, corpus: Path, authorization: Path, runtime_attestation: Path) -> dict[str, str]:
    require_owner_only(root)
    packet_payload = json.loads(packet.read_text(encoding="utf-8"))
    if not isinstance(packet_payload, dict) or packet_payload.get("state_slice") != STATE_SLICE:
        raise ValueError("packet state slice mismatch")
    packet_digest = packet_payload.get("packet_digest")
    if not isinstance(packet_digest, str):
        raise ValueError("packet digest is missing")
    verify_packet_binding(packet_payload, packet_digest)
    paths = {"model": model_root, "sae": sae_weights, "corpus": corpus, "authorization": authorization, "runtime_attestation": runtime_attestation}
    components: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        relative = _relative(root, path)
        observed = file_manifest(path)["manifest_digest"] if path.is_dir() else _sha256(path)
        components[name] = {"path": relative, "sha256": str(observed)}
    manifest: dict[str, Any] = {"schema": "interp-lab.internal-custody-manifest.v1", "state_slice": STATE_SLICE, "packet_digest": packet_digest, "components": components}
    manifest["manifest_digest"] = digest(manifest)
    output = root / "custody-manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(output, 0o600)
    return {"manifest_digest": manifest["manifest_digest"], "packet_digest": packet_digest}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--custody-root", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--sae-weights", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--runtime-attestation", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_manifest(root=args.custody_root, packet=args.packet, model_root=args.model_root, sae_weights=args.sae_weights, corpus=args.corpus, authorization=args.authorization, runtime_attestation=args.runtime_attestation), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
