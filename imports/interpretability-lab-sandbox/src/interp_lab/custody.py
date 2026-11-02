"""Owner-only custody and packet checks for future real runs.

State slice: interpretability-lab-sandbox-v1.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Any

from .canonical import digest


class CustodyError(ValueError):
    pass


def file_manifest(root: Path) -> dict[str, object]:
    if not root.is_dir():
        raise CustodyError("custody root is missing")
    mode = root.stat().st_mode & 0o777
    if mode & 0o077:
        raise CustodyError("custody root must be owner-only")
    entries = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise CustodyError("custody symlinks are not allowed")
        if path.is_file():
            entries.append({"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if not entries:
        raise CustodyError("custody root is empty")
    return {"schema":"interp-lab.custody-manifest.v1","state_slice":"interpretability-lab-sandbox-v1","files":entries,"manifest_digest":digest(entries)}


def require_owner_only(path: Path) -> None:
    if not path.exists() or path.is_symlink() or (path.stat().st_mode & 0o077):
        raise CustodyError(f"owner-only custody required: {path}")


def verify_packet_binding(packet: Mapping[str, object], expected_digest: str) -> None:
    actual = digest({key: value for key, value in packet.items() if key != "packet_digest"})
    if packet.get("packet_digest") != actual or actual != expected_digest:
        raise CustodyError("packet digest mismatch")


def _sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise CustodyError(f"custody file is missing or is a symlink: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise CustodyError(f"custody input is outside the root: {path}") from exc


def require_internal_custody_binding(
    *,
    root: Path,
    packet: Mapping[str, Any],
    packet_digest: str,
    model_root: Path,
    sae_weights: Path,
    corpus: Path,
    authorization: Path,
    runtime_attestation: Path,
) -> Mapping[str, Any]:
    """Verify the immutable input hashes for an internal model run.

    The manifest is intentionally separate from the packet so it can include
    the packet digest without creating a circular hash.  Every scientific
    input is checked both against the manifest and the packet's declared
    identity before a model library is imported.
    """
    require_owner_only(root)
    manifest_path = root / "custody-manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise CustodyError("internal custody manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CustodyError("internal custody manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise CustodyError("internal custody manifest must be an object")
    required = {"schema", "state_slice", "packet_digest", "components", "manifest_digest"}
    if set(manifest) != required or manifest.get("schema") != "interp-lab.internal-custody-manifest.v1":
        raise CustodyError("internal custody manifest schema mismatch")
    if manifest.get("state_slice") != "interpretability-lab-sandbox-v1" or manifest.get("packet_digest") != packet_digest:
        raise CustodyError("internal custody manifest identity mismatch")
    if digest({key: value for key, value in manifest.items() if key != "manifest_digest"}) != manifest.get("manifest_digest"):
        raise CustodyError("internal custody manifest digest mismatch")
    components = manifest.get("components")
    if not isinstance(components, Mapping) or set(components) != {"model", "sae", "corpus", "authorization", "runtime_attestation"}:
        raise CustodyError("internal custody component roster mismatch")

    paths = {
        "model": model_root,
        "sae": sae_weights,
        "corpus": corpus,
        "authorization": authorization,
        "runtime_attestation": runtime_attestation,
    }
    observed: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        relative = _relative(root, path)
        entry = components[name]
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
            raise CustodyError(f"internal custody component {name} schema mismatch")
        if entry.get("path") != relative:
            raise CustodyError(f"internal custody component {name} path mismatch")
        if path.is_dir():
            observed_digest = str(file_manifest(path)["manifest_digest"])
        else:
            observed_digest = _sha256_file(path)
        if entry.get("sha256") != observed_digest:
            raise CustodyError(f"internal custody component {name} digest mismatch")
        observed[name] = {"path": relative, "sha256": observed_digest}

    model = packet.get("model")
    asset = packet.get("feature_asset")
    corpus_payload = packet.get("corpus")
    runtime = packet.get("runtime")
    if not isinstance(model, Mapping) or not isinstance(asset, Mapping) or not isinstance(corpus_payload, Mapping) or not isinstance(runtime, Mapping):
        raise CustodyError("internal packet identity sections are incomplete")
    checkpoints = model.get("checkpoints")
    base = checkpoints.get("base") if isinstance(checkpoints, Mapping) else None
    if not isinstance(base, Mapping) or base.get("sha256") != observed["model"]["sha256"]:
        raise CustodyError("internal packet model digest is not custody-bound")
    if asset.get("weights_sha256") != observed["sae"]["sha256"] or corpus_payload.get("sha256") != observed["corpus"]["sha256"]:
        raise CustodyError("internal packet feature or corpus digest is not custody-bound")
    if not runtime.get("transformers"):
        raise CustodyError("internal packet Transformers version is not frozen")
    return {"manifest": manifest, "components": observed}
