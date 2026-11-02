#!/usr/bin/env python3
"""Fail-closed gate for the Gemma V10 real assessment.

State slice: interpretability-lab-sandbox-v1.

The gate verifies the exact custody-bound packet and an external Ed25519
receipt. It never starts a model or provider job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interp_lab.assessment import FROZEN_STATISTICS, PredictionLock  # noqa: E402
from interp_lab.canonical import digest  # noqa: E402
from interp_lab.custody import CustodyError, file_manifest, require_owner_only  # noqa: E402
from interp_lab.review import ReviewVerificationError, verify_external_accept  # noqa: E402


STATE_SLICE = "interpretability-lab-sandbox-v1"
PROTOCOL_ID = "gemma3-27b-heldout-causal-retention-v10"
EXPECTED_MODEL = "google/gemma-3-27b-pt"
EXPECTED_MODEL_REVISION = "9fe3c4ebc93fbadb14913801536d022054ef11cc"
EXPECTED_FEATURE_REPOSITORY = "google/gemma-scope-2-27b-pt"
EXPECTED_FEATURE_REVISION = "5bf702afc144f3ba57b65bb3f76b12616385a213"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _check_manifest(root: Path, packet: Mapping[str, Any], reasons: list[str]) -> None:
    manifest_path = root / "custody-manifest.json"
    try:
        require_owner_only(root)
        manifest = _load(manifest_path)
        if not isinstance(manifest, Mapping):
            raise CustodyError("Gemma V10 custody manifest must be an object")
        required = {"schema", "state_slice", "protocol_id", "components", "manifest_digest"}
        if set(manifest) != required or manifest.get("schema") != "interp-lab.gemma3-v10-custody-manifest.v1":
            raise CustodyError("Gemma V10 custody manifest schema mismatch")
        if manifest.get("state_slice") != STATE_SLICE or manifest.get("protocol_id") != PROTOCOL_ID:
            raise CustodyError("Gemma V10 custody manifest identity mismatch")
        if digest({key: value for key, value in manifest.items() if key != "manifest_digest"}) != manifest.get("manifest_digest"):
            raise CustodyError("Gemma V10 custody manifest digest mismatch")
        custody = packet.get("custody")
        if not isinstance(custody, Mapping) or custody.get("manifest_digest") != manifest.get("manifest_digest"):
            raise CustodyError("Gemma V10 packet is not bound to custody manifest")
        component_paths = manifest.get("components")
        if not isinstance(component_paths, Mapping):
            raise CustodyError("Gemma V10 custody component roster is missing")
        for name, entry in component_paths.items():
            if not isinstance(entry, Mapping) or entry.get("path") is None:
                raise CustodyError(f"Gemma V10 custody component {name} is malformed")
            component = root / str(entry["path"])
            try:
                component.resolve().relative_to(root.resolve())
            except ValueError as error:
                raise CustodyError(f"Gemma V10 custody component {name} escapes root") from error
            if not component.exists() or component.is_symlink():
                raise CustodyError(f"Gemma V10 custody component {name} is missing")
            if component.is_dir():
                observed_digest = str(file_manifest(component)["manifest_digest"])
                if entry.get("manifest_digest") != observed_digest:
                    raise CustodyError(f"Gemma V10 custody component {name} manifest mismatch")
            elif entry.get("sha256") != _sha256(component):
                raise CustodyError(f"Gemma V10 custody component {name} digest mismatch")
    except (CustodyError, OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        reasons.append(str(error))


def check_packet_gate(packet: Mapping[str, Any], *, custody_root: Path | None, review_receipt: Mapping[str, Any] | None) -> dict[str, Any]:
    reasons: list[str] = []
    if packet.get("state_slice") != STATE_SLICE:
        reasons.append("packet state slice mismatch")
    if packet.get("protocol_id") != PROTOCOL_ID:
        reasons.append("packet protocol identity mismatch")
    recorded = packet.get("packet_digest")
    if not isinstance(recorded, str) or digest({key: value for key, value in packet.items() if key != "packet_digest"}) != recorded:
        reasons.append("packet digest mismatch")
    if packet.get("status") != "CUSTODY_BOUND_PENDING_EXTERNAL_ACCEPT":
        reasons.append("packet is not the custody-bound V10 execution packet")
    model = packet.get("model")
    if not isinstance(model, Mapping) or model.get("id") != EXPECTED_MODEL or model.get("revision") != EXPECTED_MODEL_REVISION:
        reasons.append("Gemma model identity is missing or mismatched")
    checkpoints = model.get("checkpoints") if isinstance(model, Mapping) else None
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != {"base", "post_update_1"} or any(not isinstance(value, Mapping) or not value.get("manifest_digest") for value in checkpoints.values()):
        reasons.append("base and post-update checkpoint manifests are missing")
    feature = packet.get("feature_asset")
    if not isinstance(feature, Mapping) or feature.get("repository") != EXPECTED_FEATURE_REPOSITORY or feature.get("revision") != EXPECTED_FEATURE_REVISION:
        reasons.append("Gemma Scope feature identity is missing or mismatched")
    if not isinstance(feature, Mapping) or feature.get("input_width") != 5376 or feature.get("feature_width") != 65536 or not feature.get("config_sha256") or not feature.get("params_sha256"):
        reasons.append("feature dimensions or file digests are missing")
    corpus = packet.get("corpus")
    if not isinstance(corpus, Mapping) or corpus.get("sha256") is None or corpus.get("splits") != {"fit": 48, "tune": 48, "assessment": 48}:
        reasons.append("fresh V10 corpus binding is missing")
    selection = packet.get("selection")
    if not isinstance(selection, Mapping) or type(selection.get("feature_index")) is not int or not 0 <= selection["feature_index"] < 65536:
        reasons.append("fit/tune feature selection is missing or out of range")
    prediction_lock = packet.get("prediction_lock")
    if not isinstance(prediction_lock, Mapping):
        reasons.append("prediction lock is missing")
    else:
        try:
            lock = PredictionLock(feature_index=int(prediction_lock["feature_index"]), expected_sign=int(prediction_lock["expected_sign"]), source=str(prediction_lock["source"]))
            if prediction_lock.get("digest") != lock.digest or lock.feature_index != selection.get("feature_index"):
                reasons.append("prediction lock is not bound to selected feature")
        except (KeyError, TypeError, ValueError):
            reasons.append("prediction lock is malformed")
    runtime = packet.get("runtime")
    required_runtime = {"python": "3.11.11", "torch": "2.6.0", "transformers": "4.57.3", "safetensors": "0.8.0", "cuda": "12.4", "dtype": "bfloat16", "device_model": "NVIDIA H100"}
    if not isinstance(runtime, Mapping) or any(runtime.get(key) != value for key, value in required_runtime.items()):
        reasons.append("pinned runtime/device identity is missing or mismatched")
    elif runtime.get("determinism") != {"torch_deterministic_algorithms": True, "tf32": False, "cudnn_benchmark": False, "cublas_workspace_config": ":4096:8"}:
        reasons.append("deterministic CUDA settings are not frozen")
    if not isinstance(runtime, Mapping) or not runtime.get("attestation_sha256"):
        reasons.append("runtime attestation digest is missing")
    execution = packet.get("execution")
    if not isinstance(execution, Mapping) or execution.get("mode") != "external-real" or execution.get("network_during_model_execution") is not False:
        reasons.append("offline external-real execution mode is not frozen")
    if not isinstance(execution, Mapping) or float(execution.get("hard_usd_ceiling", 0.0)) <= 0.0:
        reasons.append("positive hard USD ceiling is missing")
    if packet.get("statistics") != FROZEN_STATISTICS:
        reasons.append("statistical design does not match the frozen assessment plan")
    if custody_root is None:
        reasons.append("owner-only V10 custody root is missing")
    else:
        _check_manifest(custody_root, packet, reasons)
    if review_receipt is None:
        reasons.append("independent packet-bound Ed25519 ACCEPT receipt is missing")
    else:
        try:
            verify_external_accept(review_receipt, str(recorded))
        except ReviewVerificationError as error:
            reasons.append(str(error))
    status = "READY_FOR_EXTERNAL_ASSESSMENT" if not reasons else "CLOSED"
    return {"status": status, "state_slice": STATE_SLICE, "packet_digest": recorded, "reasons": reasons}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--custody-root", type=Path)
    parser.add_argument("--review-receipt", type=Path)
    args = parser.parse_args()
    packet = _load(args.packet)
    receipt = _load(args.review_receipt) if args.review_receipt else None
    result = check_packet_gate(packet, custody_root=args.custody_root, review_receipt=receipt)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "READY_FOR_EXTERNAL_ASSESSMENT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
