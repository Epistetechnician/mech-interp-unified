#!/usr/bin/env python3
"""Assemble a custody-bound Gemma V10 execution packet.

State slice: interpretability-lab-sandbox-v1.

This is intake only. It never downloads, executes, signs, or treats a local
review as independent. The emitted packet remains locked until an external
Ed25519 ACCEPT is verified against its exact digest.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from interp_lab.assessment import PredictionLock  # noqa: E402
from interp_lab.canonical import digest  # noqa: E402
from interp_lab.custody import file_manifest, require_owner_only  # noqa: E402


STATE_SLICE = "interpretability-lab-sandbox-v1"
PROTOCOL_ID = "gemma3-27b-heldout-causal-retention-v10"
ROW_FIELDS = {
    "row_id",
    "split",
    "family",
    "document_id",
    "text",
    "target_token_id",
    "contrast_token_id",
    "donor_row_id",
    "shuffled_donor_row_id",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_owner_only(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def _validate_corpus(path: Path, design: Mapping[str, Any]) -> dict[str, Any]:
    payload = _load(path)
    corpus_design = design["corpus"]
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "interp-lab.assessment-corpus.v1"
        or payload.get("state_slice") != STATE_SLICE
        or payload.get("fresh") is not True
    ):
        raise ValueError("Gemma V10 corpus schema or freshness is invalid")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Gemma V10 corpus rows are missing")
    expected_counts = corpus_design["splits"]
    counts = {split: 0 for split in expected_counts}
    row_ids: set[str] = set()
    families: set[str] = set()
    documents: set[str] = set()
    id_to_split: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != ROW_FIELDS:
            raise ValueError("Gemma V10 corpus row schema mismatch")
        row_id = str(row["row_id"])
        split = str(row["split"])
        family = str(row["family"])
        document = str(row["document_id"])
        if split not in counts or not row_id or row_id in row_ids:
            raise ValueError("Gemma V10 corpus row identity is invalid")
        if family in families or document in documents or not str(row["text"]).strip():
            raise ValueError("Gemma V10 corpus families or documents are not disjoint")
        row_ids.add(row_id)
        id_to_split[row_id] = split
        families.add(family)
        documents.add(document)
        counts[split] += 1
    if counts != expected_counts:
        raise ValueError(f"Gemma V10 corpus split counts differ: {counts!r}")
    for row in rows:
        if row["split"] == "assessment" and any(
            id_to_split.get(str(row[field])) not in {"fit", "tune"}
            for field in ("donor_row_id", "shuffled_donor_row_id")
        ):
            raise ValueError("Gemma V10 assessment donor is not in the corpus")
    return {"sha256": _sha256(path), "rows": len(rows), "splits": counts}


def _validate_prediction_lock(path: Path, feature_width: int) -> PredictionLock:
    payload = _load(path)
    if not isinstance(payload, dict):
        raise ValueError("Gemma V10 prediction lock must be an object")
    lock = PredictionLock(
        feature_index=int(payload.get("feature_index", -1)),
        expected_sign=int(payload.get("expected_sign", 0)),
        source=str(payload.get("source", "")),
    )
    if set(payload) != {"schema", "state_slice", "feature_index", "expected_sign", "source"}:
        raise ValueError("Gemma V10 prediction lock schema mismatch")
    if payload != lock.as_dict() or lock.feature_index >= feature_width:
        raise ValueError("Gemma V10 prediction lock is noncanonical or out of range")
    return lock


def _validate_attestation(path: Path, design: Mapping[str, Any]) -> dict[str, Any]:
    attestation = _load(path)
    if not isinstance(attestation, dict) or attestation.get("state_slice") != STATE_SLICE:
        raise ValueError("Gemma V10 runtime attestation identity is invalid")
    runtime = attestation.get("runtime")
    expected = design["runtime"]
    if not isinstance(runtime, Mapping):
        raise ValueError("Gemma V10 runtime attestation is missing runtime data")
    for key in ("python", "torch", "transformers", "safetensors"):
        if runtime.get(key) != expected[key]:
            raise ValueError(f"Gemma V10 runtime attestation mismatch for {key}")
    device = attestation.get("device")
    if not isinstance(device, Mapping) or expected["device_model"] not in str(device.get("model")):
        raise ValueError("Gemma V10 runtime attestation is not for an H100")
    if attestation.get("determinism") != expected["determinism"]:
        raise ValueError("Gemma V10 deterministic settings are not attested")
    return attestation


def assemble(
    *,
    design_path: Path,
    custody_root: Path,
    corpus_path: Path,
    base_model_root: Path,
    post_update_model_root: Path,
    sae_config_path: Path,
    sae_params_path: Path,
    prediction_lock_path: Path,
    runtime_attestation_path: Path,
    output_packet_path: Path,
    intake_record_path: Path,
) -> dict[str, Any]:
    require_owner_only(custody_root)
    design = _load(design_path)
    if (
        design.get("state_slice") != STATE_SLICE
        or design.get("protocol_id") != PROTOCOL_ID
        or design.get("status") != "FROZEN_DESIGN_REQUIRES_EXTERNAL_REVIEW"
    ):
        raise ValueError("Gemma V10 design identity or status mismatch")
    inputs = (
        corpus_path,
        base_model_root,
        post_update_model_root,
        sae_config_path,
        sae_params_path,
        prediction_lock_path,
        runtime_attestation_path,
    )
    outputs = (output_packet_path, intake_record_path, custody_root / "custody-manifest.json")
    if any(not _inside(custody_root, path) for path in inputs + outputs):
        raise ValueError("all V10 intake inputs and outputs must be below custody root")
    if output_packet_path.resolve() == design_path.resolve():
        raise ValueError("the design packet is immutable")
    if not base_model_root.is_dir() or not post_update_model_root.is_dir():
        raise ValueError("both V10 checkpoint roots are required")
    if not sae_config_path.is_file() or not sae_params_path.is_file():
        raise ValueError("V10 feature config and params are required")
    corpus_info = _validate_corpus(corpus_path, design)
    lock = _validate_prediction_lock(prediction_lock_path, int(design["feature_asset"]["feature_width"]))
    _validate_attestation(runtime_attestation_path, design)
    checkpoint_manifests = {
        "base": file_manifest(base_model_root),
        "post_update_1": file_manifest(post_update_model_root),
    }
    components = {
        "corpus": {"path": _relative(custody_root, corpus_path), "sha256": corpus_info["sha256"]},
        "base_model": {"path": _relative(custody_root, base_model_root), "manifest_digest": checkpoint_manifests["base"]["manifest_digest"]},
        "post_update_1_model": {"path": _relative(custody_root, post_update_model_root), "manifest_digest": checkpoint_manifests["post_update_1"]["manifest_digest"]},
        "sae_config": {"path": _relative(custody_root, sae_config_path), "sha256": _sha256(sae_config_path)},
        "sae_params": {"path": _relative(custody_root, sae_params_path), "sha256": _sha256(sae_params_path)},
        "prediction_lock": {"path": _relative(custody_root, prediction_lock_path), "sha256": _sha256(prediction_lock_path)},
        "runtime_attestation": {"path": _relative(custody_root, runtime_attestation_path), "sha256": _sha256(runtime_attestation_path)},
    }
    custody_manifest = {
        "schema": "interp-lab.gemma3-v10-custody-manifest.v1",
        "state_slice": STATE_SLICE,
        "protocol_id": PROTOCOL_ID,
        "components": components,
    }
    custody_manifest["manifest_digest"] = digest(custody_manifest)
    _write_owner_only(custody_root / "custody-manifest.json", custody_manifest)
    packet = copy.deepcopy(design)
    packet["status"] = "CUSTODY_BOUND_PENDING_EXTERNAL_ACCEPT"
    packet["corpus"]["sha256"] = corpus_info["sha256"]
    packet["model"]["checkpoints"]["base"]["manifest_digest"] = checkpoint_manifests["base"]["manifest_digest"]
    packet["model"]["checkpoints"]["post_update_1"]["manifest_digest"] = checkpoint_manifests["post_update_1"]["manifest_digest"]
    packet["feature_asset"]["config_sha256"] = components["sae_config"]["sha256"]
    packet["feature_asset"]["params_sha256"] = components["sae_params"]["sha256"]
    packet["feature_asset"]["manifest_digest"] = digest({"config_sha256": components["sae_config"]["sha256"], "params_sha256": components["sae_params"]["sha256"]})
    packet["selection"]["feature_index"] = lock.feature_index
    packet["prediction_lock"] = {**lock.as_dict(), "digest": lock.digest, "reviewer": None, "created_from": "fit_and_tune_only_before_assessment"}
    packet["custody"]["manifest_digest"] = custody_manifest["manifest_digest"]
    packet["custody"]["input_paths"] = {name: value["path"] for name, value in components.items()}
    packet["runtime"]["attestation_sha256"] = components["runtime_attestation"]["sha256"]
    packet["packet_digest"] = digest({key: value for key, value in packet.items() if key != "packet_digest"})
    _write_owner_only(output_packet_path, packet)
    record = {
        "schema": "interp-lab.gemma3-v10-assessment-intake.v1",
        "state_slice": STATE_SLICE,
        "protocol_id": PROTOCOL_ID,
        "packet_digest": packet["packet_digest"],
        "custody_manifest_digest": custody_manifest["manifest_digest"],
        "corpus_sha256": corpus_info["sha256"],
        "checkpoint_manifest_digests": {name: value["manifest_digest"] for name, value in checkpoint_manifests.items()},
        "prediction_lock_digest": lock.digest,
        "review_status": "PENDING_EXTERNAL_PACKET_BOUND_ACCEPT",
    }
    _write_owner_only(intake_record_path, record)
    return {"status": "CUSTODY_BOUND_PENDING_EXTERNAL_ACCEPT", "packet_digest": packet["packet_digest"], "intake_record_digest": digest(record)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, default=ROOT / "configs/gemma3-27b-heldout-causal-retention-v10-design-only.json")
    parser.add_argument("--custody-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--base-model-root", type=Path, required=True)
    parser.add_argument("--post-update-model-root", type=Path, required=True)
    parser.add_argument("--sae-config", type=Path, required=True)
    parser.add_argument("--sae-params", type=Path, required=True)
    parser.add_argument("--prediction-lock", type=Path, required=True)
    parser.add_argument("--runtime-attestation", type=Path, required=True)
    parser.add_argument("--output-packet", type=Path, required=True)
    parser.add_argument("--intake-record", type=Path, required=True)
    args = parser.parse_args()
    result = assemble(
        design_path=args.design,
        custody_root=args.custody_root,
        corpus_path=args.corpus,
        base_model_root=args.base_model_root,
        post_update_model_root=args.post_update_model_root,
        sae_config_path=args.sae_config,
        sae_params_path=args.sae_params,
        prediction_lock_path=args.prediction_lock,
        runtime_attestation_path=args.runtime_attestation,
        output_packet_path=args.output_packet,
        intake_record_path=args.intake_record,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
