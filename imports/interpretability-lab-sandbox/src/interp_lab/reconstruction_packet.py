"""Compile the V5 reconstruction design into a custody-bound packet.

State slice: interpretability-lab-sandbox-v1.

This module hashes already acquired owner-only inputs.  It never downloads,
loads a model, signs a review, or authorizes execution by itself.
"""
from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .canonical import digest
from .custody import CustodyError, file_manifest, require_owner_only


STATE_SLICE = "interpretability-lab-sandbox-v1"
DESIGN_SCHEMA = "interp-lab.qwen38-reconstruction-protocol.v1"
V6_DESIGN_SCHEMA = "interp-lab.qwen38-reconstruction-protocol.v2"
V7_DESIGN_SCHEMA = "interp-lab.qwen38-reconstruction-protocol.v3"
PACKET_SCHEMA = "interp-lab.assessment-packet.v1"
V6_TRANSFORMERS_VERSION = "5.18.0.dev0"
V6_TRANSFORMERS_SOURCE_REVISION = "dbea10932a0e18f40d8103a3bae6fa7927f8d36e"
V7_TORCHVISION_VERSION = "0.21.0"
V7_PILLOW_VERSION = "11.3.0"


class ReconstructionPacketError(ValueError):
    """Raised when a V5 design or custody input is incomplete."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionPacketError(f"JSON object is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ReconstructionPacketError(f"JSON object required: {path}")
    return value


def _sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ReconstructionPacketError(f"custody file is missing or a symlink: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _relative(root: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ReconstructionPacketError(f"input is outside custody root: {path}") from exc
    if path.is_symlink():
        raise ReconstructionPacketError(f"custody symlinks are not allowed: {path}")
    return relative


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or any(char.isspace() for char in value):
        raise ReconstructionPacketError(f"{field} must be a nonempty token")
    return value


def _validate_design(design: Mapping[str, Any]) -> None:
    schema = design.get("schema")
    is_v6 = schema == V6_DESIGN_SCHEMA
    is_v7 = schema == V7_DESIGN_SCHEMA
    if schema not in {DESIGN_SCHEMA, V6_DESIGN_SCHEMA, V7_DESIGN_SCHEMA} or design.get("state_slice") != STATE_SLICE:
        raise ReconstructionPacketError("Qwen reconstruction design identity mismatch")
    if is_v6 and design.get("continuation") != "transformers-main-qwen3_5":
        raise ReconstructionPacketError("V6 continuation identity is not frozen")
    if is_v7 and design.get("continuation") != "transformers-main-qwen3_5-vision-runtime":
        raise ReconstructionPacketError("V7 continuation identity is not frozen")
    if design.get("status") != "DESIGN_ONLY_NO_MODEL_EXECUTION":
        raise ReconstructionPacketError("V5 input is not the immutable design-only boundary")
    model = design.get("model")
    asset = design.get("feature_asset")
    corpus = design.get("corpus")
    runtime = design.get("runtime")
    execution = design.get("execution")
    if not isinstance(model, Mapping) or not isinstance(asset, Mapping) or not isinstance(corpus, Mapping) or not isinstance(runtime, Mapping) or not isinstance(execution, Mapping):
        raise ReconstructionPacketError("Qwen design sections are incomplete")
    if model.get("id") != "Qwen/Qwen3.8-27B" or model.get("revision") != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0":
        raise ReconstructionPacketError("Qwen model identity is not frozen")
    if (model.get("hidden_width"), model.get("layers")) != (5120, 64):
        raise ReconstructionPacketError("Qwen model shape is not frozen")
    if (asset.get("input_width"), asset.get("feature_width"), asset.get("top_k")) != (5120, 40960, 50):
        raise ReconstructionPacketError("Qwen feature shape or Top-K is not frozen")
    if asset.get("normalization") != "x_scaled_by_sqrt_d_over_mean_norm" or asset.get("decoder_rows") != "unit_norm":
        raise ReconstructionPacketError("Qwen feature normalization contract is not frozen")
    if corpus.get("fresh") is not True or corpus.get("document_disjoint") is not True or corpus.get("token_capture") != "all_nonpadding_positions":
        raise ReconstructionPacketError("Qwen corpus contract is incomplete")
    if execution.get("hard_usd_ceiling") != 333.0 or execution.get("network_during_model_execution") is not False:
        raise ReconstructionPacketError("Qwen execution ceiling or offline requirement changed")
    hook_selection = design.get("hook_selection")
    candidates = hook_selection.get("candidates") if isinstance(hook_selection, Mapping) else None
    if not isinstance(candidates, list) or [item.get("id") for item in candidates if isinstance(item, Mapping)] != ["decoder_input", "decoder_output"]:
        raise ReconstructionPacketError("Qwen hook candidates are not fixed")
    layer = hook_selection.get("layer", 32) if isinstance(hook_selection, Mapping) else 32
    if not isinstance(layer, int) or not 0 <= layer < int(model["layers"]):
        raise ReconstructionPacketError("Qwen canonical decoder layer is invalid")
    expected_path = f"model.language_model.layers[{layer}]"
    if any(not isinstance(item, Mapping) or item.get("path_template") != expected_path for item in candidates):
        raise ReconstructionPacketError("Qwen hook paths are not bound to the canonical layer")
    strategy = asset.get("strategy")
    if strategy == "upstream_model_matched_affine_topk_sae":
        for field in ("repository", "revision", "path"):
            if not isinstance(asset.get(field), str) or not asset[field] or any(char.isspace() for char in asset[field]):
                raise ReconstructionPacketError("Qwen upstream feature asset identity is incomplete")
        if asset.get("path") != "v2/sae_L33_final.pt" or layer != 33:
            raise ReconstructionPacketError("Qwen upstream feature asset is not the locked fresh layer-33 pair")
    elif strategy != "train_model_matched_affine_topk_sae":
        raise ReconstructionPacketError("Qwen feature asset strategy is not frozen")
    if is_v6 or is_v7:
        if runtime.get("transformers") != V6_TRANSFORMERS_VERSION or runtime.get("transformers_source_revision") != V6_TRANSFORMERS_SOURCE_REVISION:
            raise ReconstructionPacketError("V6/V7 Transformers source/runtime binding is not frozen")
        if is_v7 and (runtime.get("torchvision") != V7_TORCHVISION_VERSION or runtime.get("pillow") != V7_PILLOW_VERSION):
            raise ReconstructionPacketError("V7 vision runtime binding is not frozen")
    elif runtime.get("transformers") is not None or runtime.get("transformers_source_revision") is not None:
        raise ReconstructionPacketError("V5 design must leave runtime revision binding to packet compilation")


def freeze_reconstruction_packet(
    *,
    design_path: Path,
    custody_root: Path,
    model_root: Path,
    sae_weights: Path,
    corpus_path: Path,
    corpus_identity: str,
    transformers_version: str,
    transformers_source_revision: str,
    packet_id: str,
    output_path: Path,
) -> dict[str, str]:
    """Bind the immutable V5 design to already acquired custody inputs."""

    try:
        require_owner_only(custody_root)
    except CustodyError as exc:
        raise ReconstructionPacketError(str(exc)) from exc
    design = _read_json(design_path)
    _validate_design(design)
    _required_text(corpus_identity, "corpus_identity")
    _required_text(transformers_version, "transformers_version")
    _required_text(transformers_source_revision, "transformers_source_revision")
    _required_text(packet_id, "packet_id")
    if output_path.resolve() == design_path.resolve():
        raise ReconstructionPacketError("design-only input is immutable; output must be a new packet")
    for path in (model_root, sae_weights, corpus_path, output_path):
        _relative(custody_root, path)
    try:
        model_manifest = file_manifest(model_root)
    except CustodyError as exc:
        raise ReconstructionPacketError(str(exc)) from exc
    sae_digest = _sha256_file(sae_weights)
    corpus_digest = _sha256_file(corpus_path)
    model = design["model"]
    asset = design["feature_asset"]
    corpus = design["corpus"]
    runtime = deepcopy(design["runtime"])
    runtime["transformers"] = transformers_version
    runtime["transformers_source_revision"] = transformers_source_revision
    hook_selection = deepcopy(design["hook_selection"])
    hook_layer = int(hook_selection.get("layer", 32))
    asset_strategy = str(asset["strategy"])
    external_asset = asset_strategy == "upstream_model_matched_affine_topk_sae"
    packet: dict[str, Any] = {
        "schema": PACKET_SCHEMA,
        "state_slice": STATE_SLICE,
        "status": "FROZEN_INTERNAL_CUSTODY_BOUND",
        "packet_id": packet_id,
        "claim_ceiling": "LOCAL_INTERNAL_EXPLORATORY_ONLY",
        "purpose": design["purpose"],
        "mode": "internal-real",
        "execution": {
            **deepcopy(design["execution"]),
            "mode": "internal-real",
            "review_mode": "internal",
            "real_model_enabled": True,
        },
        "model": {
            "id": model["id"],
            "revision": model["revision"],
            "architecture": "Qwen3_5ForConditionalGeneration",
            "hidden_size": model["hidden_width"],
            "layers": model["layers"],
            "canonical_layer": hook_layer,
            "canonical_path": f"model.language_model.layers[{hook_layer}]",
            "checkpoints": {"base": {"identity": f"{packet_id}:base", "sha256": model_manifest["manifest_digest"]}},
        },
        "runtime": runtime,
        "feature_asset": {
            "strategy": asset_strategy,
            "repository": asset.get("repository") if external_asset else "local-model-matched",
            "revision": asset.get("revision") if external_asset else "custody-trained",
            "path": _relative(custody_root, sae_weights),
            "d_in": asset["input_width"],
            "d_sae": asset["feature_width"],
            "top_k": asset["top_k"],
            "normalization": asset["normalization"],
            "decoder_rows": asset["decoder_rows"],
            "weights_sha256": sae_digest,
        },
        "corpus": {
            "identity": corpus_identity,
            "source": "fresh-owner-custody",
            "fresh": True,
            "document_disjoint": corpus["document_disjoint"],
            "token_capture": corpus["token_capture"],
            "splits": {"fit": corpus["fit_families"], "tune": corpus["tune_families"], "assessment": corpus["assessment_families"]},
            "sha256": corpus_digest,
        },
        "hook_selection": hook_selection,
        "reconstruction": deepcopy(design["reconstruction"]),
        "downstream": deepcopy(design["downstream"]),
        "custody": {
            "root_required": True,
            "owner_only_mode": "0700",
            "raw_inputs_outside_git": True,
            "input_paths": {
                "model_root": _relative(custody_root, model_root),
                "sae_weights": _relative(custody_root, sae_weights),
                "corpus": _relative(custody_root, corpus_path),
                "manifest": "custody-manifest.json",
            },
            "model_manifest_digest": model_manifest["manifest_digest"],
        },
        "nonclaims": [
            "no independent review or proof",
            "no held-out scientific assessment claim",
            "no continual-learning retention claim",
            "no alignment or production claim",
        ],
    }
    if design.get("schema") == V7_DESIGN_SCHEMA:
        packet["protocol"] = "qwen38-reconstruction-v7"
        packet["design_schema"] = V7_DESIGN_SCHEMA
        packet["continuation"] = "transformers-main-qwen3_5-vision-runtime"
    elif design.get("schema") == V6_DESIGN_SCHEMA:
        packet["protocol"] = "qwen38-reconstruction-v6"
        packet["design_schema"] = V6_DESIGN_SCHEMA
        packet["continuation"] = "transformers-main-qwen3_5"
    else:
        packet["protocol"] = "qwen38-reconstruction-v5"
        packet["design_schema"] = DESIGN_SCHEMA
    packet["packet_digest"] = digest(packet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(output_path, 0o600)
    return {
        "packet_digest": packet["packet_digest"],
        "model_manifest_digest": model_manifest["manifest_digest"],
        "sae_sha256": sae_digest,
        "corpus_sha256": corpus_digest,
    }
