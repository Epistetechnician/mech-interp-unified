"""Assemble a custody-bound assessment packet from externally supplied inputs.

State slice: interpretability-lab-sandbox-v1.

This command performs intake only. It never downloads, copies, executes, or
signs anything. The resulting packet still requires an external reviewer
controlled Ed25519 ACCEPT before model execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from interp_lab.assessment import PredictionLock  # noqa: E402
from interp_lab.canonical import digest  # noqa: E402
from interp_lab.custody import file_manifest, require_owner_only  # noqa: E402
from scripts.run_assessment import _rows  # noqa: E402


STATE_SLICE = "interpretability-lab-sandbox-v1"


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


def _write_owner_only(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def assemble(
    *,
    design_packet_path: Path,
    custody_root: Path,
    corpus_path: Path,
    base_model_root: Path,
    update_model_root: Path,
    sae_config_path: Path,
    sae_weights_path: Path,
    prediction_lock_path: Path,
    output_packet_path: Path,
    intake_record_path: Path,
) -> dict[str, Any]:
    require_owner_only(custody_root)
    design = _load(design_packet_path)
    if design.get("state_slice") != STATE_SLICE:
        raise ValueError("design packet state slice mismatch")
    design_digest = design.get("packet_digest")
    if not isinstance(design_digest, str) or digest({key: value for key, value in design.items() if key != "packet_digest"}) != design_digest:
        raise ValueError("design packet digest mismatch")
    if design.get("status") != "FROZEN_DESIGN_REQUIRES_EXTERNAL_CUSTODY":
        raise ValueError("input packet is not the frozen design packet")
    # The immutable design packet lives in Git; every external input and every
    # emitted custody artifact must live under the owner-only root.
    inputs = (corpus_path, base_model_root, update_model_root, sae_config_path, sae_weights_path, prediction_lock_path)
    outputs = (output_packet_path, intake_record_path)
    if any(not _inside(custody_root, path) for path in inputs + outputs):
        raise ValueError("all intake inputs and outputs must be below the owner-only custody root")
    if output_packet_path == design_packet_path:
        raise ValueError("design packet is immutable; output must be a new packet")
    corpus = _load(corpus_path)
    panel, _ = _rows(corpus)
    if {split: len(panel.rows[split]) for split in panel.rows} != design["corpus"]["splits"]:
        raise ValueError("corpus split counts do not match the design packet")
    lock_payload = _load(prediction_lock_path)
    if not isinstance(lock_payload, dict) or set(lock_payload) != {"schema", "state_slice", "feature_index", "expected_sign", "source"}:
        raise ValueError("prediction lock schema mismatch")
    lock = PredictionLock(feature_index=int(lock_payload["feature_index"]), expected_sign=int(lock_payload["expected_sign"]), source=str(lock_payload["source"]))
    if lock_payload != lock.as_dict():
        raise ValueError("prediction lock payload is not canonical")
    if lock.feature_index != design["selection"]["feature_index"]:
        raise ValueError("prediction lock feature does not match the frozen selection")
    checkpoint_manifests = {
        "base": file_manifest(base_model_root),
        "post_update_1": file_manifest(update_model_root),
    }
    packet = deepcopy(design)
    packet["status"] = "FROZEN_DESIGN_READY_FOR_CUSTODY"
    packet["corpus"]["sha256"] = _sha256(corpus_path)
    packet["model"]["checkpoints"]["base"]["sha256"] = checkpoint_manifests["base"]["manifest_digest"]
    packet["model"]["checkpoints"]["post_update_1"]["sha256"] = checkpoint_manifests["post_update_1"]["manifest_digest"]
    packet["feature_asset"]["config_sha256"] = _sha256(sae_config_path)
    packet["feature_asset"]["weights_sha256"] = _sha256(sae_weights_path)
    packet["prediction_lock"]["digest"] = lock.digest
    packet["prediction_lock"]["reviewer"] = None
    packet["custody"]["input_paths"] = {
        "corpus": corpus_path.resolve().relative_to(custody_root.resolve()).as_posix(),
        "base_model_root": base_model_root.resolve().relative_to(custody_root.resolve()).as_posix(),
        "post_update_model_root": update_model_root.resolve().relative_to(custody_root.resolve()).as_posix(),
        "sae_config": sae_config_path.resolve().relative_to(custody_root.resolve()).as_posix(),
        "sae_weights": sae_weights_path.resolve().relative_to(custody_root.resolve()).as_posix(),
        "prediction_lock": prediction_lock_path.resolve().relative_to(custody_root.resolve()).as_posix(),
    }
    packet["custody"]["checkpoint_manifest_digests"] = {label: value["manifest_digest"] for label, value in checkpoint_manifests.items()}
    packet["packet_digest"] = digest({key: value for key, value in packet.items() if key != "packet_digest"})
    _write_owner_only(output_packet_path, packet)
    record = {
        "schema": "interp-lab.assessment-intake-record.v1",
        "state_slice": STATE_SLICE,
        "source_design_packet_digest": design_digest,
        "packet_digest": packet["packet_digest"],
        "corpus_sha256": packet["corpus"]["sha256"],
        "checkpoint_manifest_digests": packet["custody"]["checkpoint_manifest_digests"],
        "sae_config_sha256": packet["feature_asset"]["config_sha256"],
        "sae_weights_sha256": packet["feature_asset"]["weights_sha256"],
        "prediction_lock_digest": lock.digest,
        "review_status": "PENDING_EXTERNAL_PACKET_BOUND_ACCEPT",
    }
    _write_owner_only(intake_record_path, record)
    return {"status": "CUSTODY_PACKET_ASSEMBLED_PENDING_EXTERNAL_REVIEW", "packet_digest": packet["packet_digest"], "intake_record_digest": digest(record)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-packet", type=Path, default=Path("configs/gpt2-resid-mid-assessment-v1.json"))
    parser.add_argument("--custody-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--base-model-root", type=Path, required=True)
    parser.add_argument("--update-model-root", type=Path, required=True)
    parser.add_argument("--sae-config", type=Path, required=True)
    parser.add_argument("--sae-weights", type=Path, required=True)
    parser.add_argument("--prediction-lock", type=Path, required=True)
    parser.add_argument("--output-packet", type=Path, required=True)
    parser.add_argument("--intake-record", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(design_packet_path=args.design_packet, custody_root=args.custody_root, corpus_path=args.corpus, base_model_root=args.base_model_root, update_model_root=args.update_model_root, sae_config_path=args.sae_config, sae_weights_path=args.sae_weights, prediction_lock_path=args.prediction_lock, output_packet_path=args.output_packet, intake_record_path=args.intake_record), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
