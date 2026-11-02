"""Fail-closed gate for the real held-out assessment packet.

State slice: interpretability-lab-sandbox-v1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

# Keep direct invocation hermetic from a clean checkout.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interp_lab.assessment import FROZEN_STATISTICS  # noqa: E402
from interp_lab.canonical import digest  # noqa: E402
from interp_lab.custody import CustodyError, file_manifest, require_owner_only  # noqa: E402
from interp_lab.review import ReviewVerificationError, verify_external_accept  # noqa: E402


STATE_SLICE = "interpretability-lab-sandbox-v1"


def check_packet_gate(packet: Mapping[str, Any], *, custody_root: Path | None, review_receipt: Mapping[str, Any] | None) -> dict[str, Any]:
    reasons: list[str] = []
    if packet.get("state_slice") != STATE_SLICE:
        reasons.append("packet state slice mismatch")
    recorded = packet.get("packet_digest")
    if not isinstance(recorded, str) or digest({key: value for key, value in packet.items() if key != "packet_digest"}) != recorded:
        reasons.append("packet digest mismatch")
    if packet.get("status") != "FROZEN_DESIGN_READY_FOR_CUSTODY":
        reasons.append("packet is a design freeze, not a custody-ready execution packet")
    corpus = packet.get("corpus")
    if not isinstance(corpus, Mapping) or not corpus.get("sha256"):
        reasons.append("fresh corpus digest is missing")
    model = packet.get("model")
    checkpoints = model.get("checkpoints") if isinstance(model, Mapping) else None
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != {"base", "post_update_1"} or any(not isinstance(value, Mapping) or not value.get("sha256") for value in checkpoints.values()):
        reasons.append("base and post-update checkpoint digests are missing")
    selection = packet.get("selection")
    if not isinstance(selection, Mapping) or type(selection.get("feature_index")) is not int:
        reasons.append("locked feature selection is missing")
    prediction_lock = packet.get("prediction_lock")
    if not isinstance(prediction_lock, Mapping) or not prediction_lock.get("digest"):
        reasons.append("prediction lock digest is missing")
    execution = packet.get("execution")
    if not isinstance(execution, Mapping) or execution.get("mode") != "external-real":
        reasons.append("external-real execution mode is not configured")
    if isinstance(execution, Mapping) and execution.get("network_during_model_execution") is not False:
        reasons.append("network isolation is not frozen")
    if not isinstance(execution, Mapping) or float(execution.get("hard_usd_ceiling", 0.0)) <= 0.0:
        reasons.append("positive hard USD ceiling is missing")
    if packet.get("statistics") != FROZEN_STATISTICS:
        reasons.append("statistical design does not match the frozen assessment plan")
    feature = packet.get("feature_asset")
    if not isinstance(feature, Mapping) or not feature.get("config_sha256") or not feature.get("weights_sha256"):
        reasons.append("feature asset file digests are missing")
    runtime = packet.get("runtime")
    required_runtime = {"python": "3.11.11", "torch": "2.6.0", "transformers": "4.57.3", "safetensors": "0.8.0", "cuda": "12.4", "device_model": "NVIDIA H100", "dtype": "float32"}
    if not isinstance(runtime, Mapping) or any(runtime.get(key) != value for key, value in required_runtime.items()):
        reasons.append("pinned runtime/device identity is missing or mismatched")
    else:
        determinism = runtime.get("determinism")
        if determinism != {"torch_deterministic_algorithms": True, "tf32": False, "cudnn_benchmark": False, "cublas_workspace_config": ":4096:8"}:
            reasons.append("deterministic CUDA settings are not frozen")
    if custody_root is None:
        reasons.append("owner-only assessment custody root is missing")
    else:
        try:
            require_owner_only(custody_root)
            manifest = file_manifest(custody_root)
            if not manifest.get("files"):
                reasons.append("assessment custody root is empty")
        except (CustodyError, OSError) as error:
            reasons.append(str(error))
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
    parser.add_argument("--packet", type=Path, default=Path("configs/gpt2-resid-mid-assessment-v1.json"))
    parser.add_argument("--custody-root", type=Path)
    parser.add_argument("--review-receipt", type=Path)
    args = parser.parse_args()
    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    receipt = json.loads(args.review_receipt.read_text(encoding="utf-8")) if args.review_receipt else None
    result = check_packet_gate(packet, custody_root=args.custody_root, review_receipt=receipt)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "READY_FOR_EXTERNAL_ASSESSMENT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
