#!/usr/bin/env python3
"""Prepare and run the Gemma 3 V9 qualification on an acquired node.

State slice: interpretability-lab-sandbox-v1.

Acquisition is intentionally separate. This command accepts only an existing
owner-only model and feature custody root, creates the packet and local
authorization, attests the runtime, writes the manifest, and then switches to
offline execution.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.freeze_gemma3_v9_packet import freeze  # noqa: E402
from scripts.generate_gemma3_v9_corpus import generate  # noqa: E402
from scripts.run_real_gemma3_reconstruction_v9 import run  # noqa: E402
from scripts.write_internal_custody_manifest import write_manifest  # noqa: E402


STATE_SLICE = "interpretability-lab-sandbox-v1"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def prepare_and_run(*, custody_root: Path, design_path: Path, operator_id: str, packet_id: str) -> dict[str, Any]:
    custody_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(custody_root, 0o700)
    model_root = custody_root / "model"
    feature_root = custody_root / "feature"
    corpus_path = custody_root / "corpus.json"
    packet_path = custody_root / "packet.json"
    authorization_path = custody_root / "authorization.json"
    attestation_path = custody_root / "runtime-attestation.json"
    result_path = custody_root / "result.json"
    if not model_root.is_dir() or not feature_root.is_dir():
        raise RuntimeError("Gemma V9 acquisition is incomplete: model or feature root is missing")
    if not (feature_root / "config.json").is_file() or not (feature_root / "params.safetensors").is_file():
        raise RuntimeError("Gemma V9 feature root must contain config.json and params.safetensors")
    corpus_info = generate(corpus_path)
    packet_info = freeze(design=design_path, custody_root=custody_root, model_root=model_root, feature_root=feature_root, corpus=corpus_path, packet_id=packet_id, output=packet_path)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet_digest = str(packet["packet_digest"])
    authorization = {
        "schema": "interp-lab.internal-authorization.v1",
        "state_slice": STATE_SLICE,
        "source": "internal",
        "decision": "AUTHORIZE",
        "purpose": "exploration",
        "packet_digest": packet_digest,
        "operator_id": operator_id,
        "claim_ceiling": "LOCAL_INTERNAL_EXPLORATORY_ONLY",
    }
    _write_json(authorization_path, authorization)
    import safetensors  # type: ignore
    import torch  # type: ignore
    import transformers  # type: ignore

    if not torch.cuda.is_available():
        raise RuntimeError("Gemma V9 runtime attestation requires CUDA")
    attestation = {
        "schema": "interp-lab.runtime-attestation.v1",
        "state_slice": STATE_SLICE,
        "source": "internal",
        "runtime": {"python": platform.python_version(), "framework": "torch", "torch": str(torch.__version__).split("+")[0], "transformers": str(transformers.__version__), "safetensors": str(safetensors.__version__)},
        "device": {"model": torch.cuda.get_device_name(0), "cuda": str(torch.version.cuda), "index": 0, "memory_bytes": int(torch.cuda.get_device_properties(0).total_memory)},
        "determinism": {"enabled": True, "torch_deterministic_algorithms": True, "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"), "tf32": False, "cudnn_allow_tf32": False, "cudnn_benchmark": False},
    }
    _write_json(attestation_path, attestation)
    write_manifest(root=custody_root, packet=packet_path, model_root=model_root, sae_weights=feature_root, corpus=corpus_path, authorization=authorization_path, runtime_attestation=attestation_path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    result = run(config_path=packet_path, packet_path=packet_path, authorization_path=authorization_path, runtime_attestation_path=attestation_path, custody_root=custody_root, corpus_path=corpus_path, model_root=model_root, feature_root=feature_root, result_path=result_path)
    return {"packet": packet_info, "corpus": corpus_info, "result": result}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--custody-root", type=Path, default=Path("/hfcache/interp-lab-gemma3-v9"))
    parser.add_argument("--design", type=Path, default=ROOT / "configs/gemma3-27b-reconstruction-v9-design-only.json")
    parser.add_argument("--operator-id", default="gman:patel.s.shaan21@gmail.com")
    parser.add_argument("--packet-id", default="gemma3-v9-reconstruction-gman-20260916")
    args = parser.parse_args()
    payload = prepare_and_run(custody_root=args.custody_root, design_path=args.design, operator_id=args.operator_id, packet_id=args.packet_id)
    print(json.dumps({"packet": payload["packet"], "all_gates_pass": payload["result"]["all_gates_pass"], "report_digest": payload["result"]["report_digest"]}, sort_keys=True))
    return 0 if payload["result"]["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
