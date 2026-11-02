#!/usr/bin/env python3
"""Prepare and run a packet-bound Qwen reconstruction qualification in Gman.

State slice: interpretability-lab-sandbox-v1.

This entrypoint assumes that an acquisition-only job has already placed the
exact model and fresh layer-33 SAE under ``/hfcache``.  It creates the fresh
corpus, packet, internal authorization, runtime attestation, and custody
manifest before enabling offline model execution.  It emits aggregate output
only; raw activations never leave process memory.
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

from interp_lab.reconstruction_packet import freeze_reconstruction_packet  # noqa: E402
from scripts.generate_qwen38_reconstruction_v5_corpus import generate  # noqa: E402
from scripts.run_real_qwen38_reconstruction_v5 import run  # noqa: E402
from scripts.write_internal_custody_manifest import write_manifest  # noqa: E402


STATE_SLICE = "interpretability-lab-sandbox-v1"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SAE_REVISION = "0f965454ffedb93e826546a7dc11b51a4622188e"
TRANSFORMERS_VERSION = "4.57.3"
TRANSFORMERS_SOURCE_REVISION = "v4.57.3"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def prepare_and_run(*, custody_root: Path, design_path: Path, operator_id: str, packet_id: str, transformers_version: str = TRANSFORMERS_VERSION, transformers_source_revision: str = TRANSFORMERS_SOURCE_REVISION, torchvision_version: str | None = None, pillow_version: str | None = None) -> dict[str, Any]:
    custody_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(custody_root, 0o700)
    model_root = custody_root / "model"
    sae_weights = custody_root / "sae" / "v2" / "sae_L33_final.pt"
    corpus_path = custody_root / "corpus.json"
    packet_path = custody_root / "packet.json"
    authorization_path = custody_root / "authorization.json"
    attestation_path = custody_root / "runtime-attestation.json"
    result_path = custody_root / "result.json"
    if not model_root.is_dir() or not sae_weights.is_file():
        raise RuntimeError("V5 acquisition is incomplete: model or layer-33 SAE is missing")
    corpus_info = generate(corpus_path)
    packet_info = freeze_reconstruction_packet(
        design_path=design_path,
        custody_root=custody_root,
        model_root=model_root,
        sae_weights=sae_weights,
        corpus_path=corpus_path,
        corpus_identity=f"{packet_id}-layer33-corpus-{corpus_info['sha256']}",
        transformers_version=transformers_version,
        transformers_source_revision=transformers_source_revision,
        packet_id=packet_id,
        output_path=packet_path,
    )
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
    torchvision = None
    pillow = None
    if torchvision_version is not None or pillow_version is not None:
        try:
            import torchvision as _torchvision  # type: ignore
            from PIL import __version__ as _pillow_version  # type: ignore
        except ImportError as exc:
            raise RuntimeError("V7 vision runtime dependencies are missing") from exc
        torchvision = _torchvision
        pillow = _pillow_version

    if not torch.cuda.is_available():
        raise RuntimeError("V5 runtime attestation requires CUDA")
    attestation = {
        "schema": "interp-lab.runtime-attestation.v1",
        "state_slice": STATE_SLICE,
        "source": "internal",
        "runtime": {
            "python": platform.python_version(),
            "framework": "torch",
            "torch": str(torch.__version__).split("+")[0],
            "transformers": str(transformers.__version__),
            "safetensors": str(safetensors.__version__),
        },
        "device": {
            "model": torch.cuda.get_device_name(0),
            "cuda": str(torch.version.cuda),
            "index": 0,
            "memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        },
        "determinism": {
            "enabled": True,
            "torch_deterministic_algorithms": True,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
            "tf32": False,
            "cudnn_allow_tf32": False,
            "cudnn_benchmark": False,
        },
    }
    if torchvision_version is not None or pillow_version is not None:
        if torchvision is None or pillow is None:
            raise RuntimeError("V7 vision runtime dependencies are missing")
        attestation["runtime"]["torchvision"] = str(torchvision.__version__).split("+")[0]
        attestation["runtime"]["pillow"] = str(pillow)
    _write_json(attestation_path, attestation)
    write_manifest(
        root=custody_root,
        packet=packet_path,
        model_root=model_root,
        sae_weights=sae_weights,
        corpus=corpus_path,
        authorization=authorization_path,
        runtime_attestation=attestation_path,
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    result = run(
        config_path=packet_path,
        packet_path=packet_path,
        authorization_path=authorization_path,
        runtime_attestation_path=attestation_path,
        custody_root=custody_root,
        corpus_path=corpus_path,
        model_root=model_root,
        sae_weights_path=sae_weights,
        result_path=result_path,
    )
    return {"packet": packet_info, "result": result}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--custody-root", type=Path, default=Path("/hfcache/interp-lab-reconstruction-v5"))
    parser.add_argument("--design", type=Path, default=ROOT / "configs/qwen3.8-27b-reconstruction-v5-layer33-design-only.json")
    parser.add_argument("--operator-id", default="gman:patel.s.shaan21@gmail.com")
    parser.add_argument("--packet-id", default="qwen38-v5-layer33-gman-20260916")
    parser.add_argument("--transformers-version", default=TRANSFORMERS_VERSION)
    parser.add_argument("--transformers-source-revision", default=TRANSFORMERS_SOURCE_REVISION)
    parser.add_argument("--torchvision-version")
    parser.add_argument("--pillow-version")
    args = parser.parse_args()
    payload = prepare_and_run(custody_root=args.custody_root, design_path=args.design, operator_id=args.operator_id, packet_id=args.packet_id, transformers_version=args.transformers_version, transformers_source_revision=args.transformers_source_revision, torchvision_version=args.torchvision_version, pillow_version=args.pillow_version)
    print(json.dumps({"packet": payload["packet"], "all_gates_pass": payload["result"]["all_gates_pass"], "report_digest": payload["result"]["report_digest"]}, sort_keys=True))
    return 0 if payload["result"]["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
