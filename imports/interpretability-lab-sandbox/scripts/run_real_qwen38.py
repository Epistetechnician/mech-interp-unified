#!/usr/bin/env python3
"""Run bounded internal Qwen3.8 qualification against fresh custody.

State slice: interpretability-lab-sandbox-v1.

This runner emits aggregate-only internal evidence.  It never produces an
independent review, held-out publication result, continual-learning claim, or
alignment claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from interp_lab.canonical import digest  # noqa: E402
from interp_lab.custody import file_manifest, require_internal_custody_binding  # noqa: E402
from interp_lab.execution import require_internal_execution  # noqa: E402
from interp_lab.metrics import cosine, max_abs_delta, nmse  # noqa: E402
from interp_lab.real_adapters import Qwen38Runtime, Qwen38Spec, Qwen38TopKSAE  # noqa: E402


STATE_SLICE = "interpretability-lab-sandbox-v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


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


def _require_runtime(torch: Any, expected: dict[str, Any]) -> dict[str, str]:
    try:
        import safetensors  # type: ignore
        import transformers  # type: ignore
    except ImportError as exc:  # pragma: no cover - remote runner only
        raise RuntimeError("Qwen3.8 runtime dependencies are missing") from exc
    if not expected.get("transformers"):
        raise RuntimeError("Qwen3.8 Transformers support version must be frozen before execution")
    actual = {
        "python": platform.python_version(),
        "torch": str(torch.__version__).split("+")[0],
        "transformers": str(transformers.__version__),
        "safetensors": str(safetensors.__version__),
    }
    for key, value in actual.items():
        if expected.get(key) != value:
            raise RuntimeError(f"Qwen3.8 runtime mismatch for {key}: expected {expected.get(key)}, observed {value}")
    if torch.version.cuda != expected.get("cuda"):
        raise RuntimeError(f"Qwen3.8 CUDA mismatch: expected {expected.get('cuda')}, observed {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.8 qualification requires the attested CUDA device")
    if expected.get("device_model") not in str(torch.cuda.get_device_name(0)):
        raise RuntimeError("Qwen3.8 qualification requires an NVIDIA H100")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return actual


def _rows(corpus: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    if corpus.get("schema") != "interp-lab.assessment-corpus.v1":
        raise RuntimeError("Qwen3.8 corpus schema mismatch")
    expected = {"row_id", "split", "family", "document_id", "text", "target_token_id", "contrast_token_id", "donor_row_id", "shuffled_donor_row_id"}
    split_rows: dict[str, list[str]] = {"fit": [], "tune": [], "assessment": []}
    by_id: dict[str, dict[str, Any]] = {}
    families: dict[str, set[str]] = {key: set() for key in split_rows}
    documents: dict[str, set[str]] = {key: set() for key in split_rows}
    rows = corpus.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("Qwen3.8 corpus rows are missing")
    for raw in rows:
        if not isinstance(raw, dict) or set(raw) != expected:
            raise RuntimeError("Qwen3.8 corpus row schema mismatch")
        row_id = str(raw["row_id"])
        split = str(raw["split"])
        family = str(raw["family"])
        document = str(raw["document_id"])
        if split not in split_rows or row_id in by_id or not family or not document or not str(raw["text"]).strip():
            raise RuntimeError("Qwen3.8 corpus roster is invalid")
        if family in families[split] or document in documents[split]:
            raise RuntimeError("Qwen3.8 corpus repeats a family or document")
        split_rows[split].append(row_id)
        families[split].add(family)
        documents[split].add(document)
        by_id[row_id] = raw
    if any(families[left] & families[right] or documents[left] & documents[right] for left, right in (("fit", "tune"), ("fit", "assessment"), ("tune", "assessment"))):
        raise RuntimeError("Qwen3.8 corpus family/document disjointness is violated")
    if {key: len(value) for key, value in split_rows.items()} != {"fit": 6, "tune": 6, "assessment": 6}:
        raise RuntimeError("Qwen3.8 internal qualification requires 6/6/6 rows")
    return split_rows, by_id


def _margin(logits: tuple[float, ...], target: int, contrast: int) -> float:
    if not 0 <= target < len(logits) or not 0 <= contrast < len(logits):
        raise RuntimeError("target or contrast token is outside the model vocabulary")
    return float(logits[target] - logits[contrast])


def _feature_index(features: tuple[tuple[float, ...], ...]) -> int:
    counts: dict[int, int] = {}
    for vector in features:
        for index, value in enumerate(vector):
            if value > 0.0:
                counts[index] = counts.get(index, 0) + 1
    if not counts:
        raise RuntimeError("no active Qwen3.8 SAE feature in fit rows")
    return min(counts, key=lambda index: (-counts[index], index))


def _gate(name: str, passed: bool, **values: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), **values}


def run(*, config_path: Path, packet_path: Path, authorization_path: Path, runtime_attestation_path: Path, custody_root: Path, corpus_path: Path, model_root: Path, sae_weights_path: Path, result_path: Path) -> dict[str, Any]:
    config = _read(config_path)
    packet = _read(packet_path)
    authorization = _read(authorization_path)
    runtime_attestation = _read(runtime_attestation_path)
    packet_digest = packet.get("packet_digest")
    if config.get("packet", {}).get("digest") != packet_digest:
        raise RuntimeError("Qwen3.8 internal config is not bound to the packet")
    require_internal_execution(
        config=config,
        packet=packet,
        packet_digest=packet_digest,
        custody_root=custody_root,
        authorization=authorization,
        runtime_attestation=runtime_attestation,
    )
    required_paths = (corpus_path, model_root, sae_weights_path, authorization_path, runtime_attestation_path)
    if any(not _inside(custody_root, path) for path in required_paths):
        raise RuntimeError("Qwen3.8 external inputs must be below the owner-only custody root")
    custody_binding = require_internal_custody_binding(
        root=custody_root,
        packet=packet,
        packet_digest=str(packet_digest),
        model_root=model_root,
        sae_weights=sae_weights_path,
        corpus=corpus_path,
        authorization=authorization_path,
        runtime_attestation=runtime_attestation_path,
    )
    torch = __import__("torch")
    versions = _require_runtime(torch, dict(config["runtime"]))
    torch.manual_seed(20260915)
    torch.cuda.manual_seed_all(20260915)
    corpus = _read(corpus_path)
    split_rows, by_id = _rows(corpus)
    spec = config["model"]
    runtime = Qwen38Runtime(
        Qwen38Spec(
            model_id=str(spec["id"]),
            revision=str(spec["revision"]),
            model_root=model_root,
            layer=int(spec["layer"]),
            hook=str(spec["hook"]),
            device="cuda",
            dtype=str(spec["dtype"]),
        )
    )
    asset = config["feature_asset"]
    sae = Qwen38TopKSAE(sae_weights_path, d_in=int(asset["d_in"]), d_sae=int(asset["d_sae"]), top_k=int(asset["top_k"]))
    all_ids = tuple(split_rows["fit"] + split_rows["tune"] + split_rows["assessment"])
    texts = tuple(by_id[row_id]["text"] for row_id in all_ids)
    started = time.monotonic()
    baseline = runtime.logits(texts)
    captured = runtime.capture(texts)
    hooked = tuple(tuple(float(v) for v in runtime._run(text, capture=True)[0][0]) for text in texts)
    parity = max(max_abs_delta(left, right) for left, right in zip(baseline, hooked))
    repeated = runtime.capture(texts)
    repeat_delta = max(max_abs_delta(left, right) for left, right in zip(captured, repeated))
    features = tuple(sae.encode(vector) for vector in captured)
    selected = _feature_index(features[:6])
    no_op_delta = max_abs_delta(baseline[0], runtime.patched_logits(texts[0], captured[0]))
    active_value = features[0][selected]
    intervention = sae.intervene(captured[0], selected, active_value + abs(active_value) + 1.0)
    intervention_effect = _margin(runtime.patched_logits(texts[0], intervention), int(by_id[all_ids[0]]["target_token_id"]), int(by_id[all_ids[0]]["contrast_token_id"])) - _margin(baseline[0], int(by_id[all_ids[0]]["target_token_id"]), int(by_id[all_ids[0]]["contrast_token_id"]))
    reconstruction_nmse = max(nmse(target, sae.reconstruct(target)) for target in captured)
    feature_cosine = min(cosine(sae.encode(a), sae.encode(b)) for a, b in zip(captured, repeated))
    scrub_effects = []
    for offset, row_id in enumerate(split_rows["assessment"], start=12):
        metadata = by_id[row_id]
        target = int(metadata["target_token_id"])
        contrast = int(metadata["contrast_token_id"])
        base_margin = _margin(baseline[offset], target, contrast)
        scrubbed = sae.intervene(captured[offset], selected, 0.0)
        scrub_effects.append(_margin(runtime.patched_logits(metadata["text"], scrubbed), target, contrast) - base_margin)
    limits = config["qualification"]
    gates = [
        _gate("native_instrumented_parity", parity <= float(limits["parity_abs_tol"]), max_abs_logit_delta=parity),
        _gate("deterministic_repeatability", repeat_delta <= float(limits["repeatability_abs_tol"]), max_abs_activation_delta=repeat_delta),
        _gate("exact_no_op", no_op_delta <= float(limits["no_op_abs_tol"]), max_abs_logit_delta=no_op_delta),
        _gate("nonzero_intervention_reach", abs(intervention_effect) >= float(limits["intervention_min_abs_effect"]), max_abs_logit_effect=abs(intervention_effect)),
        _gate("feature_reconstruction", reconstruction_nmse <= float(limits["reconstruction_nmse_max"]), max_nmse=reconstruction_nmse),
        _gate("feature_repeat_stability", feature_cosine >= float(limits["stability_min_cosine"]), minimum_cosine=feature_cosine),
        _gate("causal_scrubbing_reach", max(abs(value) for value in scrub_effects) >= float(limits["scrub_min_abs_effect"]), maximum_abs_logit_effect=max(abs(value) for value in scrub_effects)),
    ]
    custody = file_manifest(custody_root)
    result = {
        "schema": "interp-lab.real-qwen38-qualification-report.v1",
        "state_slice": STATE_SLICE,
        "claim_status": "LOCAL_INTERNAL_REAL_MODEL_QUALIFICATION_ONLY",
        "claim_ceiling": "LOCAL_INTERNAL_EXPLORATORY_ONLY",
        "packet_digest": packet_digest,
        "model": {"id": spec["id"], "revision": spec["revision"], "canonical_path": Qwen38Spec(model_id=str(spec["id"]), revision=str(spec["revision"]), model_root=model_root, layer=int(spec["layer"])).canonical_path},
        "feature_asset": {"repository": asset["repository"], "revision": asset["revision"], "path": asset["path"], "weights_sha256": _sha256(sae_weights_path)},
        "corpus": {"rows": len(all_ids), "split_counts": {key: len(value) for key, value in split_rows.items()}, "sha256": _sha256(corpus_path)},
        "runtime": versions,
        "device": {"cuda_available": True, "name": torch.cuda.get_device_name(0)},
        "selected_feature": selected,
        "gates": gates,
        "gate_order": tuple(gate["name"] for gate in gates),
        "all_gates_pass": all(gate["passed"] for gate in gates),
        "aggregate": {"intervention_effect": intervention_effect, "reconstruction_nmse": reconstruction_nmse, "scrub_mean_abs_effect": sum(abs(value) for value in scrub_effects) / len(scrub_effects)},
        "custody_manifest_digest": custody["manifest_digest"],
        "input_binding_manifest_digest": custody_binding["manifest"]["manifest_digest"],
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "nonclaims": ("no independent review or proof", "no held-out scientific assessment claim", "no continual-learning retention claim", "no alignment or production claim"),
    }
    result["report_digest"] = digest(result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--runtime-attestation", type=Path, required=True)
    parser.add_argument("--custody-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--sae-weights", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        config_path=args.config,
        packet_path=args.packet,
        authorization_path=args.authorization,
        runtime_attestation_path=args.runtime_attestation,
        custody_root=args.custody_root,
        corpus_path=args.corpus,
        model_root=args.model_root,
        sae_weights_path=args.sae_weights,
        result_path=args.result,
    )
    return 0 if result["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
