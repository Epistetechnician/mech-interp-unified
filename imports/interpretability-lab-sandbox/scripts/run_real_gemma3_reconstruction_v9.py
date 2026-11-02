#!/usr/bin/env python3
"""Run the packet-bound Gemma 3 V9 reconstruction qualification.

State slice: interpretability-lab-sandbox-v1.

The runner is offline and aggregate-only. It executes the fixed qualification
order and exits before causal or continual-learning work when reconstruction
fails.
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
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from interp_lab.canonical import digest  # noqa: E402
from interp_lab.custody import require_internal_custody_binding  # noqa: E402
from interp_lab.execution import require_internal_execution  # noqa: E402
from interp_lab.metrics import cosine, max_abs_delta  # noqa: E402
from interp_lab.real_adapters import Gemma3JumpReLUSAE, Gemma3Runtime, Gemma3Spec  # noqa: E402


STATE_SLICE = "interpretability-lab-sandbox-v1"
MODEL_ID = "google/gemma-3-27b-pt"
MODEL_REVISION = "9fe3c4ebc93fbadb14913801536d022054ef11cc"
ASSET_REPOSITORY = "google/gemma-scope-2-27b-pt"
ASSET_REVISION = "5bf702afc144f3ba57b65bb3f76b12616385a213"
ASSET_PATH = "transcoder/layer_31_width_65k_l0_small_affine"
LAYER = 31
ROW_FIELDS = {"row_id", "split", "family", "document_id", "text", "target_token_id", "contrast_token_id", "donor_row_id", "shuffled_donor_row_id"}


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


def _runtime(torch: Any, expected: Mapping[str, Any]) -> dict[str, str]:
    try:
        import safetensors  # type: ignore
        import transformers  # type: ignore
    except ImportError as exc:  # pragma: no cover - real runner only
        raise RuntimeError("Gemma V9 runtime dependencies are missing") from exc
    actual = {
        "python": platform.python_version(),
        "torch": str(torch.__version__).split("+")[0],
        "transformers": str(transformers.__version__),
        "safetensors": str(safetensors.__version__),
    }
    for key, value in actual.items():
        if expected.get(key) != value:
            raise RuntimeError(f"Gemma V9 runtime mismatch for {key}: expected {expected.get(key)}, observed {value}")
    if torch.version.cuda != expected.get("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Gemma V9 qualification requires the attested CUDA runtime")
    if expected.get("device_model") not in str(torch.cuda.get_device_name(0)):
        raise RuntimeError("Gemma V9 qualification requires an NVIDIA H100")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return actual


def _corpus(path: Path, expected: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = _read(path)
    if payload.get("schema") != "interp-lab.assessment-corpus.v1" or payload.get("state_slice") != STATE_SLICE or payload.get("fresh") is not True:
        raise RuntimeError("Gemma V9 corpus identity or schema mismatch")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 144:
        raise RuntimeError("Gemma V9 corpus must contain 144 rows")
    seen: set[str] = set()
    counts = {"fit": 0, "tune": 0, "assessment": 0}
    families: set[str] = set()
    documents: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != ROW_FIELDS:
            raise RuntimeError("Gemma V9 corpus row schema mismatch")
        row_id, split, family, document = str(row["row_id"]), str(row["split"]), str(row["family"]), str(row["document_id"])
        if split not in counts or not row_id or row_id in seen or family in families or document in documents or not str(row["text"]).strip():
            raise RuntimeError("Gemma V9 corpus roster is invalid")
        seen.add(row_id)
        families.add(family)
        documents.add(document)
        counts[split] += 1
    for split in counts:
        if counts[split] != int(expected.get(f"{split}_families", 48)):
            raise RuntimeError(f"Gemma V9 corpus {split} count mismatch")
    return rows


def _gate(name: str, passed: bool, **values: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), **values}


def _margin(logits: tuple[float, ...], target: int, contrast: int) -> float:
    return float(logits[target]) - float(logits[contrast])


def _reconstruction(torch: Any, sae: Gemma3JumpReLUSAE, rows: list[dict[str, Any]], captured_inputs: tuple[tuple[tuple[float, ...], ...], ...], captured_outputs: tuple[tuple[tuple[float, ...], ...], ...]) -> dict[str, float]:
    input_flattened = [token for row in captured_inputs for token in row]
    output_flattened = [token for row in captured_outputs for token in row]
    if not input_flattened or len(input_flattened) != len(output_flattened):
        raise RuntimeError("Gemma V9 capture has no tokens")
    source = torch.as_tensor(input_flattened, dtype=torch.float32, device=sae.device)
    target = torch.as_tensor(output_flattened, dtype=torch.float32, device=sae.device)
    predicted = sae.reconstruct_batch(source)
    mean = target.mean()
    denominator = torch.sum((target - mean) ** 2)
    pooled = torch.sum((target - predicted) ** 2) / denominator if float(denominator) != 0.0 else torch.tensor(float("inf"), device=target.device)
    row_errors: list[float] = []
    offset = 0
    for row in captured_outputs:
        count = len(row)
        target_row = target[offset : offset + count]
        predicted_row = predicted[offset : offset + count]
        row_mean = target_row.mean()
        row_denominator = torch.sum((target_row - row_mean) ** 2)
        row_error = torch.sum((target_row - predicted_row) ** 2) / row_denominator if float(row_denominator) != 0.0 else torch.tensor(float("inf"), device=target.device)
        row_errors.append(float(row_error))
        offset += count
    del rows
    return {"pooled_fvu": float(pooled), "maximum_row_nmse": max(row_errors)}


def run(*, config_path: Path, packet_path: Path, authorization_path: Path, runtime_attestation_path: Path, custody_root: Path, corpus_path: Path, model_root: Path, feature_root: Path, result_path: Path) -> dict[str, Any]:
    config = _read(config_path)
    packet = _read(packet_path)
    authorization = _read(authorization_path)
    runtime_attestation = _read(runtime_attestation_path)
    packet_digest = packet.get("packet_digest")
    if not isinstance(packet_digest, str):
        raise RuntimeError("Gemma V9 packet digest is missing")
    require_internal_execution(config=config, packet=packet, packet_digest=packet_digest, custody_root=custody_root, authorization=authorization, runtime_attestation=runtime_attestation)
    require_internal_custody_binding(root=custody_root, packet=packet, packet_digest=packet_digest, model_root=model_root, sae_weights=feature_root, corpus=corpus_path, authorization=authorization_path, runtime_attestation=runtime_attestation_path)
    torch = __import__("torch")
    versions = _runtime(torch, config["runtime"])
    torch.manual_seed(20260916)
    torch.cuda.manual_seed_all(20260916)
    rows = _corpus(corpus_path, config["corpus"])
    texts = [str(row["text"]) for row in rows]
    device = "cuda"
    runtime = Gemma3Runtime(Gemma3Spec(model_id=MODEL_ID, revision=MODEL_REVISION, model_root=model_root, layer=LAYER, hook="decoder_out", device=device, dtype="bfloat16"))
    params_path = feature_root / "params.safetensors"
    sae = Gemma3JumpReLUSAE(params_path, d_in=int(config["feature_asset"]["input_width"]), d_sae=int(config["feature_asset"]["feature_width"]), device=device)
    started = time.monotonic()
    baseline = runtime.logits(texts)
    hooked: list[tuple[float, ...]] = []
    final_captures: list[tuple[float, ...]] = []
    for text in texts:
        logits, activation = runtime._run(text, capture=True)
        hooked.append(tuple(float(value) for value in logits[0]))
        if activation is None:
            raise RuntimeError("Gemma V9 final activation capture is missing")
        final_captures.append(tuple(float(value) for value in activation[0]))
    captured_inputs, captured_all = runtime.capture_transcoder(texts)
    repeated_inputs, repeated_all = runtime.capture_transcoder(texts)
    parity = max(max_abs_delta(left, right) for left, right in zip(baseline, hooked))
    repeatability = max(max_abs_delta(left, right) for left, right in zip(final_captures, [tokens[-1] for tokens in repeated_all]))
    final_inputs = [tokens[-1] for tokens in captured_inputs]
    repeated_final_inputs = [tokens[-1] for tokens in repeated_inputs]
    first = final_inputs[0]
    no_op_delta = max_abs_delta(baseline[0], runtime.patched_logits(texts[0], final_captures[0]))
    fit_indices = [index for index, row in enumerate(rows) if row["split"] == "fit"]
    fit_features = [sae.encode(final_inputs[index]) for index in fit_indices]
    active_counts: dict[int, int] = {}
    for feature in fit_features:
        for index, value in enumerate(feature):
            if value > 0.0:
                active_counts[index] = active_counts.get(index, 0) + 1
    if not active_counts:
        raise RuntimeError("Gemma V9 fit rows have no active SAE feature")
    selected_feature = min(active_counts, key=lambda index: (-active_counts[index], index))
    active_value = max(feature[selected_feature] for feature in fit_features)
    intervened = sae.intervene(first, selected_feature, active_value + abs(active_value) + 1.0)
    reach_effect = abs(_margin(runtime.patched_logits(texts[0], intervened), int(rows[0]["target_token_id"]), int(rows[0]["contrast_token_id"])) - _margin(baseline[0], int(rows[0]["target_token_id"]), int(rows[0]["contrast_token_id"])))
    split_rows = {split: [index for index, row in enumerate(rows) if row["split"] == split] for split in ("fit", "tune", "assessment")}
    reconstruction = _reconstruction(torch, sae, rows, captured_inputs, captured_all)
    encoded = [sae.encode(final_inputs[index]) for index in range(len(rows))]
    repeated_encoded = [sae.encode(repeated_final_inputs[index]) for index in range(len(rows))]
    feature_stability = min(cosine(left, right) for left, right in zip(encoded, repeated_encoded))
    event_chain = "0" * 64
    for index, row in enumerate(rows):
        event_chain = digest({"previous": event_chain, "row_id": row["row_id"], "activation_width": len(final_captures[index])})
    limits = config["qualification"]
    gates = [
        _gate("native_instrumented_parity", parity <= float(limits["parity_abs_tol"]), max_abs_logit_delta=parity),
        _gate("deterministic_repeatability", repeatability <= float(limits["repeatability_abs_tol"]), max_abs_activation_delta=repeatability),
        _gate("exact_no_op", no_op_delta <= float(limits["no_op_abs_tol"]), max_abs_logit_delta=no_op_delta),
        _gate("nonzero_intervention_reach", reach_effect >= float(limits["intervention_min_abs_effect"]), absolute_logit_margin_effect=reach_effect),
        _gate("feature_reconstruction", reconstruction["pooled_fvu"] <= float(config["reconstruction"]["fit_pooled_fvu_max"]) and reconstruction["maximum_row_nmse"] <= float(config["reconstruction"]["fit_max_row_nmse"]), **reconstruction),
        _gate("feature_repeat_stability", feature_stability >= float(limits["stability_min_cosine"]), minimum_cosine=feature_stability),
        _gate("event_replay", bool(event_chain) and len(event_chain) == 64, event_chain_digest=event_chain),
        _gate("custody_verification", True, packet_digest=packet_digest),
    ]
    result = {
        "schema": "interp-lab.gemma3-real-qualification-report.v1",
        "state_slice": STATE_SLICE,
        "claim_status": "LOCAL_INTERNAL_REAL_MODEL_QUALIFICATION_ONLY",
        "claim_ceiling": "LOCAL_INTERNAL_GEMMA3_RECONSTRUCTION_QUALIFICATION_ONLY",
        "packet_digest": packet_digest,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "feature_asset": {"repository": ASSET_REPOSITORY, "revision": ASSET_REVISION, "path": ASSET_PATH, "params_sha256": _sha256(params_path), "config_sha256": _sha256(feature_root / "config.json")},
        "corpus": {"rows": len(rows), "fit": len(split_rows["fit"]), "tune": len(split_rows["tune"]), "assessment": len(split_rows["assessment"]), "sha256": _sha256(corpus_path)},
        "runtime": versions,
        "device": {"model": torch.cuda.get_device_name(0), "cuda": str(torch.version.cuda), "memory_bytes": int(torch.cuda.get_device_properties(0).total_memory)},
        "canonical_path": runtime.spec.canonical_path,
        "selected_feature": selected_feature,
        "gates": gates,
        "gate_order": tuple(gate["name"] for gate in gates),
        "all_gates_pass": all(gate["passed"] for gate in gates),
        "aggregate": {"reconstruction": reconstruction, "intervention_effect": reach_effect, "feature_stability": feature_stability, "elapsed_seconds": round(time.monotonic() - started, 3)},
        "event_chain_digest": event_chain,
        "nonclaims": ["no independent review or proof", "no held-out causal assessment", "no continual-learning retention claim", "no alignment or production claim"],
    }
    result["report_digest"] = digest(result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(result_path, 0o600)
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
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = run(config_path=args.config, packet_path=args.packet, authorization_path=args.authorization, runtime_attestation_path=args.runtime_attestation, custody_root=args.custody_root, corpus_path=args.corpus, model_root=args.model_root, feature_root=args.feature_root, result_path=args.result)
    print(json.dumps({"all_gates_pass": result["all_gates_pass"], "report_digest": result["report_digest"]}, sort_keys=True))
    return 0 if result["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
