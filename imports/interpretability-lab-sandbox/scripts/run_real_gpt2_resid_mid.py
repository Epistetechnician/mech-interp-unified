#!/usr/bin/env python3
"""Run the fresh GPT-2 residual-mid continuation qualification.

State slice: interpretability-lab-sandbox-v1.

This is a new continuation identity. It does not rewrite or rerun the closed
MLP-out slice. The reconstruction gate is pooled across all 18 final-token
activations and is frozen in the continuation config before model loading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Iterable

# CUDA's deterministic cuBLAS path must be selected before the first torch
# CUDA context is created. This is part of the frozen continuation runtime.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from interp_lab.canonical import digest
from interp_lab.custody import file_manifest
from interp_lab.execution import require_internal_execution
from interp_lab.metrics import cosine, max_abs_delta
from interp_lab.real_adapters import AffineSAE, GPT2Runtime, GPT2Spec

STATE_SLICE = "interpretability-lab-sandbox-v1"
MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
FEATURE_REPO = "jbloom/GPT2-Small-OAI-v5-32k-resid-mid-SAEs"
FEATURE_REVISION = "90ed829d1ea6b776c2f4af11184b6d43683e58bc"
HOOK = "resid_mid"
CONTINUATION_ID = "gpt2-resid-mid-20260915-r3"
ROWS = (
    "A red kite crosses the autumn sky.",
    "A careful notebook records each intervention.",
    "The river bends beside a stone footbridge.",
    "A small update can preserve an earlier skill.",
    "The baker counts warm loaves before sunrise.",
    "A compass points north on a moving train.",
    "The analyst checks unseen held out examples.",
    "A bright lantern marks the quiet harbor.",
    "The gardener waters seedlings after rainfall.",
    "A model should expose uncertainty when needed.",
    "The library keeps a catalogue of local histories.",
    "A measured update should preserve old behavior.",
    "The violin rests beside the open window.",
    "A new route joins two distant villages.",
    "The clock records another repeatable observation.",
    "A robust monitor tests predictions before acting.",
    "The mountain trail turns east near the marker.",
    "A final check binds results to their inputs.",
)


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


def _require_versions(torch: Any) -> dict[str, str]:
    try:
        import safetensors
        import transformers
    except ImportError as exc:  # pragma: no cover - remote runner only
        raise RuntimeError("pinned real runtime dependencies are missing") from exc
    expected = {"python": "3.11.11", "torch": "2.6.0", "transformers": "4.57.3", "safetensors": "0.8.0"}
    actual = {
        "python": platform.python_version(),
        "torch": str(torch.__version__).split("+")[0],
        "transformers": str(transformers.__version__),
        "safetensors": str(safetensors.__version__),
    }
    if actual != expected:
        raise RuntimeError(f"runtime version mismatch: expected {expected}, observed {actual}")
    return actual


def _corpus_digest() -> str:
    return digest({"schema": "interp-lab.fresh-corpus.v2", "state_slice": STATE_SLICE, "identity": "resid-mid-continuation-20260915-r3", "rows": ROWS})


def _effect(base: Iterable[float], changed: Iterable[float]) -> float:
    return max_abs_delta(tuple(base), tuple(changed))


def _feature_index(features: tuple[tuple[float, ...], ...]) -> int:
    counts: dict[int, int] = {}
    for vector in features:
        for index, value in enumerate(vector):
            if value > 0.0:
                counts[index] = counts.get(index, 0) + 1
    if not counts:
        raise RuntimeError("no active SAE feature in fit rows")
    return min(counts, key=lambda index: (-counts[index], index))


def _pooled_centered_nmse(targets: tuple[tuple[float, ...], ...], predictions: tuple[tuple[float, ...], ...]) -> float:
    if len(targets) != len(predictions) or not targets:
        raise RuntimeError("reconstruction batches are empty or mismatched")
    flat_target = [float(value) for row in targets for value in row]
    flat_prediction = [float(value) for row in predictions for value in row]
    mean = sum(flat_target) / len(flat_target)
    denominator = sum((value - mean) ** 2 for value in flat_target)
    if denominator == 0.0:
        return 0.0 if flat_target == flat_prediction else float("inf")
    return sum((left - right) ** 2 for left, right in zip(flat_target, flat_prediction)) / denominator


def _gate(name: str, passed: bool, **values: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), **values}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--runtime-attestation", type=Path, required=True)
    parser.add_argument("--custody-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    config = _read(args.config)
    packet = _read(args.packet)
    authorization = _read(args.authorization)
    runtime_attestation = _read(args.runtime_attestation)
    packet_digest = packet.get("packet_digest")
    if not isinstance(packet_digest, str):
        raise RuntimeError("packet digest is missing")
    require_internal_execution(
        config=config,
        packet=packet,
        packet_digest=packet_digest,
        custody_root=args.custody_root,
        authorization=authorization,
        runtime_attestation=runtime_attestation,
    )
    torch = __import__("torch")
    versions = _require_versions(torch)
    torch.manual_seed(20260915)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260915)
    model = GPT2Runtime(
        GPT2Spec(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            model_root=args.model_root,
            layer=6,
            hook=HOOK,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype="float32",
        )
    )
    cfg_path = next(args.feature_root.rglob("cfg.json"), None)
    weights_path = next(args.feature_root.rglob("sae_weights.safetensors"), None)
    if cfg_path is None or weights_path is None:
        raise RuntimeError("fresh residual-mid SAE custody is missing cfg.json or sae_weights.safetensors")
    sae = AffineSAE(weights_path, _read(cfg_path))
    started = time.monotonic()
    fit = ROWS[:6]
    tune = ROWS[6:12]
    assessment = ROWS[12:]
    baseline = model.logits(ROWS)
    captured = model.capture(ROWS)
    hooked_logits = tuple(tuple(float(v) for v in model._run(row, capture=True)[0][0]) for row in ROWS)
    parity = max(max_abs_delta(left, right) for left, right in zip(baseline, hooked_logits))
    repeat_captured = model.capture(ROWS)
    repeat_delta = max(max_abs_delta(left, right) for left, right in zip(captured, repeat_captured))
    fit_features = tuple(sae.encode(vector) for vector in captured[:6])
    selected = _feature_index(fit_features)
    original = captured[0]
    no_op_logits = model.patched_logits(fit[0], original)
    no_op_delta = _effect(baseline[0], no_op_logits)
    active_value = fit_features[0][selected]
    intervened = sae.intervene(original, selected, active_value + abs(active_value) + 1.0)
    intervention_effect = _effect(baseline[0], model.patched_logits(fit[0], intervened))
    reconstruction_targets = tuple(captured)
    reconstruction_predictions = tuple(sae.reconstruct(vector) for vector in captured)
    reconstruction_nmse = _pooled_centered_nmse(reconstruction_targets, reconstruction_predictions)
    repeat_feature_cosine = min(cosine(sae.encode(a), sae.encode(b)) for a, b in zip(captured, repeat_captured))
    scrub_effects = []
    for index, row in enumerate(assessment, start=12):
        scrubbed = sae.intervene(captured[index], selected, 0.0)
        scrub_effects.append(_effect(baseline[index], model.patched_logits(row, scrubbed)))
    limits = config.get("qualification", {})
    gates = [
        _gate("native_instrumented_parity", parity <= float(limits["parity_abs_tol"]), max_abs_logit_delta=parity),
        _gate("deterministic_repeatability", repeat_delta <= float(limits["repeatability_abs_tol"]), max_abs_activation_delta=repeat_delta),
        _gate("exact_no_op", no_op_delta <= float(limits["no_op_abs_tol"]), max_abs_logit_delta=no_op_delta),
        _gate("nonzero_intervention_reach", intervention_effect >= float(limits["intervention_min_abs_effect"]), max_abs_logit_effect=intervention_effect),
        _gate("feature_reconstruction", reconstruction_nmse <= float(limits["reconstruction_nmse_max"]), pooled_centered_nmse=reconstruction_nmse),
        _gate("feature_repeat_stability", repeat_feature_cosine >= float(limits["stability_min_cosine"]), minimum_cosine=repeat_feature_cosine),
        _gate("heldout_causal_scrubbing", max(scrub_effects) >= float(limits["scrub_min_abs_effect"]), maximum_abs_logit_effect=max(scrub_effects)),
    ]
    custody = file_manifest(args.custody_root)
    result = {
        "schema": "interp-lab.real-qualification-report.v2",
        "state_slice": STATE_SLICE,
        "continuation_identity": CONTINUATION_ID,
        "claim_status": "LOCAL_INTERNAL_REAL_MODEL_CONTINUATION_QUALIFICATION_ONLY",
        "claim_ceiling": "LOCAL_INTERNAL_EXPLORATORY_ONLY",
        "packet_digest": packet_digest,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "feature_asset": {"repository": FEATURE_REPO, "revision": FEATURE_REVISION, "weights_sha256": _sha256(weights_path), "hook": HOOK},
        "corpus": {"rows": len(ROWS), "fit": len(fit), "tune": len(tune), "assessment": len(assessment), "digest": _corpus_digest()},
        "estimand": {"reconstruction": "pooled_global_centered_nmse", "denominator": "sum_squared_centered_targets"},
        "runtime": versions,
        "device": {"cuda_available": bool(torch.cuda.is_available()), "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
        "canonical_path": "transformer.h[6].ln_2.input",
        "selected_feature": selected,
        "gates": gates,
        "gate_order": tuple(gate["name"] for gate in gates),
        "all_gates_pass": all(gate["passed"] for gate in gates),
        "aggregate": {"fit_intervention_max_abs_logit_effect": intervention_effect, "assessment_max_abs_scrub_effect": max(scrub_effects), "tune_selected_feature_active_rows": sum(sae.encode(vector)[selected] > 0.0 for vector in captured[6:12])},
        "custody_manifest_digest": custody["manifest_digest"],
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "nonclaims": ("no independent review or proof", "no cross-checkpoint stability claim", "no continual-learning retention claim", "no alignment or production claim"),
    }
    result["report_digest"] = digest(result)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result_path = os.environ.get("GMN_RESULT_PATH")
    if result_path:
        Path(result_path).write_text(json.dumps({"status": "completed", "claim_status": result["claim_status"], "all_gates_pass": result["all_gates_pass"], "report_digest": result["report_digest"]}) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
