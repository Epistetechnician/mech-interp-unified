#!/usr/bin/env python3
"""Run the packet-bound V5 Qwen3.8 reconstruction qualification.

State slice: interpretability-lab-sandbox-v1.

The runner is offline and aggregate-only.  It selects one of the two
predeclared layer-32 hooks on fit data, then locks that choice before tune and
assessment.  It refuses to start before custody, packet, authorization, and
runtime checks pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from interp_lab.canonical import digest  # noqa: E402
from interp_lab.custody import file_manifest, require_internal_custody_binding  # noqa: E402
from interp_lab.execution import require_internal_execution  # noqa: E402
from interp_lab.metrics import cosine, max_abs_delta  # noqa: E402
from interp_lab.real_adapters import Qwen38Runtime, Qwen38Spec, Qwen38TopKSAE  # noqa: E402
from interp_lab.reconstruction import summarize_reconstruction  # noqa: E402


STATE_SLICE = "interpretability-lab-sandbox-v1"
ROW_FIELDS = {"row_id", "split", "family", "document_id", "text", "target_token_id", "contrast_token_id", "donor_row_id", "shuffled_donor_row_id"}
ADAPTER_HOOKS = {"decoder_input": "decoder_in", "decoder_output": "decoder_out"}


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON object is unreadable: {path}") from exc
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


def _runtime(torch: Any, expected: Mapping[str, Any]) -> dict[str, str]:
    try:
        import safetensors  # type: ignore
        import transformers  # type: ignore
    except ImportError as exc:  # pragma: no cover - remote runner only
        raise RuntimeError("V5 runtime dependencies are missing") from exc
    actual = {
        "python": platform.python_version(),
        "torch": str(torch.__version__).split("+")[0],
        "transformers": str(transformers.__version__),
        "safetensors": str(safetensors.__version__),
    }
    if expected.get("torchvision") is not None or expected.get("pillow") is not None:
        try:
            import torchvision  # type: ignore
            from PIL import __version__ as pillow_version  # type: ignore
        except ImportError as exc:  # pragma: no cover - remote runner only
            raise RuntimeError("V7 vision runtime dependencies are missing") from exc
        actual["torchvision"] = str(torchvision.__version__).split("+")[0]
        actual["pillow"] = str(pillow_version)
    for key, value in actual.items():
        if expected.get(key) != value:
            raise RuntimeError(f"V5 runtime mismatch for {key}: expected {expected.get(key)}, observed {value}")
    if torch.version.cuda != expected.get("cuda") or not torch.cuda.is_available():
        raise RuntimeError("V5 qualification requires the attested CUDA runtime")
    if expected.get("device_model") not in str(torch.cuda.get_device_name(0)):
        raise RuntimeError("V5 qualification requires an NVIDIA H100")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return actual


def _corpus(corpus: Mapping[str, Any], expected_splits: Mapping[str, Any]) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    if corpus.get("schema") != "interp-lab.assessment-corpus.v1" or corpus.get("fresh") is not True:
        raise RuntimeError("V5 corpus identity or schema mismatch")
    rows = corpus.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("V5 corpus rows are missing")
    split_rows: dict[str, list[str]] = {"fit": [], "tune": [], "assessment": []}
    by_id: dict[str, dict[str, Any]] = {}
    families = {key: set() for key in split_rows}
    documents = {key: set() for key in split_rows}
    for row in rows:
        if not isinstance(row, dict) or set(row) != ROW_FIELDS:
            raise RuntimeError("V5 corpus row schema mismatch")
        row_id, split, family, document = str(row["row_id"]), str(row["split"]), str(row["family"]), str(row["document_id"])
        if split not in split_rows or not row_id or row_id in by_id or not family or not document or not str(row["text"]).strip():
            raise RuntimeError("V5 corpus roster is invalid")
        if family in families[split] or document in documents[split]:
            raise RuntimeError("V5 corpus repeats a family or document")
        split_rows[split].append(row_id)
        families[split].add(family)
        documents[split].add(document)
        by_id[row_id] = row
    if any(families[left] & families[right] or documents[left] & documents[right] for left, right in (("fit", "tune"), ("fit", "assessment"), ("tune", "assessment"))):
        raise RuntimeError("V5 corpus family/document disjointness is violated")
    counts = {key: len(value) for key, value in split_rows.items()}
    expected = {key: int(expected_splits[key]) for key in split_rows}
    if counts != expected:
        raise RuntimeError(f"V5 corpus split counts differ: expected {expected}, observed {counts}")
    return split_rows, by_id


def _margin(logits: Sequence[float], target: int, contrast: int) -> float:
    if not 0 <= target < len(logits) or not 0 <= contrast < len(logits):
        raise RuntimeError("target or contrast token is outside the vocabulary")
    return float(logits[target] - logits[contrast])


def _feature_index(features: Sequence[Sequence[float]]) -> int:
    counts: dict[int, int] = {}
    for vector in features:
        for index, value in enumerate(vector):
            if value > 0.0:
                counts[index] = counts.get(index, 0) + 1
    if not counts:
        raise RuntimeError("no active V5 feature in fit rows")
    return min(counts, key=lambda index: (-counts[index], index))


def _all_token_summary(torch: Any, sae: Qwen38TopKSAE, runtime: Qwen38Runtime, texts: Sequence[str]) -> tuple[Any, int]:
    rows = runtime.capture_all(texts)
    flat = [token for row in rows for token in row]
    if not flat:
        raise RuntimeError("V5 all-token capture is empty")
    activations = torch.as_tensor(flat, dtype=torch.float32, device=sae.device)
    reconstructed = sae.reconstruct_batch(activations).detach().cpu().tolist()
    return summarize_reconstruction(flat, reconstructed), len(flat)


def _gate(name: str, passed: bool, **values: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), **values}


def run(*, config_path: Path, packet_path: Path, authorization_path: Path, runtime_attestation_path: Path, custody_root: Path, corpus_path: Path, model_root: Path, sae_weights_path: Path, result_path: Path) -> dict[str, Any]:
    config = _read(config_path)
    packet = _read(packet_path)
    authorization = _read(authorization_path)
    attestation = _read(runtime_attestation_path)
    packet_digest = packet.get("packet_digest")
    config_packet = config.get("packet", {}).get("digest") if isinstance(config.get("packet"), Mapping) else config.get("packet_digest")
    if config_packet != packet_digest:
        raise RuntimeError("V5 config is not bound to the packet")
    require_internal_execution(config=config, packet=packet, packet_digest=packet_digest, custody_root=custody_root, authorization=authorization, runtime_attestation=attestation)
    paths = (corpus_path, model_root, sae_weights_path, authorization_path, runtime_attestation_path, result_path)
    if any(not _inside(custody_root, path) for path in paths):
        raise RuntimeError("V5 execution inputs and output must be below custody root")
    custody_binding = require_internal_custody_binding(root=custody_root, packet=packet, packet_digest=str(packet_digest), model_root=model_root, sae_weights=sae_weights_path, corpus=corpus_path, authorization=authorization_path, runtime_attestation=runtime_attestation_path)
    torch = __import__("torch")
    versions = _runtime(torch, dict(packet["runtime"]))
    torch.manual_seed(20260915)
    torch.cuda.manual_seed_all(20260915)
    corpus = _read(corpus_path)
    split_rows, by_id = _corpus(corpus, packet["corpus"]["splits"])
    asset = packet["feature_asset"]
    model = packet["model"]
    candidates = packet["hook_selection"]["candidates"]
    hook_layer = int(packet["hook_selection"].get("layer", model.get("canonical_layer", 32)))
    if hook_layer != int(model.get("canonical_layer", hook_layer)):
        raise RuntimeError("V5 packet canonical layer binding is inconsistent")
    started = time.monotonic()
    selected_runtime: Qwen38Runtime | None = None
    selected_hook: str | None = None
    selected_fit_summary: Any | None = None
    selected_fit_tokens = 0
    for candidate in candidates:
        hook = str(candidate["id"])
        adapter_hook = ADAPTER_HOOKS.get(hook)
        if adapter_hook is None:
            raise RuntimeError(f"V5 packet hook id is not registered: {hook}")
        runtime = Qwen38Runtime(Qwen38Spec(model_id=str(model["id"]), revision=str(model["revision"]), model_root=model_root, layer=hook_layer, hook=adapter_hook, device="cuda", dtype=str(packet["runtime"]["dtype"])))
        sae = Qwen38TopKSAE(sae_weights_path, d_in=int(asset["d_in"]), d_sae=int(asset["d_sae"]), top_k=int(asset["top_k"]), device="cuda")
        fit_texts = tuple(by_id[row_id]["text"] for row_id in split_rows["fit"])
        fit_summary, fit_tokens = _all_token_summary(torch, sae, runtime, fit_texts)
        if math.isfinite(fit_summary.pooled_fvu) and fit_summary.pooled_fvu <= float(packet["reconstruction"]["fit_pooled_fvu_max"]) and fit_summary.max_row_nmse <= float(packet["reconstruction"]["fit_max_row_nmse"]):
            selected_runtime, selected_hook, selected_fit_summary, selected_fit_tokens = runtime, hook, fit_summary, fit_tokens
            break
        del sae, runtime
        torch.cuda.empty_cache()
    if selected_runtime is None or selected_hook is None or selected_fit_summary is None:
        raise RuntimeError("no predeclared V5 hook passed the fit reconstruction gate")
    runtime = selected_runtime
    sae = Qwen38TopKSAE(sae_weights_path, d_in=int(asset["d_in"]), d_sae=int(asset["d_sae"]), top_k=int(asset["top_k"]), device="cuda")
    all_ids = tuple(split_rows["fit"] + split_rows["tune"] + split_rows["assessment"])
    texts = tuple(by_id[row_id]["text"] for row_id in all_ids)
    baseline = runtime.logits(texts)
    captured = runtime.capture(texts)
    hooked = tuple(tuple(float(v) for v in runtime._run(text, capture=True)[0][0]) for text in texts)
    parity = max(max_abs_delta(left, right) for left, right in zip(baseline, hooked))
    repeated = runtime.capture(texts)
    repeat_delta = max(max_abs_delta(left, right) for left, right in zip(captured, repeated))
    features = tuple(sae.encode(vector) for vector in captured)
    selected_feature = _feature_index(features[: len(split_rows["fit"])])
    no_op_delta = max_abs_delta(baseline[0], runtime.patched_logits(texts[0], captured[0]))
    first = by_id[all_ids[0]]
    active_value = features[0][selected_feature]
    intervention = sae.intervene(captured[0], selected_feature, active_value + abs(active_value) + 1.0)
    intervention_effect = _margin(runtime.patched_logits(texts[0], intervention), int(first["target_token_id"]), int(first["contrast_token_id"])) - _margin(baseline[0], int(first["target_token_id"]), int(first["contrast_token_id"]))
    tune_texts = tuple(by_id[row_id]["text"] for row_id in split_rows["tune"])
    assessment_texts = tuple(by_id[row_id]["text"] for row_id in split_rows["assessment"])
    tune_summary, tune_tokens = _all_token_summary(torch, sae, runtime, tune_texts)
    assessment_summary, assessment_tokens = _all_token_summary(torch, sae, runtime, assessment_texts)
    feature_cosine = min(cosine(sae.encode(a), sae.encode(b)) for a, b in zip(captured, repeated))
    scrub_effects = []
    assessment_offset = len(split_rows["fit"]) + len(split_rows["tune"])
    for index, row_id in enumerate(split_rows["assessment"], start=assessment_offset):
        metadata = by_id[row_id]
        scrubbed = sae.intervene(captured[index], selected_feature, 0.0)
        scrub_effects.append(_margin(runtime.patched_logits(metadata["text"], scrubbed), int(metadata["target_token_id"]), int(metadata["contrast_token_id"])) - _margin(baseline[index], int(metadata["target_token_id"]), int(metadata["contrast_token_id"])))
    reconstruction_limit = float(packet["reconstruction"]["assessment_pooled_fvu_max"])
    row_limit = float(packet["reconstruction"]["assessment_max_row_nmse"])
    gates = [
        _gate("native_instrumented_parity", parity == 0.0, max_abs_logit_delta=parity),
        _gate("deterministic_repeatability", repeat_delta == 0.0, max_abs_activation_delta=repeat_delta),
        _gate("exact_no_op", no_op_delta == 0.0, max_abs_logit_delta=no_op_delta),
        _gate("nonzero_intervention_reach", abs(intervention_effect) > 0.0, max_abs_logit_effect=abs(intervention_effect)),
        _gate("fit_reconstruction", selected_fit_summary.pooled_fvu <= float(packet["reconstruction"]["fit_pooled_fvu_max"]) and selected_fit_summary.max_row_nmse <= float(packet["reconstruction"]["fit_max_row_nmse"]), **selected_fit_summary.as_dict()),
        _gate("tune_reconstruction", tune_summary.pooled_fvu <= float(packet["reconstruction"]["tune_pooled_fvu_max"]) and tune_summary.max_row_nmse <= float(packet["reconstruction"]["tune_max_row_nmse"]), **tune_summary.as_dict()),
        _gate("assessment_reconstruction", assessment_summary.pooled_fvu <= reconstruction_limit and assessment_summary.max_row_nmse <= row_limit, **assessment_summary.as_dict()),
        _gate("feature_repeat_stability", feature_cosine >= 0.8, minimum_cosine=feature_cosine),
        _gate("causal_scrubbing_reach", max(abs(value) for value in scrub_effects) > 0.0, maximum_abs_logit_effect=max(abs(value) for value in scrub_effects)),
    ]
    custody = file_manifest(custody_root)
    result: dict[str, Any] = {
        "schema": "interp-lab.real-qwen38-reconstruction-v6-report.v1" if packet.get("protocol") == "qwen38-reconstruction-v6" else "interp-lab.real-qwen38-reconstruction-v5-report.v1",
        "state_slice": STATE_SLICE,
        "claim_status": "LOCAL_INTERNAL_REAL_MODEL_QUALIFICATION_ONLY",
        "claim_ceiling": "LOCAL_INTERNAL_EXPLORATORY_ONLY",
        "packet_digest": packet_digest,
        "model": {"id": model["id"], "revision": model["revision"], "canonical_layer": hook_layer, "canonical_path": f"model.language_model.layers[{hook_layer}]", "hook": selected_hook},
        "feature_asset": {"path": asset["path"], "weights_sha256": _sha256(sae_weights_path)},
        "corpus": {"rows": len(all_ids), "split_counts": {key: len(value) for key, value in split_rows.items()}, "sha256": _sha256(corpus_path)},
        "runtime": versions,
        "device": {"cuda_available": True, "name": torch.cuda.get_device_name(0)},
        "selected_feature": selected_feature,
        "token_counts": {"fit": selected_fit_tokens, "tune": tune_tokens, "assessment": assessment_tokens},
        "gates": gates,
        "gate_order": tuple(gate["name"] for gate in gates),
        "all_gates_pass": all(gate["passed"] for gate in gates),
        "aggregate": {"intervention_effect": intervention_effect, "fit_pooled_fvu": selected_fit_summary.pooled_fvu, "tune_pooled_fvu": tune_summary.pooled_fvu, "assessment_pooled_fvu": assessment_summary.pooled_fvu, "scrub_mean_abs_effect": sum(abs(value) for value in scrub_effects) / len(scrub_effects)},
        "custody_manifest_digest": custody["manifest_digest"],
        "input_binding_manifest_digest": custody_binding["manifest"]["manifest_digest"],
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "nonclaims": ("no independent review or proof", "no held-out scientific assessment claim", "no continual-learning retention claim", "no alignment or production claim"),
    }
    result["report_digest"] = digest(result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(result_path, 0o600)
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
    result = run(config_path=args.config, packet_path=args.packet, authorization_path=args.authorization, runtime_attestation_path=args.runtime_attestation, custody_root=args.custody_root, corpus_path=args.corpus, model_root=args.model_root, sae_weights_path=args.sae_weights, result_path=args.result)
    return 0 if result["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
