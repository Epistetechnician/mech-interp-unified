"""Run the packet-bound held-out real-model assessment after every gate opens.

State slice: interpretability-lab-sandbox-v1.

The script is intentionally fail-closed. It requires a custody-ready packet,
owner-only custody, a reviewer-controlled Ed25519 ACCEPT, and already-acquired
model/feature/corpus files. It never downloads inputs or changes packet fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

# Keep direct ``python scripts/run_assessment.py`` invocation hermetic from a
# clean checkout, matching the other repository runners.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from interp_lab.assessment import AssessmentPanel, PredictionLock, evaluate_heldout_effects  # noqa: E402
from interp_lab.custody import file_manifest  # noqa: E402
from interp_lab.real_adapters import AffineSAE, GPT2Runtime, GPT2Spec  # noqa: E402
from scripts.check_assessment_gate import check_packet_gate  # noqa: E402


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


def _require_runtime(expected: dict[str, Any]) -> None:
    try:
        import safetensors  # type: ignore
        import torch  # type: ignore
        import transformers  # type: ignore
    except ImportError as error:  # pragma: no cover - real runner only
        raise RuntimeError("pinned assessment runtime dependencies are missing") from error
    actual = {
        "python": platform.python_version(),
        "torch": str(torch.__version__).split("+")[0],
        "transformers": str(transformers.__version__),
        "safetensors": str(safetensors.__version__),
    }
    for key, value in actual.items():
        if expected.get(key) != value:
            raise RuntimeError(f"assessment runtime mismatch for {key}: expected {expected.get(key)}, observed {value}")
    if expected.get("cuda") and torch.version.cuda != expected["cuda"]:
        raise RuntimeError(f"assessment CUDA mismatch: expected {expected['cuda']}, observed {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("assessment requires the attested CUDA device")
    device_name = str(torch.cuda.get_device_name(0))
    if expected.get("device_model") not in device_name:
        raise RuntimeError(f"assessment GPU mismatch: expected {expected.get('device_model')}, observed {device_name}")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def _margin(logits: tuple[float, ...], target_token: int, contrast_token: int) -> float:
    if not 0 <= target_token < len(logits) or not 0 <= contrast_token < len(logits):
        raise ValueError("target or contrast token is outside the model vocabulary")
    return float(logits[target_token] - logits[contrast_token])


def _rows(corpus: Any) -> tuple[AssessmentPanel, dict[str, dict[str, Any]]]:
    if not isinstance(corpus, dict) or corpus.get("schema") != "interp-lab.assessment-corpus.v1":
        raise ValueError("assessment corpus schema mismatch")
    rows = corpus.get("rows")
    if not isinstance(rows, list):
        raise ValueError("assessment corpus rows are missing")
    split_rows: dict[str, list[str]] = {"fit": [], "tune": [], "assessment": []}
    by_id: dict[str, dict[str, Any]] = {}
    families: dict[str, set[str]] = {"fit": set(), "tune": set(), "assessment": set()}
    documents: dict[str, set[str]] = {"fit": set(), "tune": set(), "assessment": set()}
    expected_keys = {"row_id", "split", "family", "document_id", "text", "target_token_id", "contrast_token_id", "donor_row_id", "shuffled_donor_row_id"}
    for raw in rows:
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise ValueError("assessment corpus row schema mismatch")
        row_id = str(raw["row_id"])
        split = str(raw["split"])
        family = str(raw["family"])
        document = str(raw["document_id"])
        if split not in split_rows or row_id in by_id or not family or not document or not str(raw["text"]).strip():
            raise ValueError("assessment corpus roster is invalid")
        if family in families[split] or document in documents[split]:
            raise ValueError("assessment corpus repeats a family or document within a split")
        split_rows[split].append(row_id)
        families[split].add(family)
        documents[split].add(document)
        by_id[row_id] = raw
    if any(families[left].intersection(families[right]) or documents[left].intersection(documents[right]) for left, right in (("fit", "tune"), ("fit", "assessment"), ("tune", "assessment"))):
        raise ValueError("assessment corpus family/document disjointness is violated")
    return AssessmentPanel(split_rows), by_id


def run(*, packet_path: Path, corpus_path: Path, base_model_root: Path, update_model_root: Path, sae_config_path: Path, sae_weights_path: Path, custody_root: Path, review_receipt_path: Path) -> dict[str, Any]:
    packet = _load(packet_path)
    receipt = _load(review_receipt_path)
    gate = check_packet_gate(packet, custody_root=custody_root, review_receipt=receipt)
    if gate["status"] != "READY_FOR_EXTERNAL_ASSESSMENT":
        raise RuntimeError(json.dumps(gate, sort_keys=True))
    required_paths = (corpus_path, base_model_root, update_model_root, sae_config_path, sae_weights_path, review_receipt_path)
    if any(not _inside(custody_root, path) for path in required_paths):
        raise RuntimeError("all assessment inputs and the external receipt must be below the owner-only custody root")
    if _sha256(corpus_path) != packet["corpus"]["sha256"]:
        raise RuntimeError("assessment corpus digest does not match the packet")
    checkpoints = packet["model"]["checkpoints"]
    for label, model_root in (("base", base_model_root), ("post_update_1", update_model_root)):
        try:
            checkpoint_manifest = file_manifest(model_root)
        except Exception as error:
            raise RuntimeError(f"{label} checkpoint custody manifest is invalid") from error
        if checkpoint_manifest["manifest_digest"] != checkpoints[label]["sha256"]:
            raise RuntimeError(f"{label} checkpoint manifest does not match the packet")
    feature = packet["feature_asset"]
    if _sha256(sae_config_path) != feature["config_sha256"] or _sha256(sae_weights_path) != feature["weights_sha256"]:
        raise RuntimeError("feature asset file digest does not match the packet")
    _require_runtime(dict(packet["runtime"]))
    corpus = _load(corpus_path)
    panel, by_id = _rows(corpus)
    expected_counts = packet["corpus"]["splits"]
    if {split: len(panel.rows[split]) for split in panel.rows} != expected_counts:
        raise ValueError("corpus split counts do not match packet")
    lock_payload = _load(custody_root / "prediction-lock.json")
    if not isinstance(lock_payload, dict) or set(lock_payload) != {"schema", "state_slice", "feature_index", "expected_sign", "source"}:
        raise ValueError("prediction lock schema mismatch")
    lock = PredictionLock(feature_index=int(lock_payload["feature_index"]), expected_sign=int(lock_payload["expected_sign"]), source=str(lock_payload["source"]))
    if lock_payload != lock.as_dict():
        raise ValueError("prediction lock payload is not canonical")
    if packet["selection"]["feature_index"] != lock.feature_index or packet["prediction_lock"]["digest"] != lock.digest:
        raise ValueError("prediction lock is not packet-bound")
    model = packet["model"]
    feature = packet["feature_asset"]
    if str(feature["hook"]) != "transformer.h[6].ln_2.input":
        raise ValueError("assessment packet does not bind the canonical residual-mid hook")
    # The packet's fully qualified module path maps to the adapter's explicit
    # hook enum.  Never derive this by taking the final path component
    # (``input`` is a tensor position, not an adapter hook).
    spec_base = GPT2Spec(model_id=str(model["id"]), revision=str(model["revision"]), model_root=base_model_root, layer=6, hook="resid_mid", device="cuda", dtype="float32")
    runtime_base = GPT2Runtime(spec_base)
    # The post-update checkpoint is custody-gated for the subsequent matched
    # continual-learning lane.  This held-out causal pass is defined on the
    # frozen base checkpoint, so do not load a second model or silently use it
    # in the estimand.
    sae = AffineSAE(sae_weights_path, _load(sae_config_path))
    assessment_rows = panel.assessment_rows
    baseline_logits = {row: runtime_base.logits((by_id[row]["text"],))[0] for row in assessment_rows}
    baseline_activation = {row: runtime_base.capture((by_id[row]["text"],))[0] for row in assessment_rows}
    donor_ids = {row: str(by_id[row]["donor_row_id"]) for row in assessment_rows}
    shuffled_donor_ids = {row: str(by_id[row]["shuffled_donor_row_id"]) for row in assessment_rows}
    for donor_id in (*donor_ids.values(), *shuffled_donor_ids.values()):
        if donor_id not in by_id or donor_id in assessment_rows or donor_id not in panel.rows["fit"] + panel.rows["tune"]:
            raise ValueError("held-out donors must be present in fit/tune and absent from assessment")
    all_rows = tuple(by_id)
    donor_activation = {row: runtime_base.capture((by_id[row]["text"],))[0] for row in all_rows}
    effects: dict[str, dict[str, float]] = {}
    for row in assessment_rows:
        metadata = by_id[row]
        target = int(metadata["target_token_id"])
        contrast = int(metadata["contrast_token_id"])
        original = baseline_activation[row]
        donor = donor_activation[donor_ids[row]]
        source_features = sae.encode(original)
        donor_features = sae.encode(donor)
        base_margin = _margin(baseline_logits[row], target, contrast)

        def effect(activation: tuple[float, ...]) -> float:
            patched = runtime_base.patched_logits(metadata["text"], activation)
            return _margin(patched, target, contrast) - base_margin

        changed = sae.intervene(original, lock.feature_index, donor_features[lock.feature_index])
        scrubbed = sae.intervene(original, lock.feature_index, 0.0)
        constant = sae.intervene(original, lock.feature_index, 0.125)
        exact = sae.intervene(original, lock.feature_index, source_features[lock.feature_index])
        shuffled_id = shuffled_donor_ids[row]
        shuffled_features = sae.encode(donor_activation[shuffled_id])
        shuffled = sae.intervene(original, lock.feature_index, shuffled_features[lock.feature_index])
        matched = changed
        text_only_logits = runtime_base.logits((by_id[donor_ids[row]]["text"],))[0]
        effects[row] = {
            "intervention": effect(changed),
            "scrub": effect(scrubbed),
            "exact_copy": effect(exact),
            "activation_only": effect(changed),
            "text_only": _margin(text_only_logits, target, contrast) - base_margin,
            "shuffled": effect(shuffled),
            "constant": effect(constant),
            "matched": effect(matched),
        }
    report = evaluate_heldout_effects(panel=panel, effects=effects, prediction_lock=lock, thresholds=packet["thresholds"], statistics=packet["statistics"], claim_status=str(packet["claim_ceiling"]))
    if not report["all_gates_pass"]:
        raise RuntimeError(json.dumps(report, sort_keys=True))
    (custody_root / "assessment-effects.json").write_text(json.dumps({"schema": "interp-lab.assessment-effects.v1", "state_slice": "interpretability-lab-sandbox-v1", "packet_digest": packet["packet_digest"], "effects": effects}, sort_keys=True) + "\n", encoding="utf-8")
    (custody_root / "assessment-report.json").write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    manifest = file_manifest(custody_root)
    record = {"schema": "interp-lab.assessment-run-record.v1", "state_slice": "interpretability-lab-sandbox-v1", "packet_digest": packet["packet_digest"], "report_digest": report["report_digest"], "custody_manifest_digest": manifest["manifest_digest"], "raw_trace_retention_hours": 72}
    (custody_root / "run-record.json").write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "ASSESSMENT_COMPLETED_PENDING_INDEPENDENT_VALIDATION", "packet_digest": packet["packet_digest"], "report_digest": report["report_digest"], "custody_manifest_digest": manifest["manifest_digest"], "all_gates_pass": report["all_gates_pass"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, default=Path("configs/gpt2-resid-mid-assessment-v1.json"))
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--base-model-root", type=Path, required=True)
    parser.add_argument("--update-model-root", type=Path, required=True)
    parser.add_argument("--sae-config", type=Path, required=True)
    parser.add_argument("--sae-weights", type=Path, required=True)
    parser.add_argument("--custody-root", type=Path, required=True)
    parser.add_argument("--review-receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(packet_path=args.packet, corpus_path=args.corpus, base_model_root=args.base_model_root, update_model_root=args.update_model_root, sae_config_path=args.sae_config, sae_weights_path=args.sae_weights, custody_root=args.custody_root, review_receipt_path=args.review_receipt), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
