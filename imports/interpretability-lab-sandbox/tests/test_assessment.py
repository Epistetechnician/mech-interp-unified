"""Held-out assessment and matched continual-learning tests.

State slice: interpretability-lab-sandbox-v1.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.assemble_assessment_packet import assemble
from interp_lab.assessment import AssessmentError, AssessmentPanel, PredictionLock, evaluate_heldout_effects
from interp_lab.assessment_validator import AssessmentValidationError, validate_heldout_report
from interp_lab.canonical import digest
from interp_lab.continual import run_matched_continual_assessment
from interp_lab.continual_validator import ContinualValidationError, evaluate_v10_controller_success, validate_matched_continual_report


class AssessmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.panel = AssessmentPanel(
            {
                "fit": ("fresh-fit-a", "fresh-fit-b"),
                "tune": ("fresh-tune-a", "fresh-tune-b"),
                "assessment": ("fresh-assessment-a", "fresh-assessment-b"),
            }
        )
        self.lock = PredictionLock(feature_index=0, expected_sign=1)
        self.effects = {
            row: {
                "intervention": 0.2,
                "scrub": -0.3,
                "exact_copy": 0.0,
                "activation_only": 0.2,
                "text_only": 0.01,
                "shuffled": 0.02,
                "constant": 0.03,
                "matched": 0.04,
            }
            for row in self.panel.assessment_rows
        }

    def test_heldout_report_and_independent_recompute(self):
        report = evaluate_heldout_effects(panel=self.panel, effects=self.effects, prediction_lock=self.lock, thresholds={})
        self.assertTrue(report["all_gates_pass"])
        self.assertEqual(report["claim_status"], "LOCAL_SYNTHETIC_ONLY")
        validation = validate_heldout_report(report, panel=self.panel, effects=self.effects, prediction_lock=self.lock)
        self.assertEqual(validation["status"], "INDEPENDENT_AGGREGATE_VALIDATED")
        self.assertEqual(validation["report_digest"], report["report_digest"])

    def test_validator_rejects_tampered_aggregate(self):
        report = evaluate_heldout_effects(panel=self.panel, effects=self.effects, prediction_lock=self.lock, thresholds={})
        tampered = {**report, "aggregate": {**report["aggregate"], "intervention_mean_abs": 999.0}}
        with self.assertRaisesRegex(AssessmentValidationError, "report digest mismatch"):
            validate_heldout_report(tampered, panel=self.panel, effects=self.effects, prediction_lock=self.lock)

    def test_panel_rejects_overlap(self):
        with self.assertRaisesRegex(AssessmentError, "disjoint"):
            AssessmentPanel({"fit": ("same",), "tune": ("same",), "assessment": ("other",)})

    def test_gate_is_closed_without_external_inputs(self):
        root = Path(__file__).resolve().parents[1]
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(root / "src")}
        result = subprocess.run(
            [sys.executable, str(root / "scripts" / "check_assessment_gate.py")],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "CLOSED")
        self.assertIn("fresh corpus digest is missing", payload["reasons"])
        self.assertIn("prediction lock digest is missing", payload["reasons"])
        self.assertIn("independent packet-bound Ed25519 ACCEPT receipt is missing", payload["reasons"])

    def test_validator_recomputes_gate_results(self):
        report = evaluate_heldout_effects(panel=self.panel, effects=self.effects, prediction_lock=self.lock, thresholds={})
        gates = [{**gate} for gate in report["gates"]]
        gates[2]["passed"] = False
        body = {**report, "gates": gates}
        body["report_digest"] = digest({key: value for key, value in body.items() if key != "report_digest"})
        with self.assertRaisesRegex(AssessmentValidationError, "gate result mismatch"):
            validate_heldout_report(body, panel=self.panel, effects=self.effects, prediction_lock=self.lock)

    def test_matched_continual_accounting(self):
        report = run_matched_continual_assessment()
        self.assertTrue(report["matched_data"])
        self.assertTrue(report["matched_update_count"])
        self.assertTrue(report["matched_total_compute"])
        self.assertEqual(set(report["controllers"]), {"sgd", "ewc", "behavioral_only", "causal_monitor"})
        self.assertEqual({row["total_compute_units"] for row in report["controllers"].values()}, {report["total_compute_budget"]})
        self.assertEqual({row["updates"] for row in report["controllers"].values()}, {report["stream_length"]})

    def test_continual_validator_recomputes_accounting(self):
        report = run_matched_continual_assessment()
        validated = validate_matched_continual_report(report)
        self.assertEqual(validated["status"], "INDEPENDENT_CONTINUAL_AGGREGATE_VALIDATED")
        tampered = {**report, "controllers": {**report["controllers"], "sgd": {**report["controllers"]["sgd"], "retention": 2.0}}}
        tampered["report_digest"] = digest({key: value for key, value in tampered.items() if key != "report_digest"})
        with self.assertRaisesRegex(ContinualValidationError, "summary does not match|accuracy is outside|controller"):
            validate_matched_continual_report(tampered)

    def test_v10_controller_rule_is_falsifiable(self):
        report = run_matched_continual_assessment()
        decision = evaluate_v10_controller_success(report)
        self.assertEqual(decision["status"], "NO_CANDIDATE")
        self.assertFalse(decision["comparisons"]["ewc"]["retention_superiority"])

    def test_custody_intake_assembles_new_packet_without_signing(self):
        root = Path(__file__).resolve().parents[1]
        design_path = root / "configs" / "gpt2-resid-mid-assessment-v1.json"
        with tempfile.TemporaryDirectory() as directory:
            custody = Path(directory)
            os.chmod(custody, 0o700)
            corpus_path = custody / "corpus.json"
            rows = []
            for split in ("fit", "tune", "assessment"):
                for index in range(48):
                    row_id = f"{split}-{index:02d}"
                    rows.append(
                        {
                            "row_id": row_id,
                            "split": split,
                            "family": f"{split}-family-{index:02d}",
                            "document_id": f"{split}-document-{index:02d}",
                            "text": f"Fresh {split} row {index}.",
                            "target_token_id": 1,
                            "contrast_token_id": 2,
                            "donor_row_id": f"fit-{index:02d}",
                            "shuffled_donor_row_id": f"tune-{index:02d}",
                        }
                    )
            corpus_path.write_text(json.dumps({"schema": "interp-lab.assessment-corpus.v1", "rows": rows}), encoding="utf-8")
            prediction_lock_path = custody / "prediction-lock.json"
            lock = PredictionLock(feature_index=13637, expected_sign=1)
            prediction_lock_path.write_text(json.dumps(lock.as_dict()), encoding="utf-8")
            base_root = custody / "base-model"
            update_root = custody / "post-update-model"
            for model_root in (base_root, update_root):
                model_root.mkdir()
                os.chmod(model_root, 0o700)
                (model_root / "weights.bin").write_bytes(b"synthetic checkpoint")
            sae_root = custody / "sae"
            sae_root.mkdir()
            sae_config_path = sae_root / "cfg.json"
            sae_weights_path = sae_root / "sae_weights.safetensors"
            sae_config_path.write_text("{}", encoding="utf-8")
            sae_weights_path.write_bytes(b"synthetic weights")
            output_path = custody / "assembled-packet.json"
            record_path = custody / "intake-record.json"
            result = assemble(
                design_packet_path=design_path,
                custody_root=custody,
                corpus_path=corpus_path,
                base_model_root=base_root,
                update_model_root=update_root,
                sae_config_path=sae_config_path,
                sae_weights_path=sae_weights_path,
                prediction_lock_path=prediction_lock_path,
                output_packet_path=output_path,
                intake_record_path=record_path,
            )
            assembled = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "CUSTODY_PACKET_ASSEMBLED_PENDING_EXTERNAL_REVIEW")
            self.assertEqual(assembled["status"], "FROZEN_DESIGN_READY_FOR_CUSTODY")
            self.assertEqual(assembled["prediction_lock"]["digest"], lock.digest)
            self.assertIsNone(assembled["independent_review"]["receipt"])
            self.assertEqual(assembled["packet_digest"], digest({key: value for key, value in assembled.items() if key != "packet_digest"}))
            self.assertEqual(output_path.stat().st_mode & 0o077, 0)
            self.assertEqual(record_path.stat().st_mode & 0o077, 0)


if __name__ == "__main__":
    unittest.main()
