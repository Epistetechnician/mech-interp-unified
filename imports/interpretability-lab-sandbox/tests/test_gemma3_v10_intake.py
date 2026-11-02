"""Custody-bound V10 intake and fail-closed gate tests.

State slice: interpretability-lab-sandbox-v1.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.assemble_gemma3_v10_packet import assemble
from scripts.check_gemma3_v10_gate import check_packet_gate


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "configs/gemma3-27b-heldout-causal-retention-v10-design-only.json"


class Gemma3V10IntakeTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        custody = root / "custody"
        custody.mkdir(mode=0o700)
        os.chmod(custody, 0o700)
        for name in ("base", "post"):
            model = custody / name
            model.mkdir(mode=0o700)
            os.chmod(model, 0o700)
            (model / "config.json").write_text(json.dumps({"model": name}), encoding="utf-8")
        feature = custody / "feature"
        feature.mkdir(mode=0o700)
        os.chmod(feature, 0o700)
        (feature / "config.json").write_text("{}\n", encoding="utf-8")
        (feature / "params.safetensors").write_bytes(b"fixture")
        rows = []
        for split in ("fit", "tune", "assessment"):
            for index in range(48):
                row_id = f"{split}-{index:03d}"
                rows.append(
                    {
                        "row_id": row_id,
                        "split": split,
                        "family": f"{split}-family-{index:03d}",
                        "document_id": f"{split}-document-{index:03d}",
                        "text": f"fresh {split} row {index}",
                        "target_token_id": 1,
                        "contrast_token_id": 2,
                        "donor_row_id": f"fit-{index:03d}",
                        "shuffled_donor_row_id": f"tune-{index:03d}",
                    }
                )
        corpus = custody / "corpus.json"
        corpus.write_text(
            json.dumps(
                {
                    "schema": "interp-lab.assessment-corpus.v1",
                    "state_slice": "interpretability-lab-sandbox-v1",
                    "fresh": True,
                    "rows": rows,
                }
            ),
            encoding="utf-8",
        )
        lock = custody / "prediction-lock.json"
        lock.write_text(
            json.dumps(
                {
                    "schema": "interp-lab.prediction-lock.v1",
                    "state_slice": "interpretability-lab-sandbox-v1",
                    "feature_index": 3,
                    "expected_sign": 1,
                    "source": "fit_tune_only",
                }
            ),
            encoding="utf-8",
        )
        attestation = custody / "runtime-attestation.json"
        attestation.write_text(
            json.dumps(
                {
                    "schema": "interp-lab.runtime-attestation.v1",
                    "state_slice": "interpretability-lab-sandbox-v1",
                    "runtime": {
                        "python": "3.11.11",
                        "torch": "2.6.0",
                        "transformers": "4.57.3",
                        "safetensors": "0.8.0",
                    },
                    "device": {"model": "NVIDIA H100 80GB HBM3"},
                    "determinism": {
                        "cublas_workspace_config": ":4096:8",
                        "cudnn_benchmark": False,
                        "tf32": False,
                        "torch_deterministic_algorithms": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        return custody, feature

    def test_assembly_binds_inputs_and_gate_stays_closed_without_accept(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            custody, feature = self._fixture(Path(directory))
            packet_path = custody / "packet.json"
            intake_path = custody / "intake.json"
            result = assemble(
                design_path=DESIGN,
                custody_root=custody,
                corpus_path=custody / "corpus.json",
                base_model_root=custody / "base",
                post_update_model_root=custody / "post",
                sae_config_path=feature / "config.json",
                sae_params_path=feature / "params.safetensors",
                prediction_lock_path=custody / "prediction-lock.json",
                runtime_attestation_path=custody / "runtime-attestation.json",
                output_packet_path=packet_path,
                intake_record_path=intake_path,
            )
            self.assertEqual(result["status"], "CUSTODY_BOUND_PENDING_EXTERNAL_ACCEPT")
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
            self.assertEqual(packet["status"], "CUSTODY_BOUND_PENDING_EXTERNAL_ACCEPT")
            self.assertEqual(packet["selection"]["feature_index"], 3)
            self.assertEqual(packet["corpus"]["splits"], {"fit": 48, "tune": 48, "assessment": 48})
            gate = check_packet_gate(packet, custody_root=custody, review_receipt=None)
            self.assertEqual(gate["status"], "CLOSED")
            self.assertIn("independent packet-bound Ed25519 ACCEPT receipt is missing", gate["reasons"])


if __name__ == "__main__":
    unittest.main()
