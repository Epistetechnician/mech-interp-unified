"""Tests for custody-bound V5 packet compilation.

State slice: interpretability-lab-sandbox-v1.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from interp_lab.custody import verify_packet_binding
from interp_lab.reconstruction_packet import ReconstructionPacketError, freeze_reconstruction_packet


def _design() -> dict:
    return {
        "schema": "interp-lab.qwen38-reconstruction-protocol.v1",
        "state_slice": "interpretability-lab-sandbox-v1",
        "status": "DESIGN_ONLY_NO_MODEL_EXECUTION",
        "purpose": "test",
        "model": {"id": "Qwen/Qwen3.8-27B", "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "hidden_width": 5120, "layers": 64},
        "runtime": {"transformers": None, "transformers_source_revision": None, "python": "3.11.11", "torch": "2.6.0", "safetensors": "0.8.0", "cuda": "12.4", "device_model": "NVIDIA H100", "dtype": "bfloat16", "determinism": {}},
        "execution": {"hard_usd_ceiling": 333.0, "network_during_model_execution": False, "mode": "internal-real"},
        "feature_asset": {"strategy": "train_model_matched_affine_topk_sae", "input_width": 5120, "feature_width": 40960, "top_k": 50, "normalization": "x_scaled_by_sqrt_d_over_mean_norm", "decoder_rows": "unit_norm"},
        "corpus": {"fresh": True, "document_disjoint": True, "token_capture": "all_nonpadding_positions", "fit_families": 48, "tune_families": 48, "assessment_families": 48},
        "hook_selection": {"layer": 32, "candidates": [{"id": "decoder_input", "path_template": "model.language_model.layers[32]"}, {"id": "decoder_output", "path_template": "model.language_model.layers[32]"}]},
        "reconstruction": {"primary_metric": "pooled_fvu_global_centered"},
        "downstream": {},
    }


class ReconstructionPacketTests(unittest.TestCase):
    def test_compiler_binds_digests_and_never_changes_design(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.chmod(root, 0o700)
            model = root / "model"
            model.mkdir()
            os.chmod(model, 0o700)
            (model / "weights.bin").write_bytes(b"model")
            sae = root / "sae.pt"
            sae.write_bytes(b"sae")
            corpus = root / "corpus.json"
            corpus.write_text("{\"identity\":\"fresh\"}\n", encoding="utf-8")
            design = root / "design.json"
            design.write_text(json.dumps(_design(), sort_keys=True), encoding="utf-8")
            before = design.read_bytes()
            output = root / "packet.json"
            result = freeze_reconstruction_packet(
                design_path=design,
                custody_root=root,
                model_root=model,
                sae_weights=sae,
                corpus_path=corpus,
                corpus_identity="fresh-v5",
                transformers_version="5.18.0.dev0",
                transformers_source_revision="a" * 40,
                packet_id="packet-v5",
                output_path=output,
            )
            packet = json.loads(output.read_text(encoding="utf-8"))
            verify_packet_binding(packet, packet["packet_digest"])
            self.assertEqual(result["packet_digest"], packet["packet_digest"])
            self.assertEqual(design.read_bytes(), before)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertEqual(packet["feature_asset"]["weights_sha256"], result["sae_sha256"])

    def test_compiler_rejects_changed_design_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.chmod(root, 0o700)
            design = root / "design.json"
            payload = _design()
            payload["status"] = "FROZEN_INTERNAL_CUSTODY_BOUND"
            design.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ReconstructionPacketError, "design-only boundary"):
                freeze_reconstruction_packet(
                    design_path=design,
                    custody_root=root,
                    model_root=root,
                    sae_weights=root / "sae.pt",
                    corpus_path=root / "corpus.json",
                    corpus_identity="fresh-v5",
                    transformers_version="5.18.0.dev0",
                    transformers_source_revision="a" * 40,
                    packet_id="packet-v5",
                    output_path=root / "packet.json",
                )

    def test_compiler_binds_fresh_layer33_upstream_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.chmod(root, 0o700)
            model = root / "model"
            model.mkdir()
            os.chmod(model, 0o700)
            (model / "weights.bin").write_bytes(b"model")
            sae = root / "sae_L33_final.pt"
            sae.write_bytes(b"fresh-layer-33")
            corpus = root / "corpus.json"
            corpus.write_text("{\"identity\":\"fresh\"}\n", encoding="utf-8")
            design_payload = _design()
            design_payload["hook_selection"] = {
                "layer": 33,
                "candidates": [
                    {"id": "decoder_input", "path_template": "model.language_model.layers[33]"},
                    {"id": "decoder_output", "path_template": "model.language_model.layers[33]"},
                ],
            }
            design_payload["feature_asset"].update({
                "strategy": "upstream_model_matched_affine_topk_sae",
                "repository": "LeonInverse/qwen3.8-27b-scope-code",
                "revision": "0f965454ffedb93e826546a7dc11b51a4622188e",
                "path": "v2/sae_L33_final.pt",
            })
            design = root / "design.json"
            design.write_text(json.dumps(design_payload, sort_keys=True), encoding="utf-8")
            output = root / "packet.json"
            freeze_reconstruction_packet(
                design_path=design,
                custody_root=root,
                model_root=model,
                sae_weights=sae,
                corpus_path=corpus,
                corpus_identity="fresh-v5-layer33",
                transformers_version="5.18.0.dev0",
                transformers_source_revision="a" * 40,
                packet_id="packet-v5-layer33",
                output_path=output,
            )
            packet = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(packet["model"]["canonical_layer"], 33)
            self.assertEqual(packet["feature_asset"]["repository"], "LeonInverse/qwen3.8-27b-scope-code")
            self.assertEqual(packet["feature_asset"]["revision"], "0f965454ffedb93e826546a7dc11b51a4622188e")

    def test_compiler_binds_v6_transformers_main_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.chmod(root, 0o700)
            model = root / "model"
            model.mkdir()
            os.chmod(model, 0o700)
            (model / "weights.bin").write_bytes(b"model")
            sae = root / "sae_L33_final.pt"
            sae.write_bytes(b"fresh-layer-33")
            corpus = root / "corpus.json"
            corpus.write_text("{\"identity\":\"fresh\"}\n", encoding="utf-8")
            payload = _design()
            payload.update({
                "schema": "interp-lab.qwen38-reconstruction-protocol.v2",
                "continuation": "transformers-main-qwen3_5",
            })
            payload["runtime"].update({
                "transformers": "5.18.0.dev0",
                "transformers_source_revision": "dbea10932a0e18f40d8103a3bae6fa7927f8d36e",
            })
            payload["hook_selection"] = {
                "layer": 33,
                "candidates": [
                    {"id": "decoder_input", "path_template": "model.language_model.layers[33]"},
                    {"id": "decoder_output", "path_template": "model.language_model.layers[33]"},
                ],
            }
            payload["feature_asset"].update({
                "strategy": "upstream_model_matched_affine_topk_sae",
                "repository": "LeonInverse/qwen3.8-27b-scope-code",
                "revision": "0f965454ffedb93e826546a7dc11b51a4622188e",
                "path": "v2/sae_L33_final.pt",
            })
            design = root / "design.json"
            design.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            output = root / "packet.json"
            freeze_reconstruction_packet(
                design_path=design,
                custody_root=root,
                model_root=model,
                sae_weights=sae,
                corpus_path=corpus,
                corpus_identity="fresh-v6-layer33",
                transformers_version="5.18.0.dev0",
                transformers_source_revision="dbea10932a0e18f40d8103a3bae6fa7927f8d36e",
                packet_id="packet-v6-layer33",
                output_path=output,
            )
            packet = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(packet["protocol"], "qwen38-reconstruction-v6")
            self.assertEqual(packet["runtime"]["transformers"], "5.18.0.dev0")

    def test_compiler_binds_v7_vision_runtime_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.chmod(root, 0o700)
            model = root / "model"
            model.mkdir()
            os.chmod(model, 0o700)
            (model / "weights.bin").write_bytes(b"model")
            sae = root / "sae.pt"
            sae.write_bytes(b"sae")
            corpus = root / "corpus.json"
            corpus.write_text("{}\n", encoding="utf-8")
            payload = _design()
            payload.update({"schema": "interp-lab.qwen38-reconstruction-protocol.v3", "continuation": "transformers-main-qwen3_5-vision-runtime"})
            payload["runtime"].update({"transformers": "5.18.0.dev0", "transformers_source_revision": "dbea10932a0e18f40d8103a3bae6fa7927f8d36e", "torchvision": "0.21.0", "pillow": "11.3.0"})
            design = root / "design.json"
            design.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            output = root / "packet.json"
            freeze_reconstruction_packet(
                design_path=design,
                custody_root=root,
                model_root=model,
                sae_weights=sae,
                corpus_path=corpus,
                corpus_identity="fresh-v7",
                transformers_version="5.18.0.dev0",
                transformers_source_revision="dbea10932a0e18f40d8103a3bae6fa7927f8d36e",
                packet_id="packet-v7",
                output_path=output,
            )
            packet = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(packet["protocol"], "qwen38-reconstruction-v7")
            self.assertEqual(packet["runtime"]["torchvision"], "0.21.0")
            self.assertEqual(packet["runtime"]["pillow"], "11.3.0")
