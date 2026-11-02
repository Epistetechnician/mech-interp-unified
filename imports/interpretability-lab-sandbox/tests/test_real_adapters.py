"""Contract tests for the optional real adapters.

State slice: interpretability-lab-sandbox-v1.
"""
import tempfile
import unittest
from pathlib import Path

from interp_lab.real_adapters import GPT2Runtime, GPT2Spec, Gemma3JumpReLUSAE, Gemma3Runtime, Gemma3Spec, Qwen38Runtime, Qwen38Spec, RealAdapterError
from scripts.run_real_qwen38_reconstruction_v5 import ADAPTER_HOOKS


class RealAdapterContractTests(unittest.TestCase):
    def test_runner_hook_ids_bind_adapter_aliases(self):
        self.assertEqual(ADAPTER_HOOKS, {"decoder_input": "decoder_in", "decoder_output": "decoder_out"})

    def test_gpt2_spec_freezes_canonical_layer(self):
        spec = GPT2Spec(
            model_id="openai-community/gpt2",
            revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
            model_root=Path("/owner-only/model"),
        )
        self.assertEqual(spec.layer, 6)

    def test_gpt2_spec_rejects_layer_shopping(self):
        with self.assertRaisesRegex(RealAdapterError, "canonical GPT-2"):
            GPT2Spec(
                model_id="openai-community/gpt2",
                revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
                model_root=Path("/owner-only/model"),
                layer=7,
            )

    def test_gpt2_spec_freezes_supported_resid_mid_hook(self):
        spec = GPT2Spec(
            model_id="openai-community/gpt2",
            revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
            model_root=Path("/owner-only/model"),
            hook="resid_mid",
        )
        self.assertEqual(spec.hook, "resid_mid")

    def test_gpt2_spec_rejects_unregistered_hook(self):
        with self.assertRaisesRegex(RealAdapterError, "unsupported GPT-2 hook"):
            GPT2Spec(
                model_id="openai-community/gpt2",
                revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
                model_root=Path("/owner-only/model"),
                hook="attention_out",
            )

    def test_runtime_rejects_missing_model_root_before_importing_transformers(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaisesRegex(RealAdapterError, "model root is missing"):
                GPT2Runtime(
                    GPT2Spec(
                        model_id="openai-community/gpt2",
                        revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
                        model_root=missing,
                    )
                )

    def test_qwen38_spec_freezes_decoder_registry(self):
        spec = Qwen38Spec(
            model_id="Qwen/Qwen3.8-27B",
            revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            model_root=Path("/owner-only/model"),
        )
        self.assertEqual(spec.canonical_path, "model.language_model.layers[32]")

    def test_qwen38_spec_rejects_layer_shopping(self):
        with self.assertRaisesRegex(RealAdapterError, "decoder layer must be in 0..63"):
            Qwen38Spec(
                model_id="Qwen/Qwen3.8-27B",
                revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                model_root=Path("/owner-only/model"),
                layer=64,
            )

    def test_qwen38_spec_accepts_declared_decoder_input_hook(self):
        spec = Qwen38Spec(
            model_id="Qwen/Qwen3.8-27B",
            revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            model_root=Path("/owner-only/model"),
            hook="decoder_in",
        )
        self.assertEqual(spec.hook, "decoder_in")

    def test_qwen38_spec_rejects_unregistered_hook(self):
        with self.assertRaisesRegex(RealAdapterError, "choose decoder_in or decoder_out"):
            Qwen38Spec(
                model_id="Qwen/Qwen3.8-27B",
                revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                model_root=Path("/owner-only/model"),
                hook="attention_out",
            )

    def test_qwen38_runtime_rejects_missing_model_root_before_importing_transformers(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaisesRegex(RealAdapterError, "model root is missing"):
                from interp_lab.real_adapters import Qwen38Runtime

                Qwen38Runtime(
                    Qwen38Spec(
                        model_id="Qwen/Qwen3.8-27B",
                        revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                        model_root=missing,
                    )
                )

    def test_qwen38_capture_all_returns_each_token_activation(self):
        runtime = object.__new__(Qwen38Runtime)
        runtime._run = lambda row, **kwargs: (None, [[[1.0, 2.0], [3.0, 4.0]]])
        self.assertEqual(runtime.capture_all(["row"]), (((1.0, 2.0), (3.0, 4.0)),))

    def test_gemma3_spec_freezes_official_scope_layer(self):
        spec = Gemma3Spec(
            model_id="google/gemma-3-27b-pt",
            revision="9fe3c4ebc93fbadb14913801536d022054ef11cc",
            model_root=Path("/owner-only/model"),
        )
        self.assertEqual(spec.canonical_path, "model.language_model.layers[31]")

    def test_gemma3_spec_rejects_layer_shopping(self):
        with self.assertRaisesRegex(RealAdapterError, "canonical Gemma 3 Scope layer"):
            Gemma3Spec(
                model_id="google/gemma-3-27b-pt",
                revision="9fe3c4ebc93fbadb14913801536d022054ef11cc",
                model_root=Path("/owner-only/model"),
                layer=30,
            )

    def test_gemma3_runtime_rejects_missing_model_root_before_importing_transformers(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RealAdapterError, "model root is missing"):
                Gemma3Runtime(
                    Gemma3Spec(
                        model_id="google/gemma-3-27b-pt",
                        revision="9fe3c4ebc93fbadb14913801536d022054ef11cc",
                        model_root=Path(directory) / "missing",
                    )
                )

    def test_gemma3_capture_all_returns_each_token_activation(self):
        runtime = object.__new__(Gemma3Runtime)
        runtime._run = lambda row, **kwargs: (None, [[[1.0, 2.0], [3.0, 4.0]]])
        self.assertEqual(runtime.capture_all(["row"]), (((1.0, 2.0), (3.0, 4.0)),))

    def test_gemma3_transcoder_capture_preserves_input_output_pair(self):
        runtime = object.__new__(Gemma3Runtime)
        runtime._run = lambda row, **kwargs: (None, ([[[1.0, 2.0], [3.0, 4.0]]], [[[5.0, 6.0], [7.0, 8.0]]]))
        inputs, outputs = runtime.capture_transcoder(["row"])
        self.assertEqual(inputs, (((1.0, 2.0), (3.0, 4.0)),))
        self.assertEqual(outputs, (((5.0, 6.0), (7.0, 8.0)),))

    def test_gemma3_sae_rejects_missing_weights_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RealAdapterError, "Gemma 3 SAE weights are missing"):
                Gemma3JumpReLUSAE(Path(directory) / "missing.safetensors")
