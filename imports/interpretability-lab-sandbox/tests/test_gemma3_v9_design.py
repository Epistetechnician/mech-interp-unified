"""Design-only checks for the Gemma 3 V9 fallback packet.

State slice: interpretability-lab-sandbox-v1.
"""
import json
import unittest
from pathlib import Path

from interp_lab.canonical import digest


ROOT = Path(__file__).resolve().parents[1]


class Gemma3V9DesignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads((ROOT / "configs/gemma3-27b-reconstruction-v9-design-only.json").read_text(encoding="utf-8"))

    def test_packet_is_design_only_and_real_execution_is_closed(self) -> None:
        self.assertEqual(self.config["status"], "DESIGN_ONLY_NO_MODEL_EXECUTION")
        self.assertFalse(self.config["execution"]["real_model_enabled"])
        self.assertTrue(self.config["model"]["model_assets_external_only"])

    def test_model_and_official_asset_are_pinned(self) -> None:
        self.assertEqual(self.config["model"]["id"], "google/gemma-3-27b-pt")
        self.assertEqual(self.config["model"]["revision"], "9fe3c4ebc93fbadb14913801536d022054ef11cc")
        asset = self.config["feature_asset"]
        self.assertEqual(asset["repository"], "google/gemma-scope-2-27b-pt")
        self.assertEqual(asset["revision"], "5bf702afc144f3ba57b65bb3f76b12616385a213")
        self.assertEqual(asset["path"], "transcoder/layer_31_width_65k_l0_small_affine")
        self.assertEqual(asset["input_width"], self.config["model"]["hidden_width"])

    def test_hook_and_gate_order_are_closed_world(self) -> None:
        hook = self.config["hook"]
        self.assertEqual(hook["layer"], 31)
        self.assertEqual(hook["selection"], "fixed_output")
        self.assertEqual(
            self.config["qualification"]["gate_order"],
            [
                "native_instrumented_parity",
                "deterministic_repeatability",
                "exact_no_op",
                "nonzero_intervention_reach",
                "feature_reconstruction",
                "feature_repeat_stability",
                "event_replay",
                "custody_verification",
            ],
        )

    def test_downstream_lanes_are_locked_until_reconstruction(self) -> None:
        self.assertEqual(self.config["downstream"]["weco"], "closed_until_reconstruction_pass")
        self.assertEqual(self.config["downstream"]["causal_intervention"], "closed_until_reconstruction_pass_and_review")
        self.assertEqual(self.config["downstream"]["continual_learning"], "closed_until_heldout_causal_assessment")
        self.assertIsNone(self.config["feature_asset"]["weights_digest"])
        self.assertIsNone(self.config["corpus"]["corpus_digest"])

    def test_review_packet_binds_this_exact_design(self) -> None:
        review = json.loads((ROOT / "docs/review/gemma3-v9-review-packet.json").read_text(encoding="utf-8"))
        self.assertEqual(review["protocol_digest"], digest(self.config))
        self.assertEqual(review["packet_digest"], digest({key: value for key, value in review.items() if key != "packet_digest"}))
        self.assertEqual(review["review_status"], "PENDING_INDEPENDENT_REVIEW")


if __name__ == "__main__":
    unittest.main()
