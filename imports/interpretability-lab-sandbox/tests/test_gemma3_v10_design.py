"""Design-only checks for the Gemma 3 V10 assessment packet.

State slice: interpretability-lab-sandbox-v1.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from interp_lab.canonical import digest


ROOT = Path(__file__).resolve().parents[1]


class Gemma3V10DesignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(
            (ROOT / "configs/gemma3-27b-heldout-causal-retention-v10-design-only.json").read_text(
                encoding="utf-8"
            )
        )
        self.review = json.loads(
            (ROOT / "docs/review/gemma3-v10-review-packet.json").read_text(encoding="utf-8")
        )

    def test_packet_is_design_only_and_external_accept_is_required(self) -> None:
        self.assertEqual(self.config["status"], "FROZEN_DESIGN_REQUIRES_EXTERNAL_REVIEW")
        self.assertEqual(self.config["execution"]["mode"], "external-real")
        self.assertTrue(self.config["execution"]["external_review_required"])
        self.assertFalse(self.config["independent_review"]["operator_may_self_sign"])
        self.assertEqual(self.review["review_status"], "PENDING_INDEPENDENT_REVIEW")
        self.assertFalse(self.review["acceptance"]["operator_may_self_sign"])

    def test_model_asset_hook_and_fresh_corpus_are_locked(self) -> None:
        self.assertEqual(self.config["model"]["id"], "google/gemma-3-27b-pt")
        self.assertEqual(
            self.config["model"]["revision"],
            "9fe3c4ebc93fbadb14913801536d022054ef11cc",
        )
        asset = self.config["feature_asset"]
        self.assertEqual(asset["repository"], "google/gemma-scope-2-27b-pt")
        self.assertEqual(asset["revision"], "5bf702afc144f3ba57b65bb3f76b12616385a213")
        self.assertEqual(asset["input_width"], self.config["model"]["hidden_width"])
        self.assertEqual(asset["hook_input"], "model.layers.31.pre_feedforward_layernorm.output")
        self.assertEqual(asset["hook_output"], "model.layers.31.post_feedforward_layernorm.output")
        self.assertTrue(self.config["corpus"]["fresh"])
        self.assertEqual(self.config["corpus"]["identity"], "fresh-gemma3-v10-heldout-causal-retention-20260917")
        self.assertFalse(self.config["custody"]["v9_scientific_artifacts_as_inputs"])

    def test_thresholds_and_controller_success_rule_are_fixed(self) -> None:
        thresholds = self.config["qualification"]
        self.assertEqual(thresholds["reconstruction_pooled_fvu_max"], 0.05)
        self.assertEqual(thresholds["reconstruction_max_row_nmse"], 0.05)
        self.assertEqual(thresholds["cross_checkpoint_min_cosine"], 0.8)
        self.assertEqual(thresholds["heldout_sign_agreement_min"], 0.8)
        self.assertEqual(thresholds["scrub_min_abs_effect"], 0.01)
        self.assertTrue(thresholds["zero_attrition"])
        continual = self.config["continual_learning"]
        self.assertEqual(
            continual["controllers"], ["sgd", "ewc", "behavioral_only", "causal_monitor"]
        )
        self.assertTrue(continual["matched_total_compute"])
        self.assertEqual(continual["superiority_min_abs"], 0.05)
        self.assertEqual(continual["adaptation_noninferiority_margin"], 0.05)

    def test_review_packet_binds_exact_design(self) -> None:
        self.assertEqual(self.review["protocol_digest"], digest(self.config))
        self.assertEqual(
            self.review["packet_digest"],
            digest({key: value for key, value in self.review.items() if key != "packet_digest"}),
        )


if __name__ == "__main__":
    unittest.main()
