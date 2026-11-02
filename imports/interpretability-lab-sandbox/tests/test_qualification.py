"""Qualification and continual-learning tests.

State slice: interpretability-lab-sandbox-v1.
"""
import unittest
from interp_lab.continual import run_continual_suite
from interp_lab.qualification import run_qualification


class QualificationTests(unittest.TestCase):
    def test_fixed_qualification_passes_and_is_synthetic_only(self):
        report = run_qualification()
        self.assertTrue(report["all_gates_pass"])
        self.assertEqual(report["claim_status"], "LOCAL_SYNTHETIC_ONLY")
        self.assertEqual(report["gate_order"], ("native_instrumented_parity", "deterministic_repeatability", "exact_no_op", "nonzero_intervention_reach", "feature_reconstruction", "feature_stability", "causal_scrubbing"))
        self.assertEqual(len(report["report_digest"]), 64)

    def test_qualification_is_repeatable(self):
        self.assertEqual(run_qualification(), run_qualification())

    def test_continual_suite_accounts_for_monitoring(self):
        report = run_continual_suite()
        self.assertTrue(report["matched_stream"])
        self.assertEqual(set(report["controllers"]), {"sgd", "replay", "ewc", "behavioral_only", "causal_monitor"})
        self.assertEqual(report["controllers"]["causal_monitor"]["monitoring_overhead"], 1.0)
