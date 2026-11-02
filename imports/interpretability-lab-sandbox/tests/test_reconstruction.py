"""Tests for the fresh model-matched reconstruction diagnostics.

State slice: interpretability-lab-sandbox-v1.
"""
import math
import unittest

from interp_lab.reconstruction import ReconstructionError, summarize_reconstruction


class ReconstructionTests(unittest.TestCase):
    def test_pooled_fvu_uses_one_global_center(self):
        summary = summarize_reconstruction(((1.0, 3.0), (5.0, 7.0)), ((1.0, 2.0), (6.0, 7.0)))
        self.assertEqual(summary.row_count, 2)
        self.assertEqual(summary.width, 2)
        self.assertAlmostEqual(summary.pooled_fvu, 0.1)

    def test_row_metric_exposes_localized_failure(self):
        summary = summarize_reconstruction(((1.0, 1.0), (1.0, 100.0)), ((0.0, 0.0), (1.0, 100.0)))
        self.assertGreater(summary.max_row_nmse, summary.pooled_fvu)
        self.assertAlmostEqual(summary.max_row_nmse, 1.0)

    def test_constant_targets_are_fail_closed(self):
        exact = summarize_reconstruction(((2.0, 2.0),), ((2.0, 2.0),))
        self.assertEqual(exact.pooled_fvu, 0.0)
        failed = summarize_reconstruction(((2.0, 2.0),), ((2.0, 3.0),))
        self.assertTrue(math.isinf(failed.pooled_fvu))

    def test_shape_and_finite_checks(self):
        with self.assertRaisesRegex(ReconstructionError, "same shape"):
            summarize_reconstruction(((1.0, 2.0),), ((1.0,),))
        with self.assertRaisesRegex(ReconstructionError, "finite"):
            summarize_reconstruction(((float("nan"),),), ((0.0,),))
