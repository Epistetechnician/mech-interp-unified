"""Contract tests for the synthetic sandbox.

State slice: interpretability-lab-sandbox-v1.
"""
import unittest
from interp_lab.contracts import LineageEdge, StateSnapshot, Transition


class ContractTests(unittest.TestCase):
    def test_state_digest_changes_with_behavior(self):
        base = dict(state_id="s0", step=0, parameters=(1, 2), representation=(1, 2), circuits={"c": 1}, intervention_effects={"p": 1}, behavior=(1, 2))
        self.assertNotEqual(StateSnapshot(**base).state_digest, StateSnapshot(**{**base, "behavior": (1, 3)}).state_digest)

    def test_lineage_mass_is_bounded(self):
        with self.assertRaises(ValueError):
            Transition("a", "b", (LineageEdge("x", "y", 0.6), LineageEdge("x", "z", 0.5)))

    def test_transition_rejects_duplicate_edges(self):
        with self.assertRaises(ValueError):
            Transition("a", "b", (LineageEdge("x", "y", 0.5), LineageEdge("x", "y", 0.5)))
