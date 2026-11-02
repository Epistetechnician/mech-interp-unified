"""Tests for the deterministic V5 corpus generator.

State slice: interpretability-lab-sandbox-v1.
"""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.generate_qwen38_reconstruction_v5_corpus import generate


class Qwen38CorpusTests(unittest.TestCase):
    def test_roster_is_deterministic_and_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            first_result = generate(first)
            second_result = generate(second)
            first_payload = json.loads(first.read_text(encoding="utf-8"))
            second_payload = json.loads(second.read_text(encoding="utf-8"))
            self.assertEqual(first_payload, second_payload)
            self.assertEqual(first_result["sha256"], second_result["sha256"])
            self.assertEqual(len(first_payload["rows"]), 144)
            for key in ("family", "document_id"):
                values = {split: {row[key] for row in first_payload["rows"] if row["split"] == split} for split in ("fit", "tune", "assessment")}
                self.assertEqual(sum(map(len, values.values())), len(set().union(*values.values())))


if __name__ == "__main__":
    unittest.main()
