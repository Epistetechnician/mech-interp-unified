import glob
import os
import sys
import tempfile
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "experiments", "r6"))
import r6  # noqa: E402
from r6 import v50  # noqa: E402

LAYERS = r6.FROZEN6["target_layers"]
PLANTED = {0, 1, 2, 3, 200, 201, 202, 400, 401, 402, 403}


class Fake:
    """600 units over the 3 frozen layers. planted: units that carry the effect; flags choose which tasks see it."""
    def __init__(self, planted=True, t0=True, l=True, p=True, kb=0.01):
        self.planted, self.kb = planted, kb
        self.task_on = {"swap": t0, "last": l, "patch": p}
        self.locked, self.calls = None, []
        self._sc = {u: (5.0 if u in PLANTED else (u % 7) / 7.0) for u in range(600)}
        self.swap_set = set(sorted(PLANTED)[:8])

    def split(self):
        return {"fit": ["f"], "tune": ["t"], "assess": ["a1", "a2", "a3", "a4"]}
    def scores(self, fit):
        self.calls.append("scores"); return dict(self._sc)
    def layers(self): return {u: LAYERS[u // 200] for u in range(600)}
    def magnitude(self): return {u: float(u % 11) for u in range(600)}
    def mark_locked(self, lock): self.locked = lock
    def effect(self, units, docs, readout="margin"):
        if docs != ["f"] and self.locked is None:
            raise AssertionError("effect read before lock")
        e = sum(2.0 if (u in PLANTED and self.planted) else 0.002 * (u % 5) for u in units)
        return e if readout == "margin" else 0.5 * e
    def known_bad_effect(self, s, units, docs): return self.kb * (1 + s % 3) / 3
    def flip_scale(self, units, docs): return 0.1
    def activation_values(self, selected, docs): return [-3.0, -0.2, 0.4, 7.0]
    def unit_stats(self, units, docs): return {u: {"act_sing": 0.1, "act_plur": 0.2} for u in units}
    def item_classes(self, docs): return {i: ("sing" if n % 2 else "plur") for n, i in enumerate(sorted(docs))}
    def capture_last(self, units, docs): return {u: {i: 0.0 for i in docs} for u in units}
    def _task(self, units, name):
        self.calls.append(name)
        x = 0.5 if (self.task_on[name] and set(units) == self.swap_set) else 0.0
        return {"sing": x, "plur": x, "all": x}
    def swap_effect(self, units, docs, means): return self._task(units, "swap")
    def swap_effect_last(self, units, docs, means): return self._task(units, "last")
    def patch_last(self, units, docs, vals): return self._task(units, "patch")


CFG = {"seed": 71, "target_layers": LAYERS, "model_id": "fake", "corpus_digest": "c" * 64, "code_commit": "abc"}


class Frozen(unittest.TestCase):
    def test_layers_follow_repo_rule(self):
        self.assertEqual(LAYERS, [8, 15, 22])
        self.assertEqual(r6.pythia_plan.layers_for(32), [10, 18, 26])  # V61's plan is the same rule

    def test_values(self):
        f = r6.FROZEN6
        self.assertEqual((f["ctl_draws"], f["gate_draws"], f["swap_size"], f["k_per_layer"]), (1000, 1000, 8, 32))
        self.assertEqual((f["min_effect"], f["dtype"], f["stage_cap_s"]), (0.1, "float32", 28800))

    def test_digest_guard(self):
        r6.FROZEN6["ctl_draws"] = 100
        try:
            with self.assertRaises(v50.V50Error):
                r6.run_r6(Fake(), CFG)
        finally:
            r6.FROZEN6["ctl_draws"] = 1000
        self.assertEqual(v50.digest(r6.FROZEN6), r6.FROZEN6_DIGEST)

    def test_rejects_other_layers(self):
        with self.assertRaises(v50.V50Error):
            r6.run_r6(Fake(), dict(CFG, target_layers=[10, 18, 26]))


class Decision(unittest.TestCase):
    def test_classify(self):
        c = r6.classify
        self.assertEqual(c(False, True, True, True), "InstrumentNotValid")
        self.assertEqual(c(True, True, True, True), "Replicates")
        self.assertEqual(c(True, True, False, True), "PartiallyReplicates")
        self.assertEqual(c(True, True, True, False), "PartiallyReplicates")
        self.assertEqual(c(True, False, True, True), "DoesNotReplicate")  # T0 failing dominates (protocol sec. 3)

    def test_e2e_replicates(self):
        r = r6.run_r6(Fake(), CFG)
        self.assertEqual(r["outcome"], "Replicates")
        self.assertEqual(r["gate"]["status"], "InstrumentValid")
        self.assertEqual(r["swap_units"], sorted(PLANTED)[:8])
        for t in ("t0_all_position_swap", "l_last_position_swap", "p_donor_patch"):
            self.assertTrue(r[t]["supported"])
            self.assertAlmostEqual(r[t]["p_all"], 1 / 1001)  # 1000 controls, floor 0.000999
        self.assertEqual(r["selection_repeat_jaccard"], 1.0)

    def test_checkpoints_cover_every_finished_stage(self):
        seen = []
        r6.run_r6(Fake(), CFG, checkpoint=lambda o: seen.append(sorted(k for k in o if k in (
            "gate", "swap_units", "t0_all_position_swap", "l_last_position_swap", "p_donor_patch"))))
        self.assertEqual(len(seen), 5)  # gate, rank, T0, L, P
        self.assertEqual(seen[0], ["gate"])
        self.assertIn("t0_all_position_swap", seen[2])
        self.assertIn("p_donor_patch", seen[4])

    def test_partial_and_none(self):
        self.assertEqual(r6.run_r6(Fake(l=False), CFG)["outcome"], "PartiallyReplicates")
        self.assertEqual(r6.run_r6(Fake(t0=False), CFG)["outcome"], "DoesNotReplicate")

    def test_gate_failure_reads_no_effect(self):
        f = Fake(kb=5.0)  # known-bad control is large: instrument is not null-calibrated
        r = r6.run_r6(f, CFG)
        self.assertEqual(r["outcome"], "InstrumentNotValid")
        self.assertIn("null_consistency", r["gate"]["fails"])
        self.assertNotIn("swap", f.calls)
        self.assertNotIn("t0_all_position_swap", r)

    def test_no_effect_read_before_lock(self):
        f = Fake()
        r6.run_r6(f, CFG)  # Fake.effect raises AssertionError on non-fit docs before mark_locked
        self.assertIsNotNone(f.locked)

    def test_stage_timeout_is_reported_not_retried(self):
        t = iter(range(0, 10 ** 9, 20000))
        r = r6.run_r6(Fake(), CFG, clock=lambda: next(t))
        self.assertEqual(r["outcome"], "NotCompleted")
        self.assertEqual(r["timed_out_stage"], "gate")

    def test_controls_are_disjoint_from_candidates(self):
        f = Fake()
        seen = []
        orig = f.swap_effect
        f.swap_effect = lambda units, docs, means: (seen.append(set(units)), orig(units, docs, means))[1]
        r6.run_r6(f, CFG)
        self.assertEqual(len(seen), 1001)  # selected + 1000 controls
        ctl_units = set().union(*seen[1:])
        self.assertFalse(ctl_units & set(sorted(PLANTED)[:8]))


class RunnerCli(unittest.TestCase):
    def test_corpus_digest_mismatch_aborts(self):
        import json
        import run_r6
        with tempfile.TemporaryDirectory() as d:
            corpus = os.path.join(d, "c.json")
            with open(corpus, "w") as fh:
                json.dump({"manifest": {"a": 1}, "split_items": {}}, fh)
            hashes = os.path.join(d, "h.json")
            mdir = os.path.join(d, "model")
            os.mkdir(mdir)
            with open(os.path.join(mdir, "x.safetensors"), "wb"):
                pass
            with open(hashes, "w") as fh:
                json.dump(run_r6.hash_model(mdir), fh)
            args = ["run", "--model-dir", mdir, "--model-hashes", hashes, "--corpus", corpus, "--out", os.path.join(d, "o"),
                    "--expected-corpus-digest", "bad", "--code-commit", "abc"]
            with self.assertRaisesRegex(v50.V50Error, "corpus manifest differs"):
                run_r6.main(args)
            with open(hashes, "w") as fh:
                json.dump({"x.safetensors": "0" * 64}, fh)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                run_r6.main(args)


SNAP = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/*"))


@unittest.skipUnless(SNAP and os.path.exists(os.path.join(SNAP[0], "tokenizer.json")), "Qwen3 tokenizer not cached")
class QwenAdapter(unittest.TestCase):
    """Tiny random Qwen3 (no weights downloaded; only the cached tokenizer files are read)."""
    @classmethod
    def setUpClass(cls):
        os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config
        cls.torch = torch
        cls.tok = AutoTokenizer.from_pretrained(SNAP[0], local_files_only=True)
        cfg = Qwen3Config(vocab_size=len(cls.tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
        torch.manual_seed(0)
        cls.tmp = tempfile.TemporaryDirectory()
        AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).save_pretrained(cls.tmp.name)
        cls.tok.save_pretrained(cls.tmp.name)
        mk = lambda n, ids: [{"id": "%s%d" % (n, i), "text": t, "good": g, "bad": b}
                             for i, (t, g, b) in enumerate(ids)]
        pairs = [("The key on the old table", " is", " are"), ("The keys on the old table", " are", " is")] * 2
        cls.items = {"fit": mk("f", pairs), "tune": mk("t", pairs), "assess": mk("a", pairs)}
        import qwen3_backend
        cls.b = qwen3_backend.make_backend_class()(cls.tmp.name, cls.items, [0, 1, 2], device="cpu")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_units_and_scores(self):
        b = self.b
        self.assertEqual(b.I, 64)
        sc = b.scores(b.split()["fit"])
        self.assertEqual(len(sc), 3 * 64)
        self.assertTrue(any(v != 0 for v in sc.values()))

    def test_hook_point_is_down_proj_input(self):
        """Ablating every unit of layer 1 must equal zeroing that layer's down_proj weight."""
        torch, b = self.torch, self.b
        docs = b.split()["fit"]
        b.mark_locked("x")
        units = [1 * b.I + i for i in range(b.I)]  # target-layer index 1 is model layer LY[1] = 1
        ablated = b.effect(units, docs, "margin")
        base = b._mean(docs)
        w = b.layers_mod[b.LY[1]].mlp.down_proj.weight
        saved = w.detach().clone()
        w.data.zero_()
        try:
            b._reset()
            weights_zeroed = b._mean(docs)
        finally:
            w.data.copy_(saved)
        self.assertAlmostEqual(base - ablated, weights_zeroed, places=4)

    def test_activation_values_run_gate_path(self):
        """Regression: the gate's activation_values crashed on Qwen (Pythia dense_h_to_4h hook). Must return one
        finite value per selected unit and capture the down_proj input."""
        import math
        b = self.b
        docs = b.split()["fit"]
        sel = [0 * b.I + 3, 1 * b.I + 5, 2 * b.I + 7]
        vals = b.activation_values(sel, docs)
        self.assertEqual(len(vals), len(sel) * len(docs))
        self.assertTrue(all(math.isfinite(v) for v in vals))
        self.assertGreater(len(set(vals)), 1)

    def test_swap_patch_capture_shapes(self):
        b = self.b
        b.mark_locked("x")
        ids = b.split()["assess"]
        units = [0, 5, 64 + 3, 128 + 9]
        means = {u: (0.1, -0.1) for u in units}
        for fn in (b.swap_effect, b.swap_effect_last):
            r = fn(units, ids, means)
            self.assertEqual(set(r), {"sing", "plur", "all"})
        cap = b.capture_last(units, ids)
        self.assertEqual(set(cap), set(units))
        r = b.patch_last(units, ids, {u: {i: cap[u][i] for i in ids} for u in units})
        self.assertAlmostEqual(r["all"], 0.0, places=4)  # patching a unit's own activation changes nothing
        self.assertEqual(set(b.item_classes(ids).values()), {"sing", "plur"})
        st = b.unit_stats(units, b.split()["fit"])
        self.assertEqual(set(st[0]), {"dla_sing", "dla_plur", "dla", "act_sing", "act_plur"})


if __name__ == "__main__":
    unittest.main()
