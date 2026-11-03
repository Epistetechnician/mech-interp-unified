import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "r1b"))
import fakes  # noqa: E402
import lib  # noqa: E402

VOCAB = [f"w{i}" for i in range(60)]


def world(rng, n, length=32):
    return [lib.Ctx([VOCAB[i] for i in rng.integers(0, len(VOCAB), length)], np.zeros(length)) for _ in range(n)]


def set_pole(rng, fit, assess, sets, idx, truth):
    def apply(pool):
        return [lib.Ctx(c.tokens, np.array([1.0 if any(t in s for s in sets) else 0.0 for t in c.tokens])) for c in pool]
    f, a = apply(fit), apply(assess)
    return lib.Pole(0, idx, 1, f, a, lib.top_tokens_of(f), truth)


def build_poles(seed=0, n_each=10):
    rng = np.random.default_rng(seed)
    fit, assess = world(rng, 120), world(rng, 80)
    poles = []
    for i in range(n_each):
        poles.append(set_pole(rng, fit, assess, [set(rng.choice(VOCAB, 4, replace=False).tolist())], i, "one"))
    for i in range(n_each):
        t = rng.choice(VOCAB, 8, replace=False).tolist()
        poles.append(set_pole(rng, fit, assess, [set(t[:4]), set(t[4:])], 100 + i, "two"))
    return poles


class TestBasics(unittest.TestCase):
    def test_pearson_edges(self):
        self.assertEqual(lib.pearson([1, 1, 1], [1, 2, 3]), 0.0)
        self.assertAlmostEqual(lib.pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertEqual(lib.pearson([1, 2], [1]), 0.0)

    def test_classify(self):
        self.assertEqual(lib.classify(0.9, 0.9, 0.9, 0.5), "one")
        self.assertEqual(lib.classify(0.6, 0.9, 0.91, 0.5), "two")
        self.assertEqual(lib.classify(0.3, 0.55, 0.9, 0.5), "many")
        self.assertEqual(lib.classify(0.1, 0.2, 0.49, 0.5), "unexplained")
        self.assertEqual(lib.classify(0.9, 0.9, 0.9, 0.95), "unexplained")

    def test_parse_jobs(self):
        self.assertEqual(lib.parse_jobs('{"jobs": ["a", "b"]}'), ["a", "b"])
        self.assertEqual(lib.parse_jobs('<think>x {"jobs": ["no"]}</think>\nsure: {"jobs": ["yes"]} done'), ["yes"])
        self.assertEqual(len(lib.parse_jobs('{"jobs": ["a","b","c","d"]}')), 3)
        self.assertEqual(lib.parse_jobs("not json"), [])
        self.assertEqual(lib.parse_jobs('{"jobs": "oops"}'), [])
        self.assertEqual(lib.parse_jobs('{"jobs": ["", "  "]}'), [])

    def test_derangement(self):
        rng = np.random.default_rng(0)
        for n in (2, 5, 50):
            p = lib.derangement(n, rng)
            self.assertFalse(np.any(p == np.arange(n)))
            self.assertEqual(sorted(p.tolist()), list(range(n)))
        with self.assertRaises(ValueError):
            lib.derangement(1, rng)


class TestSampling(unittest.TestCase):
    def test_sample_assess(self):
        rng = np.random.default_rng(0)
        ctxs = [lib.Ctx(["a"] * 4, np.full(4, float(i % 7))) for i in range(100)]
        out = lib.sample_assess(ctxs, np.random.default_rng(1))
        self.assertEqual(len(out), 60)
        self.assertEqual(len({id(c) for c in out}), 60)
        self.assertTrue(all(c.x.max() == 6.0 for c in out[:14]))


class TestEvidence(unittest.TestCase):
    def test_evidence_shape_and_marks(self):
        pole = build_poles(n_each=1)[0]
        text = lib.format_evidence(pole.fit, np.random.default_rng(1))
        lines = text.split("\n")
        kinds = [ln.split("]")[0].split()[0].lstrip("[") for ln in lines]
        self.assertEqual(kinds.count("top"), lib.N_TOP)
        self.assertLessEqual(kinds.count("mid"), lib.N_MID)
        self.assertEqual(kinds.count("random"), lib.N_RAND)
        self.assertIn("«", text)
        self.assertEqual(text, lib.format_evidence(pole.fit, np.random.default_rng(1)))

    def test_prompt_never_sees_assess(self):
        poles = build_poles(n_each=2)
        p = poles[0]
        lab = fakes.FakeLabeller()
        j1 = lab.explain(lib.explainer_prompt(lib.format_evidence(p.fit, np.random.default_rng(0))))
        p2 = lib.Pole(p.layer, p.neuron, p.sign, p.fit, [lib.Ctx(c.tokens, 1 - c.x) for c in p.assess])
        j2 = lib.audit_pole(p2, lab, fakes.FakeSimulator(), np.random.default_rng(0))["jobs"]
        self.assertEqual(j1, j2)


class TestRulePoles(unittest.TestCase):
    def test_make_rule_poles(self):
        rng = np.random.default_rng(3)
        fit, assess = world(rng, 400), world(rng, 200)
        poles = lib.make_rule_poles(fit, assess, np.random.default_rng(4), n_each=3, min_cover=30)
        self.assertEqual([p.truth for p in poles], ["one"] * 6 + ["two"] * 3)
        for p in poles:
            self.assertGreaterEqual(sum(c.x.sum() for c in p.fit), 30)
            self.assertTrue(set(np.unique(flat(p.assess))) <= {0.0, 1.0})
        again = lib.make_rule_poles(fit, assess, np.random.default_rng(4), n_each=3, min_cover=30)
        self.assertEqual([p.meta for p in poles], [p.meta for p in again])

    def test_union_is_max_of_parts(self):
        rng = np.random.default_rng(5)
        fit, assess = world(rng, 400), world(rng, 200)
        u = [p for p in lib.make_rule_poles(fit, assess, np.random.default_rng(6), n_each=2) if p.truth == "two"][0]
        s = frozenset(u.meta["spec"]["set"])
        for c, orig in zip(u.fit[:20], fit[:20]):
            for i, t in enumerate(c.tokens):
                after = i > 0 and orig.tokens[i - 1] == u.meta["spec"]["after"]
                self.assertEqual(c.x[i], 1.0 if (t in s or after) else 0.0)


def flat(ctxs):
    return lib.flat_x(ctxs)


class TestCalibration(unittest.TestCase):
    def run_audit(self, labeller, sim=None):
        poles = build_poles()
        sim = sim or fakes.FakeSimulator()
        res = [lib.audit_pole(p, labeller, sim, np.random.default_rng(i)) for i, p in enumerate(poles)]
        return poles, res

    def test_c1_passes_with_good_labeller(self):
        poles, res = self.run_audit(fakes.FakeLabeller())
        m = lib.c1_metrics(res, poles)
        self.assertTrue(m["pass"], m)
        self.assertGreaterEqual(m["one_recall"], 0.8)

    def test_c1_fails_with_random_labeller(self):
        poles, res = self.run_audit(fakes.RandomLabeller(VOCAB))
        m = lib.c1_metrics(res, poles)
        self.assertFalse(m["pass"], m)

    def test_c2_shuffled_descriptions_score_low(self):
        poles, res = self.run_audit(fakes.FakeLabeller())
        perm = lib.derangement(len(poles), np.random.default_rng(1))
        sim = fakes.FakeSimulator()
        r3 = [lib.score_jobs(sim, res[j]["jobs"], poles[i].assess)["r3"] for i, j in enumerate(perm)]
        m = lib.c2_metrics(r3)
        self.assertTrue(m["pass"], m)

    def test_c2_detects_leaky_pipeline(self):
        self.assertFalse(lib.c2_metrics([0.9, 0.8, 0.7])["pass"])

    def test_verdict_logic(self):
        ok = {"pass": True}
        bad = {"pass": False}
        self.assertEqual(lib.verdict(ok, ok, ok, ok), "validated")
        self.assertEqual(lib.verdict(bad, ok, ok, ok), "labeller_too_weak")
        self.assertEqual(lib.verdict(ok, bad, ok, ok), "calibration_failed")

    def test_c3(self):
        a = ["one", "two", "one", "many", "one"]
        s = np.random.default_rng(0).random(100)
        self.assertTrue(lib.c3_metrics(a, a, s, s + 1e-3)["pass"])
        self.assertFalse(lib.c3_metrics(a, ["one"] * 5, s, s)["pass"])
        self.assertFalse(lib.c3_metrics(a, a, s, s[::-1])["pass"])

    def test_c4_requires_pairs_and_detection(self):
        self.assertFalse(lib.c4_metrics(["two"] * 10)["pass"])
        self.assertTrue(lib.c4_metrics(["two"] * 25 + ["many"] * 5)["pass"])
        self.assertFalse(lib.c4_metrics(["one"] * 20 + ["two"] * 10)["pass"])

    def test_composites_use_disjoint_top_tokens_and_are_detected(self):
        poles = build_poles(n_each=12)
        ones = [p for p in poles if p.truth == "one"]
        comps = lib.make_composites(ones, np.random.default_rng(2), max_pairs=40)
        self.assertGreater(len(comps), 0)
        for c in comps:
            self.assertEqual(c.truth, "two")
        res = [lib.audit_pole(c, fakes.FakeLabeller(), fakes.FakeSimulator(), np.random.default_rng(0)) for c in comps]
        det = np.mean([r["cls"][str(lib.PRIMARY)] in ("two", "many") for r in res])
        self.assertGreaterEqual(det, 0.8)

    def test_composite_max_semantics(self):
        a, b = build_poles(n_each=2)[:2]
        c = lib.composite(a, b)
        np.testing.assert_array_equal(flat(c.assess), np.maximum(flat(a.assess), flat(b.assess)))


if __name__ == "__main__":
    unittest.main()
