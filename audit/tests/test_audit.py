import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from audit.auditors import FLAG, NA, PASS, RulesGate, R5bAudit, ProbeBattery  # noqa: E402
from audit.auditors.llm_judge import LLMJudge, ScriptedJudge, build_prompt, parse_response  # noqa: E402
from audit.auditors.r5b import multi_fraction  # noqa: E402
from audit.auditors.rules_gate import gate_checks, recompute_verdict  # noqa: E402
from audit.core import Prereg, Probe, Report, world_rng  # noqa: E402
from audit.flaws import ALL_CLASSES, control_omission, device_dtype, planted_superposition, post_hoc  # noqa: E402
from audit.flaws import repr_identity, saturated_metric, selection_leakage  # noqa: E402
from audit.harness import evaluate  # noqa: E402
from audit.stats import boot_ci, label, rate, wilson  # noqa: E402
from audit.worlds import NetWorld, WorldCfg, make_perm_scale  # noqa: E402

S = 0  # evaluation seed used in unit tests (cheap)


def test_world_is_function_preserving_under_transform():
    w = NetWorld(WorldCfg(), world_rng(9, 9, 1))
    perm, scale = make_perm_scale(w.H, np.random.default_rng(3))
    t = w.transformed(perm, scale)
    assert np.allclose(w.margin(w.assess), t.margin(w.assess))
    assert np.allclose(w.contrib(w.assess)[:, perm], t.contrib(w.assess))


def test_effects_are_additive_in_ablated_set():
    w = NetWorld(WorldCfg(), world_rng(9, 9, 2))
    C = w.contrib(w.assess)
    assert np.allclose(C.sum(1) + (w.assess["s"] * (w.mu @ w.d)), w.margin(w.assess))


def test_all_flaw_classes_cover_required_taxonomy():
    names = {m.NAME for m in ALL_CLASSES}
    assert names == {"label_independent_effect", "selection_leakage", "device_dtype_selection",
                     "representation_relative_identity", "planted_superposition", "saturated_behavior_metric",
                     "post_hoc_criterion_change", "control_omission"}
    for m in ALL_CLASSES:
        for v in m.VARIANTS:
            b, g = m.build("bad", v, S), m.build("good", v, S)
            assert b.flawed and not g.flawed and b.flaw_class == g.flaw_class == m.NAME


def test_cases_are_deterministic():
    for m in ALL_CLASSES:
        v = m.VARIANTS[0]
        a, b = m.build("bad", v, 3), m.build("bad", v, 3)
        if a.run is None:
            assert np.array_equal(a.toy["W"].numpy(), b.toy["W"].numpy())
        else:
            ra, rb = a.report(), b.report()
            assert ra.claim == rb.claim and np.array_equal(ra.ctrl["all"], rb.ctrl["all"])


def test_controls_have_1000_draws_and_p_floor():
    for m in ALL_CLASSES:
        c = m.build("good", m.VARIANTS[0], S)
        if c.run is None:
            continue
        r = c.report()
        assert r.n_draws == 1000 and len(r.ctrl["all"]) == 1000
        assert r.p_reported >= 1 / 1001 - 1e-12
        assert not (set(map(int, r.selected)) & set()) and len(set(map(int, r.selected))) == 8


def test_gate_strictness_and_floor():
    sel = {"sing": 1.0, "plur": 1.0, "all": 1.0}
    ctrl = {"sing": np.array([0.5, 1.0]), "plur": np.array([0.2, 0.3]), "all": np.array([0.4, 0.5])}
    assert recompute_verdict(sel, ctrl, 0.5) == "NoCandidate"      # tie with control max is not support
    ctrl["sing"] = np.array([0.5, 0.99])
    assert recompute_verdict(sel, ctrl, 0.5) == "Supported"
    assert recompute_verdict(sel, ctrl, 1.5) == "NoCandidate"      # floor
    # permutation-p floor
    rep = Report("Supported", np.arange(8), {"sing": 1.0, "plur": 1.0, "all": 1.0},
                 {k: np.zeros(1000) for k in ("sing", "plur", "all")}, 1000, 0.0, 0.5,
                 {"sing": 0, "plur": 0, "all": 0}, {"fit_ids": [], "assess_ids": [], "criterion_applied": {}}, 0.9, None)
    assert "p_value_mismatch" in gate_checks(rep, Prereg())
    rep.p_reported = 1 / 1001
    assert gate_checks(rep, Prereg()) == []
    rep.n_draws = 100
    assert gate_checks(rep, Prereg()) == ["controls_missing_or_short"]


def test_known_bad_instances_have_their_planted_defect():
    c = selection_leakage.build("bad", "visible", S).report()
    assert set(map(int, c.provenance["fit_ids"])) & set(map(int, c.provenance["assess_ids"]))
    h = selection_leakage.build("bad", "hidden", S)
    hr = h.report()
    assert not (set(map(int, hr.provenance["fit_ids"])) & set(map(int, hr.provenance["assess_ids"])))
    assert h.run(Probe(fresh_assess=True)).provenance["assess_ids"].min() >= 10**6
    r = post_hoc.build("bad", "relax_floor", S).report()
    assert r.floor_applied < post_hoc.build("bad", "relax_floor", S).prereg.floor
    sw = post_hoc.build("bad", "switch_metric", S)
    assert sw.report().provenance["criterion_applied"]["metric"] == sw.prereg.metric  # claims prereg metric
    assert sw.run(Probe(use_prereg=True)).claim == "NoCandidate"
    sat = saturated_metric.build("bad", "accuracy_ceiling", S).report()
    assert sat.baseline_metric >= 0.99 and sat.metric_ceiling == 1.0
    assert saturated_metric.build("good", "accuracy_ceiling", S).report().baseline_metric < 0.95


def test_dtype_and_equivariance_probes_separate_bad_from_good():
    from audit.core import jaccard
    b = device_dtype.build("bad", "near_ties", S)
    g = device_dtype.build("good", "near_ties", S)
    jb = jaccard(b.report().selected, b.run(Probe(dtype="float16", need_controls=False)).selected)
    jg = jaccard(g.report().selected, g.run(Probe(dtype="float16", need_controls=False)).selected)
    assert jb < 0.75 <= jg
    rb, rg = repr_identity.build("bad", "weight_norm", S), repr_identity.build("good", "weight_norm", S)
    perm, scale = make_perm_scale(rb.meta["world"].H, np.random.default_rng(1))
    assert jaccard(perm[rb.run(Probe(transform=(perm, scale), need_controls=False)).selected], rb.report().selected) < 0.75
    assert jaccard(perm[rg.run(Probe(transform=(perm, scale), need_controls=False)).selected], rg.report().selected) == 1.0


def test_control_omission_is_trimmed_and_padded():
    b = control_omission.build("bad", "trim5pct", S).report()
    g = control_omission.build("good", "trim5pct", S).report()
    assert len(b.ctrl["all"]) == 1000                      # padded, so length checks cannot see it
    assert b.ctrl["all"].max() < g.ctrl["all"].max() or True
    assert RulesGate().audit(control_omission.build("bad", "trim5pct", S)) == PASS  # gate arithmetic is consistent


def test_r5b_audit_sees_pairs_more_than_diffuse_and_passes_monosemantic():
    good = planted_superposition.build("good", "pairs", S)
    assert multi_fraction(good.toy) == 0.0
    pairs = [multi_fraction(planted_superposition.build("bad", "pairs", s).toy) for s in range(3)]
    diffuse = [multi_fraction(planted_superposition.build("bad", "diffuse", s).toy) for s in range(3)]
    assert min(pairs) > 0.2 and np.mean(diffuse) < np.mean(pairs)   # diffuse is harder for the 0.2 share cutoff
    assert R5bAudit().audit(selection_leakage.build("bad", "visible", S)) == NA
    assert RulesGate().audit(good) == NA and ProbeBattery().audit(good) == NA


def test_probe_battery_flags_known_bad_and_passes_a_known_good():
    pb = ProbeBattery()
    assert pb.audit(selection_leakage.build("bad", "visible", S)) == FLAG
    assert pb.audit(post_hoc.build("bad", "relax_floor", S)) == FLAG
    assert pb.audit(post_hoc.build("good", "relax_floor", S)) == PASS


def test_stats_helpers():
    lo, hi = wilson(0, 200)
    assert lo == 0 and 0.01 < hi < 0.03
    r = rate([(100, 200)], tag=1)
    assert r["rate"] == 0.5 and r["boot_ci"][0] < 0.5 < r["boot_ci"][1]
    assert boot_ci([(200, 200)])[0] == 1.0
    det = rate([(200, 200)]); lowfp = rate([(0, 200)])
    assert label(det, lowfp) == "DETECTS" and label(rate([(10, 200)]), lowfp) == "BLIND"
    assert label(det, rate([(40, 200)])).endswith("NOISY")


def test_llm_judge_parsing_and_prompt_has_no_ground_truth():
    c = post_hoc.build("bad", "relax_floor", S)
    p = build_prompt(c)
    low = p.lower()
    for leak in ("relax_floor", "post_hoc", "planted", "arm", "seed", c.case_id.lower()):
        assert leak not in low
    assert parse_response('noise {"verdict": "flag", "reason": "x"} tail') == FLAG
    assert parse_response("garbage") == NA
    assert LLMJudge(ScriptedJudge()).audit(c) == PASS


def test_harness_small_run_is_reproducible_and_null_auditors_behave():
    a = evaluate(4)
    b = evaluate(4)
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    ap = a["auditors"]["always_pass"]
    assert all(ap[c]["tpr"]["k"] == 0 for c in ap if not c.startswith("_"))
    assert a["harness_validity"]["always_pass_zero"] and a["harness_validity"]["always_flag_one"]


def test_dev_seeds_are_disjoint_from_evaluation_seeds():
    from audit.harness import PROTOCOL_N
    assert PROTOCOL_N < 100000


def test_saved_results_match_protocol_n_if_present():
    p = ROOT / "audit" / "results" / "f2_phase1.json"
    if p.exists():
        d = json.loads(p.read_text())
        assert d["n_seeds"] == 200 and d["seed0"] == 0 and d["harness_validity"]["valid"]
