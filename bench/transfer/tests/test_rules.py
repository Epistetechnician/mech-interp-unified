import pytest
from f1 import rules


def test_decide_order():
    assert rules.decide("inconclusive", "not_run")[:2] == ("not_tested", "L5")
    assert rules.decide("supported", "not_run")[:2] == ("not_tested", "L0")
    assert rules.decide("supported", "closed_before_measurement")[:2] == ("not_tested", "L1")
    assert rules.decide("supported", "inconclusive")[2] == "arm_inconclusive_by_source_rule"
    # L2 beats L4 (V55 case)
    assert rules.decide("supported", "completed_instrument_invalid", l2=("S-V54R", False)) == ("failed", "L2:S-V54R", "rule_not_met")
    assert rules.decide("supported", "completed_instrument_invalid")[:2] == ("not_tested", "L4")
    assert rules.decide("supported", "completed", verdict="PASS")[0] == "replicated"
    assert rules.decide("supported", "completed", verdict="FAIL")[0] == "failed"
    assert rules.decide("supported", "completed", no_rule=True)[2] == "no_preregistered_rule"
    with pytest.raises(rules.DerivationError):
        rules.decide("supported", "completed")


def test_d201_threshold_edges():
    assert rules.d201_supported(0.2, 0.2, 0.1, 0.1, 0.1)
    assert not rules.d201_supported(0.2, 0.2, 0.0999, 0.1, 0.1)   # combined floor
    assert not rules.d201_supported(0.1, 0.2, 0.2, 0.1, 0.1)      # not strictly above control max
    assert not rules.d201_supported(0.2, 0.05, 0.2, 0.1, 0.1)


def test_jaccard():
    assert rules.jaccard(7, 7, 8) == pytest.approx(7 / 8)
    assert rules.jaccard(0, 4, 4) == 0
    with pytest.raises(rules.DerivationError):
        rules.jaccard(0, 0, 0)


def test_r4e_edges():
    ok = {"mean_rho": -0.05, "p": 0.01, "qualifying_neurons": 30}
    assert rules.r4e_layer_pass(ok)
    assert not rules.r4e_layer_pass({**ok, "mean_rho": -0.0499})
    assert not rules.r4e_layer_pass({**ok, "p": 0.0101})
    assert not rules.r4e_layer_pass({**ok, "qualifying_neurons": 29})
    assert rules.r4e_arm_pass({"1": ok, "2": ok, "3": {**ok, "p": 0.5}})[0]
    assert not rules.r4e_arm_pass({"1": ok, "2": {**ok, "p": 0.5}, "3": {**ok, "p": 0.5}})[0]


def test_v54r_nonreplication_when_floor_failed():
    v = dict(base_status="Candidate", base_gate="InstrumentValid", tgt_status="InstrumentNotValid",
             tgt_gate="InstrumentNotValid", tgt_p_tune=0.001, tgt_p_assess=0.001, tgt_spec_pass=False)
    r = rules.ev_astral_v54r(v)
    assert (r["label"], r["rule"]) == ("failed", "L2:S-V54R")
    v.update(tgt_status="Candidate", tgt_gate="InstrumentValid", tgt_spec_pass=True)
    assert rules.ev_astral_v54r(v)["label"] == "replicated"


def test_d201_flag_mismatch_raises():
    v = dict(tgt_sing=0.01, tgt_plur=0.2, tgt_all=0.2, tgt_ctl_sing=0.1, tgt_ctl_plur=0.1, tgt_flag=True, base_flag=True)
    with pytest.raises(rules.DerivationError):
        rules.ev_d201_task(v)


def test_r6_needs_valid_gate():
    v = dict(tgt_sing=0.3, tgt_plur=0.3, tgt_all=0.3, tgt_ctl_sing=0.1, tgt_ctl_plur=0.1, tgt_flag=True,
             base_flag=True, tgt_gate="InstrumentNotValid")
    assert rules.ev_d201_task(v)["label"] == "failed"
