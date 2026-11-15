"""Phase-2 flaw classes build, are deterministic, and their defects are instrument-side only."""
from audit.core import Probe
from audit.flaws import PHASE2_CLASSES

DEV = 100123


def test_build_and_determinism():
    for mod in PHASE2_CLASSES:
        for v in mod.VARIANTS:
            for arm in ("bad", "good"):
                a, b = mod.build(arm, v, DEV), mod.build(arm, v, DEV)
                assert a.flawed == (arm == "bad")
                assert a.report().claim == b.report().claim
                assert list(a.report().selected) == list(b.report().selected)


def test_plain_controls_remove_control_defects():
    from audit.flaws import unmatched_controls as C
    for v in C.VARIANTS:
        bad, good = C.build("bad", v, DEV), C.build("good", v, DEV)
        rb = bad.run(Probe(control_seed=5, plain_controls=True))
        rg = good.run(Probe(control_seed=5, plain_controls=True))
        # the good twin's item-order shuffle changes float summation order, so compare to rounding error
        assert abs(rb.ctrl["all"] - rg.ctrl["all"]).max() < 1e-12


def test_underpowered_bad_is_wrong_null():
    from audit.flaws import underpowered_null as U
    c = U.build("bad", "partial_ablation", DEV)
    assert c.report().claim == "NoCandidate" and c.claim_correct is False
