"""R6: second-model replication of the V61-local agreement-unit pipeline (Qwen3-1.7B-Base, 1000 controls).

Protocol: docs/protocols/R6-second-model-agreement-gate.md (frozen) plus its Addendum 1.
Logic is the V61-local pipeline (select on fit, rank on tune, test on assess) with two protocol-mandated
differences: 1000 matched-random draws instead of 100, and an instrument-validity gate (v3) that must pass
before any T0/L/P effect is read. Pure Python: the backend object does all model work (see qwen3_backend.py).
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "..", "imports", "astral-neuron", "tools")
for d in ("60", "59", "58", "57", "54", "53", "52", "51", "50", "49"):
    sys.path.insert(0, os.path.join(TOOLS, "astral-neuron-level-causal-lane-v" + d))
import v50, v51, v60, pythia_plan  # noqa: E402  (imports are immutable; this repo only reads them)

N_LAYERS = 28  # Qwen3-1.7B-Base num_hidden_layers (checked against the cached config in tests when available)
TARGET_LAYERS = pythia_plan.layers_for(N_LAYERS)  # DEPTHS 0.3/0.55/0.8 as for Pythia: [8, 15, 22]

# Frozen before any data. Changing a value changes FROZEN6_DIGEST and fails the tests and the run.
FROZEN6 = {
    "model": "Qwen/Qwen3-1.7B-Base", "n_layers": N_LAYERS, "target_layers": TARGET_LAYERS,
    "k_per_layer": 32, "swap_size": 8, "ctl_draws": 1000, "gate_draws": 1000, "min_effect": 0.1,
    "donor_seed": 6060, "dtype": "float32", "stage_cap_s": 8 * 3600, "determinism_repeat": True,
}
FROZEN6_DIGEST = v50.digest(FROZEN6)

OUTCOMES = ("Replicates", "PartiallyReplicates", "DoesNotReplicate", "InstrumentNotValid", "NotCompleted")


class StageTimeout(Exception):
    pass


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def classify(gate_valid, t0, l, p):
    """Frozen decision rule (protocol section 3). Inputs are the per-task `supported` flags."""
    if not gate_valid:
        return "InstrumentNotValid"
    if not t0:
        return "DoesNotReplicate"
    return "Replicates" if (l and p) else "PartiallyReplicates"


def run_r6(backend, cfg, log=None, clock=time.monotonic, checkpoint=None):
    if v50.digest(FROZEN6) != FROZEN6_DIGEST:
        raise v50.V50Error("R6 frozen values changed")
    v50.check_frozen()
    if list(cfg["target_layers"]) != FROZEN6["target_layers"]:
        raise v50.V50Error("target layers differ from the frozen plan")
    say = log or (lambda *a: None)
    ckpt = checkpoint or (lambda o: None)
    cap = FROZEN6["stage_cap_s"]
    out = {"frozen6_digest": FROZEN6_DIGEST, "ceiling": v50.CEILING, "stages": {}}

    class Stage:
        def __init__(self, name):
            self.name, self.t0 = name, clock()

        def tick(self):
            if clock() - self.t0 > cap:
                raise StageTimeout(self.name)

        def done(self):
            out["stages"][self.name] = round(clock() - self.t0, 1)

    cur = None
    try:
        sp = backend.split()
        layer_of = backend.layers()
        # ---- stage 1: select on fit (twice: determinism report), lock before any tune/assess read
        cur = Stage("select")
        scores = backend.scores(sp["fit"])
        k_total = FROZEN6["k_per_layer"] * len(cfg["target_layers"])
        cand_units = v50.select_per_layer(scores, layer_of, cfg["target_layers"], k_total)
        if FROZEN6["determinism_repeat"]:
            again = v50.select_per_layer(backend.scores(sp["fit"]), layer_of, cfg["target_layers"], k_total)
            out["selection_repeat_jaccard"] = jaccard(cand_units, again)
        dirs = {u: (1 if scores[u] >= 0 else -1) for u in cand_units}
        lock = v50.make_lock(cand_units, dirs, {k: sp[k] for k in ("fit", "tune", "assess")}, cfg["model_id"],
                             cfg["corpus_digest"], cfg["code_commit"])
        backend.mark_locked(lock)
        out["candidates_digest"] = v50.digest(cand_units)
        out["lock"] = lock
        say("candidates", len(cand_units))
        cur.done()

        # ---- stage 2: instrument gate v3 on the candidate set, tune split, 1000 matched draws
        cur = Stage("gate")
        mag = backend.magnitude()
        pool = v50.matched_pool(mag, cand_units, layer_of)
        gctl = v50.draw_controls(pool, cand_units, FROZEN6["gate_draws"], cfg["seed"] + 2)
        tune = sp["tune"]
        eff_m = backend.effect(cand_units, tune, "margin")
        eff_l = backend.effect(cand_units, tune, "logprob")
        cm, cl = [], []
        for i, c in enumerate(gctl):
            cm.append(backend.effect(c, tune, "margin"))
            cl.append(backend.effect(c, tune, "logprob"))
            if i % 50 == 49:
                cur.tick()
                say("gate controls", i + 1)
        gate = v51.instrument_gate_v3(backend, cand_units, scores, tune,
                                      {"margin": v50.one_sided_p(eff_m, cm), "logprob": v50.one_sided_p(eff_l, cl)},
                                      cfg.get("alpha", 0.05))
        out["gate"] = gate
        cur.done()
        ckpt(out)
        if gate["status"] != "InstrumentValid":
            out["outcome"] = "InstrumentNotValid"
            return out  # no effect is read as evidence

        # ---- stage 3: rank on tune, lock the swap set, matched controls for the assess tests
        cur = Stage("rank")
        single = {u: backend.effect([u], tune, "margin") for u in cand_units}
        swap = sorted(sorted(cand_units, key=lambda u: (-single[u], u))[:FROZEN6["swap_size"]])
        out["swap_units"] = swap
        out["swap_lock_digest"] = v50.digest({"lock": lock, "swap": swap, "tune_single": {str(u): single[u] for u in swap}})
        cset = set(cand_units)
        cand = {u: ([c for c in v if c not in cset] or v) for u, v in v50.matched_pool(mag, swap, layer_of).items()}
        ctl = v50.draw_controls(cand, swap, 1000, cfg["seed"] + 1)[:FROZEN6["ctl_draws"]]
        allu = sorted(set(swap) | {u for c in ctl for u in c})
        fs = backend.unit_stats(allu, sp["fit"])
        means = {u: (fs[u]["act_sing"], fs[u]["act_plur"]) for u in allu}  # dla fields are never read (RMSNorm)
        docs = sp["assess"]
        cur.done()
        ckpt(out)

        floor = FROZEN6["min_effect"]

        def task(name, fn):
            st = Stage(name)
            sel = fn(swap)
            cs = []
            for i, c in enumerate(ctl):
                cs.append(fn(c))
                if i % 50 == 49:
                    st.tick()
                    say(name, "controls", i + 1)
            r, cmax = v60._summ(sel, cs)
            r["supported"] = v60._supported(sel, cmax, floor)
            st.done()
            return r

        out["t0_all_position_swap"] = task("T0", lambda us: backend.swap_effect(us, docs, means))
        ckpt(out)
        out["l_last_position_swap"] = task("L", lambda us: backend.swap_effect_last(us, docs, means))
        ckpt(out)
        classes = backend.item_classes(docs)
        capt = backend.capture_last(allu, docs)
        don = v60.donors(classes, docs, FROZEN6["donor_seed"])
        vals = lambda us, dm: {u: {i: capt[u][dm[i]] for i in docs} for u in us}
        out["p_donor_patch"] = task("P", lambda us: backend.patch_last(us, docs, vals(us, don)))
        same = v60.donors(classes, docs, FROZEN6["donor_seed"] + 1, same=True)
        out["p_donor_patch"]["same_class_donor_diagnostic"] = backend.patch_last(swap, docs, vals(swap, same))
        ckpt(out)
        out["outcome"] = classify(True, out["t0_all_position_swap"]["supported"],
                                  out["l_last_position_swap"]["supported"], out["p_donor_patch"]["supported"])
    except StageTimeout as e:
        out["outcome"] = "NotCompleted"
        out["timed_out_stage"] = str(e)
    return out
