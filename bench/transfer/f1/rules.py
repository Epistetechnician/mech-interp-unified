"""Label rules L0-L5 from bench/transfer/LABEL_RULES.md. Pure functions, no I/O.

Thresholds appear here only where LABEL_RULES.md section 3 copies them from a
source protocol. Nothing is tuned.
"""
from __future__ import annotations

import re

# --- thresholds copied from source protocols (LABEL_RULES.md section 3) ---
P_MAX = 0.01               # docs 136/137: matched-random p <= 0.01 (tune and assess)
MIN_COMBINED_DROP = 0.1    # docs 141, 200, 201: combined drop >= 0.1 logits
JACCARD_STABLE = 0.5       # doc 202: Jaccard < 0.5 = selection precision-sensitive
R4E_RHO = -0.05            # R4e: layer growth-pass mean rho <= -0.05
R4E_P = 0.01               # R4e: p <= 0.01
R4E_MIN_NEURONS = 30       # R4e: >= 30 qualifying neurons
R4E_LAYERS_NEEDED = 2      # R4e: growth-pass in >= 2 of 3 layers

LABELS = ("replicated", "failed", "not_tested")


class DerivationError(ValueError):
    pass


def decide(base_class, arm_status, l2=None, verdict=None, no_rule=False):
    """Apply L5, L0, L1, L2, L4, L3 in order.

    l2: (source_rule_id, met: bool) when a source replication rule governs the arm.
    verdict: 'PASS'|'FAIL' semantic outcome for L3 (already mapped by the caller from
             the source status word), used only when no L2 rule governs.
    no_rule: True when the arm produced only descriptive numbers (L3, no_preregistered_rule).
    Returns (label, rule, reason).
    """
    if base_class != "supported":
        return "not_tested", "L5", "base_not_established"
    if arm_status == "not_run":
        return "not_tested", "L0", "no_arm_run"
    if arm_status == "closed_before_measurement":
        return "not_tested", "L1", "arm_closed_before_measurement"
    if arm_status == "inconclusive":
        return "not_tested", "L1", "arm_inconclusive_by_source_rule"
    if l2 is not None:
        sid, met = l2
        return ("replicated", f"L2:{sid}", "rule_met") if met else ("failed", f"L2:{sid}", "rule_not_met")
    if arm_status == "completed_instrument_invalid":
        return "not_tested", "L4", "instrument_invalid_no_rule"
    if no_rule:
        return "not_tested", "L3", "no_preregistered_rule"
    if verdict == "PASS":
        return "replicated", "L3", "rule_met"
    if verdict == "FAIL":
        return "failed", "L3", "rule_not_met"
    raise DerivationError(f"L3 needs verdict PASS/FAIL, got {verdict!r}")


def d201_supported(sing, plur, allv, ctl_sing, ctl_plur):
    """Doc 201 step 4: both directional drops exceed control maxima, combined drop >= 0.1."""
    return bool(sing > ctl_sing and plur > ctl_plur and allv >= MIN_COMBINED_DROP)


def jaccard(overlap, n_a, n_b):
    union = n_a + n_b - overlap
    if union <= 0:
        raise DerivationError("empty union")
    return overlap / union


def r4e_layer_pass(layer):
    return bool(layer["mean_rho"] <= R4E_RHO and layer["p"] <= R4E_P
                and layer["qualifying_neurons"] >= R4E_MIN_NEURONS)


def r4e_arm_pass(layers):
    n = sum(1 for k in sorted(layers) if r4e_layer_pass(layers[k]))
    return n >= R4E_LAYERS_NEEDED, n


# --- evaluators: take the named evidence values, return a result dict ---

def _res(base_class, base_verdict, arm_status, arm_verdict, tgt_gate, label_tuple, inputs):
    label, rule, reason = label_tuple
    return {"base_class": base_class, "base_verdict": base_verdict, "arm_status": arm_status,
            "arm_verdict": arm_verdict, "tgt_gate": tgt_gate,
            "label": label, "rule": rule, "reason": reason, "inputs": inputs}


def ev_astral_v54r(v):
    base_class = ("supported" if (v["base_status"] == "Candidate" and v["base_gate"] == "InstrumentValid")
                  else "inconclusive" if v["base_gate"] != "InstrumentValid" else "not_supported")
    arm_status = "completed" if v["tgt_gate"] == "InstrumentValid" else "completed_instrument_invalid"
    met = bool(v["tgt_status"] == "Candidate" and v["tgt_gate"] == "InstrumentValid"
               and v["tgt_p_tune"] <= P_MAX and v["tgt_p_assess"] <= P_MAX and v["tgt_spec_pass"] is True)
    inputs = {k: v[k] for k in ("base_status", "base_gate", "tgt_status", "tgt_gate",
                                "tgt_p_tune", "tgt_p_assess", "tgt_spec_pass")}
    inputs["p_max"] = P_MAX
    return _res(base_class, f"{v['base_status']}/{v['base_gate']}", arm_status, v["tgt_status"], v["tgt_gate"],
                decide(base_class, arm_status, l2=("S-V54R", met)), inputs)


def ev_d201_task(v):
    """S-D201, or S-R6 when a target gate value is supplied."""
    sup = d201_supported(v["tgt_sing"], v["tgt_plur"], v["tgt_all"], v["tgt_ctl_sing"], v["tgt_ctl_plur"])
    if sup != bool(v["tgt_flag"]):
        raise DerivationError("recomputed S-D201 disagrees with recorded supported flag")
    gate = v.get("tgt_gate")
    if gate is None:
        sid, met = "S-D201", sup
    else:
        sid, met = "S-R6", bool(sup and gate == "InstrumentValid")
    base_class = "supported" if v["base_flag"] is True else "not_supported"
    inputs = {k: v[k] for k in ("tgt_sing", "tgt_plur", "tgt_all", "tgt_ctl_sing", "tgt_ctl_plur", "tgt_flag")}
    inputs["recomputed_supported"] = sup
    inputs["min_combined_drop"] = MIN_COMBINED_DROP
    inputs["base_flag"] = v["base_flag"]
    if gate is not None:
        inputs["tgt_gate"] = gate
    return _res(base_class, "supported" if v["base_flag"] else "not_supported", "completed",
                "supported" if sup else "not_supported", gate or "n/a",
                decide(base_class, "completed", l2=(sid, met)), inputs)


def ev_jaccard(v):
    n_base = len([x for x in re.split(r"[,\s]+", v["base_units"].strip()) if x])
    n_tgt = len(v["tgt_units"])
    j = jaccard(v["overlap"], n_base, n_tgt)
    met = j >= JACCARD_STABLE
    base_class = "supported" if v["base_flag"] is True else "not_supported"
    inputs = {"overlap": v["overlap"], "n_base": n_base, "n_target": n_tgt, "jaccard": round(j, 6),
              "jaccard_stable_at": JACCARD_STABLE, "base_flag": v["base_flag"]}
    return _res(base_class, "supported" if v["base_flag"] else "not_supported", "completed",
                "stable" if met else "precision_sensitive", "n/a",
                decide(base_class, "completed", l2=("S-D202J", met)), inputs)


def ev_equality(v):
    base_ok = bool(v["base_ok"])  # regex evidence: captured text or True
    base_class = "supported" if base_ok else "not_supported"
    met = not bool(v["mismatch"])
    return _res(base_class, "supported" if base_ok else "not_supported", "completed",
                "selection_equal" if met else "selection_mismatch", "n/a",
                decide(base_class, "completed", l2=("S-D200EQ", met)),
                {"base_ok": base_ok, "mismatch_recorded": bool(v["mismatch"])})


def ev_closed(v):
    base_class = "supported" if v["base_flag"] is True else "not_supported"
    return _res(base_class, "supported" if v["base_flag"] else "not_supported", "closed_before_measurement",
                "closed_no_result", "n/a", decide(base_class, "closed_before_measurement"),
                {"base_flag": v["base_flag"], "closure_recorded": bool(v["closure"])})


def ev_overlap_no_rule(v):
    base_class = ("supported" if (v["base_status"] == "Candidate" and v["base_gate"] == "InstrumentValid")
                  else "not_supported")
    return _res(base_class, f"{v['base_status']}/{v['base_gate']}", "completed", "descriptive_overlap_only", "n/a",
                decide(base_class, "completed", no_rule=True),
                {"overlap": v["overlap"], "n_candidates": int(v["n_candidates"]),
                 "base_status": v["base_status"], "base_gate": v["base_gate"]})


def ev_reflexive(v):
    met = bool(v["effect"] > v["ctl_max"] and v["effect"] >= MIN_COMBINED_DROP)
    if met != bool(v["tgt_flag"]):
        raise DerivationError("recomputed S-D141R disagrees with recorded transfer_supported")
    base_class = "supported" if v["base_flag"] is True else "not_supported"
    return _res(base_class, "supported" if v["base_flag"] else "not_supported", "completed",
                "transfer_supported" if met else "transfer_not_supported", "n/a",
                decide(base_class, "completed", l2=("S-D141R", met)),
                {"effect": v["effect"], "ctl_max": v["ctl_max"], "tgt_flag": v["tgt_flag"],
                 "min_combined_drop": MIN_COMBINED_DROP, "base_flag": v["base_flag"]})


def ev_no_arm(v):
    base_class = "supported" if v["base_flag"] is True else "not_supported"
    return _res(base_class, "supported" if v["base_flag"] else "not_supported", "not_run", "none", "n/a",
                decide(base_class, "not_run"),
                {"base_flag": v["base_flag"], "absence_asserted": bool(v["absence"]),
                 "extra_anchor": bool(v.get("anchor", True))})


def ev_v49_arm(v):
    base_class = ("supported" if (v["base_status"] == "Candidate" and v["base_gate"] == "InstrumentValid")
                  else "inconclusive")
    arm_status = "completed" if v["tgt_gate"] == "InstrumentValid" else "completed_instrument_invalid"
    if v["tgt_status"] == "Candidate":
        verdict = "PASS"
    elif v["tgt_status"] == "NoCandidate":
        verdict = "FAIL"
    else:
        raise DerivationError(f"unexpected V49 status {v['tgt_status']}")
    return _res(base_class, f"{v['base_status']}/{v['base_gate']}", arm_status, v["tgt_status"], v["tgt_gate"],
                decide(base_class, arm_status, verdict=verdict),
                {k: v[k] for k in ("base_status", "base_gate", "tgt_status", "tgt_gate")})


def ev_r4e_arm(v):
    b_pass, b_n = r4e_arm_pass(v["base_layers"])
    t_pass, t_n = r4e_arm_pass(v["tgt_layers"])
    base_class = "supported" if b_pass else "not_supported"
    return _res(base_class, f"growth_pass_layers={b_n}/{len(v['base_layers'])}", "completed",
                f"growth_pass_layers={t_n}/{len(v['tgt_layers'])}", "n/a",
                decide(base_class, "completed", l2=("S-R4E", t_pass)),
                {"base_layers_passing": b_n, "tgt_layers_passing": t_n,
                 "tgt_layer_mean_rho": {k: v["tgt_layers"][k]["mean_rho"] for k in sorted(v["tgt_layers"])},
                 "tgt_layer_p": {k: v["tgt_layers"][k]["p"] for k in sorted(v["tgt_layers"])},
                 "tgt_layer_qualifying": {k: v["tgt_layers"][k]["qualifying_neurons"] for k in sorted(v["tgt_layers"])},
                 "thresholds": {"rho": R4E_RHO, "p": R4E_P, "min_neurons": R4E_MIN_NEURONS,
                                "layers_needed": R4E_LAYERS_NEEDED}})


def ev_theseus_arm(v):
    base_class = "supported" if v["base_ok"] is True else "not_supported"
    st = v["tgt_status"]
    if st == "PASS":
        arm_status, verdict = "completed", "PASS"
    elif st == "FAIL":
        arm_status, verdict = "completed", "FAIL"
    elif st in ("INCOMPLETE", "INCONCLUSIVE"):
        arm_status, verdict = "inconclusive", None
    else:
        raise DerivationError(f"unexpected theseus status {st}")
    inputs = {"base_ok": v["base_ok"], "tgt_status": st}
    for k in ("all_cases_pass", "case_level_all_pass", "resource_blockers"):
        if k in v:
            inputs[k] = v[k]
    return _res(base_class, "PASS" if v["base_ok"] else "unknown", arm_status, st, "n/a",
                decide(base_class, arm_status, verdict=verdict), inputs)


def ev_decision_read(v):
    """jspace: base is an inconclusive pre-registered read -> L5."""
    b = v["base_read"]
    base_class = "supported" if b.startswith(("H_route", "H_both", "H_bypass")) else "inconclusive"
    return _res(base_class, b, "completed", v["tgt_read"], "n/a",
                decide(base_class, "completed", verdict=None if base_class != "supported" else "FAIL"),
                {"base_read": b, "tgt_read": v["tgt_read"], "additivity_ok_base": v["base_additivity_ok"],
                 "additivity_ok_target": v["tgt_additivity_ok"]})


def ev_status_pair(v):
    """V50 scale pair: both runs InstrumentNotValid -> base not established (L5)."""
    base_class = "supported" if v["base_status"] in ("Candidate",) else "inconclusive"
    return _res(base_class, v["base_status"], "completed_instrument_invalid", v["tgt_status"], "InstrumentNotValid",
                decide(base_class, "completed_instrument_invalid"),
                {"base_status": v["base_status"], "tgt_status": v["tgt_status"],
                 "base_gate_fails": v["base_fails"], "tgt_gate_fails": v["tgt_fails"]})


EVALUATORS = {
    "astral_v54r": ev_astral_v54r,
    "d201_task": ev_d201_task,
    "jaccard": ev_jaccard,
    "equality": ev_equality,
    "closed": ev_closed,
    "overlap_no_rule": ev_overlap_no_rule,
    "reflexive": ev_reflexive,
    "no_arm": ev_no_arm,
    "v49_arm": ev_v49_arm,
    "r4e_arm": ev_r4e_arm,
    "theseus_arm": ev_theseus_arm,
    "decision_read": ev_decision_read,
    "status_pair": ev_status_pair,
}


def evaluate(kind, vals):
    return EVALUATORS[kind](vals)
