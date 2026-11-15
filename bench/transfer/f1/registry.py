"""Registry of the 36 items declared in LABEL_RULES.md section 4.

Static fields here (model, device, ...) are copied from the cited source text; where a
source does not state a condition the value is the string "unrecorded".
Labels are never written here: they are computed by f1.rules from the evidence.
"""
from __future__ import annotations

A = "imports/astral-neuron/runs"
P = "imports/astral-neuron/protocols"
T = "imports/theseus-lab/docs"
J = "json_pointer"
R = "regex"
NOC = "unrecorded"


def E(name, role, root, file, kind, pointer):
    return {"name": name, "role": role, "root": root, "file": file, "kind": kind, "pointer": pointer}


def cond(model, size, size_b, device, dtype, checkpoint, corpus, gate):
    return {"model": model, "size": size, "size_b": size_b, "device": device, "dtype": dtype,
            "checkpoint": checkpoint, "corpus": corpus, "instrument_gate_status": gate}


H100 = "H100 (GiveMeANode node v51-full-r1, image pytorch-2.13-cuda12.9)"
PY28 = ("Pythia-2.8b (GPT-NeoX)", "2.8B", 2.8, "EleutherAI/pythia-2.8b rev 2a259cdd96a4beb1cdf467512e3904197345f6a9")
BF16_DOC200 = "bfloat16 (doc 200 addendum 1, H100 runs)"
V54_CORPUS = "35 fresh Project Gutenberg books, 2400 items, manifest digest 59c514f0c5a89a1cf046f94a64e4abbe1254158bf3e992f37fbcf5e754a33221"
SRC = "OwnerSelfAttestedLocalDevelopment"
LDU = "LocalDevelopmentUnreviewed"

V54_BASE = cond(PY28[0], PY28[1], PY28[2], H100, BF16_DOC200, PY28[3], V54_CORPUS, "InstrumentValid (gate v3)")
V59_BASE = cond(PY28[0], PY28[1], PY28[2], H100, BF16_DOC200, PY28[3], V54_CORPUS + " (assess split, 800 items)",
                "inherited from locked V54 selection (V59 reports no gate of its own)")
V61L = cond(PY28[0], PY28[1], PY28[2], "Apple MPS (M4 Max per doc 202)", "float32", PY28[3], V54_CORPUS + " (rebuilt locally)",
            "not applicable (doc 201 pipeline has no gate step)")

ITEMS = []


def item(**kw):
    ITEMS.append(kw)


# ---------------------------------------------------------------- Astral lane (Pythia)
item(id="astral-v53-v54-corpus", cluster_id="astral-pythia-ablation",
     claim={"text": "Ablating the fit-selected units lowers the subject-verb agreement margin far beyond 1000 matched-random draws and the effect is specific to agreement context (constant-prefix control).",
            "family": "neuron_ablation_effect"},
     base=cond(PY28[0], PY28[1], PY28[2], H100, NOC, PY28[3],
               "36 Project Gutenberg books (V50 list), item-disjoint splits", "InstrumentValid (gate v3)"),
     base_run="V53",
     transfer=dict(run_id="V54", **cond(PY28[0], PY28[1], PY28[2], H100, BF16_DOC200, PY28[3], V54_CORPUS, "InstrumentValid (gate v3)")),
     axes=["corpus"], confounds=["attractor fraction 47.0% vs waiver-regime in V53 (reported, not gated)"],
     kind="astral_v54r", ceiling=SRC,
     ev=[E("base_status", "base", "repo", f"{A}/v53-run/results/out_v53.json", J, "/status"),
         E("base_gate", "base", "repo", f"{A}/v53-run/results/out_v53.json", J, "/gate/status"),
         E("tgt_status", "transfer", "repo", f"{A}/v54-run/results/out_v54.json", J, "/status"),
         E("tgt_gate", "transfer", "repo", f"{A}/v54-run/results/out_v54.json", J, "/gate/status"),
         E("tgt_p_tune", "transfer", "repo", f"{A}/v54-run/results/out_v54.json", J, "/results/tune/p"),
         E("tgt_p_assess", "transfer", "repo", f"{A}/v54-run/results/out_v54.json", J, "/results/assess/p"),
         E("tgt_spec_pass", "transfer", "repo", f"{A}/v54-run/results/out_v54.json", J, "/specificity/pass"),
         E("rule_text", "rule", "repo", f"{P}/136-neuron-level-causal-lane-v54-protocol.md", R,
           r"V54 \"replicates\" only if status Candidate with gate InstrumentValid, matched-random p <= 0\.01 on tune and assess, and specificity pass")],
     notes="Source README for V54 records the gate and specificity z, one model, AI-assisted label audit (1/200 error), same design chosen after V50-V52. Owner self-attested (V53/V54 per docs/INDEX.md).")

item(id="astral-v54-v55-scale", cluster_id="astral-pythia-ablation",
     claim={"text": "Ablating the fit-selected units lowers the subject-verb agreement margin far beyond matched-random units and the effect is specific to agreement context, on a smaller model of the same family.",
            "family": "neuron_ablation_effect"},
     base=V54_BASE, base_run="V54",
     transfer=dict(run_id="V55", **cond("Pythia-1.4b (GPT-NeoX)", "1.4B", 1.4, H100, NOC,
                                        "EleutherAI/pythia-1.4b rev fedc38a16eea3bd36a96b906d78d11d2ce18ed79", V54_CORPUS, "InstrumentNotValid (gate v3: min_effect)")),
     axes=["scale"], confounds=["units re-selected on the 1.4b model's own fit split", "frozen 0.5-logit absolute floor from 2.8b kept"],
     kind="astral_v54r", ceiling=SRC,
     ev=[E("base_status", "base", "repo", f"{A}/v54-run/results/out_v54.json", J, "/status"),
         E("base_gate", "base", "repo", f"{A}/v54-run/results/out_v54.json", J, "/gate/status"),
         E("tgt_status", "transfer", "repo", f"{A}/v55-run/results/out_v55.json", J, "/status"),
         E("tgt_gate", "transfer", "repo", f"{A}/v55-run/results/out_v55.json", J, "/gate/status"),
         E("tgt_p_tune", "transfer", "repo", f"{A}/v55-run/results/out_v55.json", J, "/results/tune/p"),
         E("tgt_p_assess", "transfer", "repo", f"{A}/v55-run/results/out_v55.json", J, "/results/assess/p"),
         E("tgt_spec_pass", "transfer", "repo", f"{A}/v55-run/results/out_v55.json", J, "/specificity/pass"),
         E("rule_text", "rule", "repo", f"{P}/137-neuron-level-causal-lane-v55-protocol.md", R,
           r"Anything else is a non-replication on this model, not retuned")],
     notes="V55 README: selected effect 0.239 logits below the frozen 0.5 floor; matched-random contrast still p=0.001 and constant-control z=14.9 but the frozen gap floor failed. The source protocol calls this a non-replication; the README also says whether the floor suits a smaller model is an open protocol question.")

item(id="astral-v50-scale-1.4b-2.8b", cluster_id="astral-pythia-ablation",
     claim={"text": "The selected-unit ablation effect passes the old instrument gate (gate v1) on Pythia-1.4B, with replication on Pythia-2.8B.",
            "family": "instrument_gate"},
     base=cond("Pythia-1.4b (GPT-NeoX)", "1.4B", 1.4, "H100 (GiveMeANode, V50 receipt)", NOC,
               "EleutherAI/pythia-1.4b (V50 packet)", "36 Project Gutenberg books", "InstrumentNotValid (gate v1: known_bad)"),
     base_run="V50 1.4b",
     transfer=dict(run_id="V50 2.8b", **cond(PY28[0], PY28[1], PY28[2], "H100 (GiveMeANode, V50 receipt)", NOC, PY28[3],
                                             "36 Project Gutenberg books", "InstrumentNotValid (gate v1: known_bad)")),
     axes=["scale"], confounds=[], kind="status_pair", ceiling=LDU,
     ev=[E("base_status", "base", "repo", f"{A}/v50-run/results/out_1.4b.json", J, "/status"),
         E("base_fails", "base", "repo", f"{A}/v50-run/results/out_1.4b.json", J, "/gate/fails"),
         E("tgt_status", "transfer", "repo", f"{A}/v50-run/results/out_2.8b.json", J, "/status"),
         E("tgt_fails", "transfer", "repo", f"{A}/v50-run/results/out_2.8b.json", J, "/gate/fails"),
         E("rule_text", "rule", "repo", f"{P}/132-neuron-level-causal-lane-v50-protocol.md", R,
           r"\*\*Replication\.\*\* The full pipeline runs once on Pythia-1\.4B \(primary\) and once on Pythia-2\.8B")],
     notes="Both V50 runs failed the known-bad control under gate v1 (1.4b known_bad_max 0.028 vs selected 0.189; 2.8b 0.063 vs 0.563). Matched-random p=0.001 at both sizes, but an effect read through a rejected instrument is not counted (AGENTS rule 6).")

item(id="astral-v59-t0-v61local", cluster_id="astral-pythia-swap",
     claim={"text": "Swapping the selected number-selective units to the opposite-class mean at all positions drops the verb margin far beyond matched-random unit sets (T0).",
            "family": "class_swap_sufficiency"},
     base=V59_BASE, base_run="V59",
     transfer=dict(run_id="V61-local", **V61L),
     axes=["device", "dtype"], confounds=["units re-selected end to end (7 of 8 swap units shared with V59)", "100 controls (p floor 0.0099) in both", "post-hoc relative to V54/V58/V59"],
     kind="d201_task", ceiling=LDU,
     ev=[E("base_flag", "base", "repo", f"{A}/v59-run/results/out_v59.json", J, "/swap/sufficiency_supported"),
         E("tgt_sing", "transfer", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, "/t0_all_position_swap/selected/sing"),
         E("tgt_plur", "transfer", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, "/t0_all_position_swap/selected/plur"),
         E("tgt_all", "transfer", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, "/t0_all_position_swap/selected/all"),
         E("tgt_ctl_sing", "transfer", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, "/t0_all_position_swap/ctl_max/sing"),
         E("tgt_ctl_plur", "transfer", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, "/t0_all_position_swap/ctl_max/plur"),
         E("tgt_flag", "transfer", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, "/t0_all_position_swap/supported"),
         E("rule_text", "rule", "repo", f"{P}/201-neuron-level-causal-lane-v61-local-protocol.md", R,
           r"\*\*Supported\*\* \(per task\) iff both directional drops exceed the control maxima and the combined drop >= 0\.1 logits")],
     notes="V59 drop 1.308 logits (control max 0.072, all-position constants, OOD edit); V61-local 1.361 on held-out-selected units.")

item(id="astral-v60-l-v61local", cluster_id="astral-pythia-swap",
     claim={"text": "A last-position-only class swap of the selected units drops the verb margin far beyond matched-random sets (Task L).",
            "family": "class_swap_sufficiency"},
     base=V59_BASE, base_run="V60 Task L",
     transfer=dict(run_id="V61-local", **V61L),
     axes=["device", "dtype"], confounds=["units re-selected end to end", "100 controls"],
     kind="d201_task", ceiling=LDU,
     ev=[E("base_flag", "base", "repo", f"{A}/v60-run/results/out_v60.json", J, "/last_swap/supported"),
         E("tgt_sing", "transfer", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, "/l_last_position_swap/selected/sing"),
         E("tgt_plur", "transfer", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, "/l_last_position_swap/selected/plur"),
         E("tgt_all", "transfer", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, "/l_last_position_swap/selected/all"),
         E("tgt_ctl_sing", "transfer", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, "/l_last_position_swap/ctl_max/sing"),
         E("tgt_ctl_plur", "transfer", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, "/l_last_position_swap/ctl_max/plur"),
         E("tgt_flag", "transfer", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, "/l_last_position_swap/supported"),
         E("rule_text", "rule", "repo", f"{P}/201-neuron-level-causal-lane-v61-local-protocol.md", R,
           r"\*\*Supported\*\* \(per task\) iff both directional drops exceed the control maxima and the combined drop >= 0\.1 logits")],
     notes="V60 H100 Task L drop 1.045; V61-local 1.094.")

item(id="astral-v60-p-v61local", cluster_id="astral-pythia-swap",
     claim={"text": "Patching the selected units' last-position activations from an opposite-class donor sentence drops the verb margin far beyond matched-random sets (Task P).",
            "family": "donor_patch_sufficiency"},
     base=V59_BASE, base_run="V60 Task P",
     transfer=dict(run_id="V61-local", **V61L),
     axes=["device", "dtype"], confounds=["units re-selected end to end", "100 controls", "same-class-donor diagnostic 0.098 (not a gate)"],
     kind="d201_task", ceiling=LDU,
     ev=[E("base_flag", "base", "repo", f"{A}/v60-run/results/out_v60.json", J, "/donor_patch/supported"),
         E("tgt_sing", "transfer", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, "/p_donor_patch/selected/sing"),
         E("tgt_plur", "transfer", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, "/p_donor_patch/selected/plur"),
         E("tgt_all", "transfer", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, "/p_donor_patch/selected/all"),
         E("tgt_ctl_sing", "transfer", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, "/p_donor_patch/ctl_max/sing"),
         E("tgt_ctl_plur", "transfer", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, "/p_donor_patch/ctl_max/plur"),
         E("tgt_flag", "transfer", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, "/p_donor_patch/supported"),
         E("rule_text", "rule", "repo", f"{P}/201-neuron-level-causal-lane-v61-local-protocol.md", R,
           r"\*\*Supported\*\* \(per task\) iff both directional drops exceed the control maxima and the combined drop >= 0\.1 logits")],
     notes="V60 H100 Task P drop 1.398; V61-local 1.446.")

item(id="astral-v59-swapset-v61local", cluster_id="astral-pythia-swap",
     claim={"text": "The set of number-selective swap units chosen by the pipeline is the same set on another device and precision.",
            "family": "unit_selection_stability"},
     base=V59_BASE, base_run="V59 swap set",
     transfer=dict(run_id="V61-local", **V61L),
     axes=["device", "dtype"], confounds=["V59 set has 7 units, V61-local 8 (top 8 by tune effect)", "S-D202J threshold (0.5) is extended from V62 arms to this pair"],
     kind="jaccard", ceiling=LDU,
     ev=[E("base_flag", "base", "repo", f"{A}/v59-run/results/out_v59.json", J, "/swap/sufficiency_supported"),
         E("base_units", "base", "repo", f"{P}/200-neuron-level-causal-lane-v60-protocol.md", R,
           r"swap_units = \[([0-9, ]+)\] \(plural-selective 4 \+ singular-selective 3\)"),
         E("overlap", "transfer", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, "/overlap_with_v59_units"),
         E("tgt_units", "transfer", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, "/swap_units"),
         E("rule_text", "rule", "repo", f"{P}/202-neuron-level-causal-lane-v62-robustness-protocol.md", R,
           r"swap-set Jaccard versus V61-local is below 0\.5 in any arm")],
     notes="V61-local README: 7 of 8 swap units are V59 units (V59's 14134 replaced by 27989).")

item(id="astral-v54-selection-v60local", cluster_id="astral-pythia-swap",
     claim={"text": "A fresh k=32 per-layer reselection on another device and precision equals the locked V54 selection.",
            "family": "unit_selection_stability"},
     base=V59_BASE, base_run="V60 cloud (guard passed)",
     transfer=dict(run_id="V60-local", **cond(PY28[0], PY28[1], PY28[2], "Apple MPS (torch 2.14.1)", "bfloat16", PY28[3], V54_CORPUS + " (rebuilt locally, digest verified)", "not applicable (selection guard)")),
     axes=["device"], confounds=["torch version differs from H100 run"],
     kind="equality", ceiling=LDU,
     ev=[E("base_ok", "base", "repo", f"{A}/v60-run/README.md", R, r"(## Task L: last-position-only swap: SUPPORTED)"),
         E("mismatch", "transfer", "repo", f"{A}/v60-run/README.md", R, r"(reselection of 32 units differed from the locked V54 selection)"),
         E("rule_text", "rule", "repo", f"{P}/200-neuron-level-causal-lane-v60-protocol.md", R,
           r"Run aborts unless a fresh k=32 reselection equals the locked V54 selection")],
     notes="V60-local README: the frozen guard raised V50Error before any swap or patch measurement; closed without retuning.")

item(id="astral-v54-candidates-v61local", cluster_id="astral-pythia-swap",
     claim={"text": "The 96 fit-selected candidate units are the same units on another device and precision.",
            "family": "unit_selection_stability"},
     base=V54_BASE, base_run="V54 selection",
     transfer=dict(run_id="V61-local", **V61L),
     axes=["device", "dtype"], confounds=["V61-local selects on fit and ranks on tune; V54 pipeline differs"],
     kind="overlap_no_rule", ceiling=LDU,
     ev=[E("base_status", "base", "repo", f"{A}/v54-run/results/out_v54.json", J, "/status"),
         E("base_gate", "base", "repo", f"{A}/v54-run/results/out_v54.json", J, "/gate/status"),
         E("overlap", "transfer", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, "/overlap_with_v54_selection"),
         E("n_candidates", "transfer", "repo", f"{P}/201-neuron-level-causal-lane-v61-local-protocol.md", R, r"top-32 units by fit-split score \(v50\.select_per_layer, layers 10/18/26\) = (\d+) candidates"),
         E("rule_text", "rule", "repo", f"{P}/201-neuron-level-causal-lane-v61-local-protocol.md", R,
           r"Overlap with the V54 selection is reported descriptively only; there is no equality guard")],
     notes="32 of 96 candidates overlap; doc 201 declares the overlap descriptive, so no label threshold exists.")

item(id="astral-v60-sufficiency-v60local", cluster_id="astral-pythia-swap",
     claim={"text": "The last-position swap and donor-patch sufficiency effects (V60 L and P) hold when the same pipeline runs on another device and precision.",
            "family": "donor_patch_sufficiency"},
     base=V59_BASE, base_run="V60 cloud",
     transfer=dict(run_id="V60-local", **cond(PY28[0], PY28[1], PY28[2], "Apple MPS (torch 2.14.1)", "bfloat16", PY28[3], V54_CORPUS + " (rebuilt locally)", "not applicable")),
     axes=["device"], confounds=[], kind="closed", ceiling=LDU,
     ev=[E("base_flag", "base", "repo", f"{A}/v60-run/results/out_v60.json", J, "/headline_hardened"),
         E("closure", "transfer", "repo", f"{A}/v60-run/README.md", R, r"(## V60-local \(protocol doc 200, addendum 1\): CLOSED, selection mismatch, no result)")],
     notes="No swap or patch was measured on the local arm.")

item(id="astral-v59-reflexive-task", cluster_id="astral-pythia-swap",
     claim={"text": "The selected number units also carry reflexive-pronoun number agreement (ablation hurts himself/themselves agreement beyond matched-random sets).",
            "family": "task_transfer"},
     base=V59_BASE, base_run="V59 swap sufficiency",
     transfer=dict(run_id="V59 Task R", **cond(PY28[0], PY28[1], PY28[2], H100, BF16_DOC200, PY28[3], "reflexive items (himself/herself/itself/themselves) from all 35 V54 books, 286 items", "inherited")),
     axes=["task"], confounds=["items drawn from fit/tune/assess documents by amendment 2 made after seeing counts, before effects", "100 controls"],
     kind="reflexive", ceiling=LDU,
     ev=[E("base_flag", "base", "repo", f"{A}/v59-run/results/out_v59.json", J, "/swap/sufficiency_supported"),
         E("effect", "transfer", "repo", f"{A}/v59-run/results/out_v59.json", J, "/reflexive/effect"),
         E("ctl_max", "transfer", "repo", f"{A}/v59-run/results/out_v59.json", J, "/reflexive/ctl_max"),
         E("tgt_flag", "transfer", "repo", f"{A}/v59-run/results/out_v59.json", J, "/reflexive/transfer_supported"),
         E("rule_text", "rule", "repo", f"{P}/141-neuron-level-causal-lane-v59-protocol.md", R,
           r"\"Transfer supported\" iff the 8-unit drop exceeds all 100 controls and is >= 0\.1 logits")],
     notes="Task and base run share one execution; axis is a different task construction.")

for _id, _claim, _fam, _base_ev, _anchor, _run, _txt in [
    ("astral-v57-mean-ablation", "Mean-ablating the selected 128 units drops the agreement margin beyond matched-random sets (mean-ablation supported).", "neuron_ablation_effect",
     E("base_flag", "base", "repo", f"{A}/v57-run/results/out_v57.json", J, "/mean_ablation/supported"), r"\| V57 \| 2\.8B \| dose-response not monotone; mean-ablation supported \|", "V57",
     "Single model, single corpus; no later run repeats mean-ablation."),
    ("astral-v58-concentration", "A few units carry most of the joint effect (top-4 units carry about 75% of the k=128 joint drop).", "neuron_ablation_effect",
     E("base_flag", "base", "repo", f"{A}/v58-run/results/out_v58.json", J, "/h_concentration"), r"\| V58 \| 2\.8B \| concentration true; agreement-selective false; direct-path false \|", "V58",
     "Single model; the same run found agreement-selectivity and direct-path false."),
    ("astral-v61-minimal-pair", "Patching the 7 swap units' last-position activations between minimal pairs (subject number differs only) lowers the correct-verb margin beyond matched-random sets.", "donor_patch_sufficiency",
     E("base_flag", "base", "repo", f"{A}/v61-run/results/out_v61.json", J, "/minimal_pair_confirmed"), r"\| V61 \(cloud\) \| 2\.8B \| minimal-pair donor patch supported \|", "V61 (cloud)",
     "Synthetic templates; drop 0.482 logits, about 10% of baseline margin."),
]:
    item(id=_id, cluster_id="astral-pythia-swap" if "v57" not in _id and "v58" not in _id else "astral-pythia-ablation",
         claim={"text": _claim, "family": _fam},
         base=cond(PY28[0], PY28[1], PY28[2], H100, NOC if "v61" not in _id else BF16_DOC200, PY28[3],
                   V54_CORPUS if "v61" not in _id else "synthetic minimal-pair templates (seed 6161, 800 items)", "inherited from locked V54 selection"),
         base_run=_run,
         transfer=dict(run_id=None, **cond("none", "none", None, "none", "none", "none", "none", "none")),
         axes=[], confounds=[], kind="no_arm", ceiling=LDU,
         ev=[_base_ev,
             E("absence", "absence", "repo", "imports/astral-neuron/runs", "absence_assertion",
               "imports/astral-neuron/runs;imports/astral-neuron/protocols;docs/protocols"),
             E("anchor", "rule", "repo", "docs/INDEX.md", R, _anchor)],
         notes=_txt + " Absence assertion: no run of the same frozen procedure on another model, device or corpus appears in the listed directories (author search, not machine-checkable).")

item(id="astral-v62-device-dtype", cluster_id="astral-pythia-swap",
     claim={"text": "The V61-local result (T0, L, P supported; swap set and candidate list) survives changing device and numeric precision (MPS f32, CPU f32, MPS bf16, MPS fp16).",
            "family": "class_swap_sufficiency"},
     base=V61L, base_run="V61-local",
     transfer=dict(run_id="V62 (prepared)", **cond(PY28[0], PY28[1], PY28[2], "Apple MPS / CPU (arms A-D)", "float32, bfloat16, fp16", PY28[3], V54_CORPUS, "n/a")),
     axes=["device", "dtype"], confounds=[], kind="no_arm", ceiling=LDU,
     ev=[E("base_flag", "base", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, "/headline_reproduced_and_hardened"),
         E("absence", "absence", "repo", f"{A}/v61-local-run", "absence_assertion", "imports/astral-neuron/runs/v61-local-run;imports/astral-neuron/protocols/202-neuron-level-causal-lane-v62-robustness-protocol.md"),
         E("anchor", "rule", "repo", "docs/INDEX.md", R, r"\| V62 \| 2\.8B \| \*\*prepared, not run\*\*")],
     notes="docs/INDEX.md: V62 prepared, not run; no v62-run directory exists.")

# ---------------------------------------------------------------- V49 (Qwen / SmolLM)
V49_BASE = cond("Qwen3-1.7B", "1.7B", 1.7, "H100 (gman node astral-v49-neuron-r1)", NOC,
                "Qwen/Qwen3-1.7B-MLX-bf16 rev 720c04346ea2b095c801ebbd545c109230964cd4",
                "templated subject-verb agreement, templates A (12 templates, template-disjoint splits)",
                "InstrumentValid (lens criterion changed post hoc after r3)")
V49_PATH = f"{A}/v49-run"


def v49(id_, arm_file, tgt, axes, confounds, note):
    item(id=id_, cluster_id="astral-v49", claim={
        "text": "Ablating the 32 fit-selected MLP neurons lowers the subject-verb agreement margin far beyond matched-random neuron sets (Holm-rejected at the 1000-draw floor), with the Lane B instrument gate valid.",
        "family": "neuron_ablation_effect"},
        base=V49_BASE, base_run="V49 r4", transfer=dict(run_id=arm_file[:-5], **tgt), axes=axes, confounds=confounds,
        kind="v49_arm", ceiling=SRC,
        ev=[E("base_status", "base", "repo", f"{V49_PATH}/r4.json", J, "/v49_run/status"),
            E("base_gate", "base", "repo", f"{V49_PATH}/r4.json", J, "/gate/status"),
            E("tgt_status", "transfer", "repo", f"{V49_PATH}/{arm_file}", J, "/v49_run/status"),
            E("tgt_gate", "transfer", "repo", f"{V49_PATH}/{arm_file}", J, "/gate/status"),
            E("tgt_fails", "transfer", "repo", f"{V49_PATH}/{arm_file}", J, "/gate/fails")],
        notes=note + " The base run's gate passed only after its criterion was rewritten post hoc (execution-record-r4); r4 reproduces r3 numerically, not scientifically. Gate criteria differ across arms (z-based in r4/g1/g2, shuffled-readout <10% floor in c1/b1/b2); each run's own recorded gate status is used.")


v49("v49-r4-g1-templates", "g1.json",
    cond("Qwen3-1.7B", "1.7B", 1.7, "H100 (same node)", NOC, "same revision", "templates B, fresh documents", "InstrumentNotValid (known_bad_instrument)"),
    ["corpus"], ["selection refit"], "g1: Candidate verdict, gate failed known-bad (z 3.79, effect 2.6% of selected).")
v49("v49-r4-g2-qwen4b", "g2.json",
    cond("Qwen3-4B-Base", "4B", 4.0, "H100 (same node)", NOC, "Qwen3-4B-Base (layers 13/21/29)", "templates B", "InstrumentValid"),
    ["scale", "corpus"], ["layers 13/21/29 vs 10/16/22"], "g2: gate all checks passed.")
v49("v49-r4-c1-smollm2", "c1.json",
    cond("SmolLM2-1.7B (Llama-style)", "1.7B", 1.7, "H100 (same node)", NOC, "HuggingFaceTB/SmolLM2-1.7B (layers 8/12/18)", "1200 natural Wikipedia sentences (wikitext-103), document-disjoint", "InstrumentNotValid (known_bad_instrument)"),
    ["model_family", "corpus"], ["class-imbalanced corpus", "known-bad rule replaced by 10%-of-selected floor"], "c1: shuffled-label readout effect 11.2% of selected vs 10% floor.")
v49("v49-r4-b1-smollm2-balanced", "b1.json",
    cond("SmolLM2-1.7B (Llama-style)", "1.7B", 1.7, "H100 (same node)", NOC, "HuggingFaceTB/SmolLM2-1.7B (layers 8/12/18)", "1500 class-balanced natural wikitext sentences (250 is / 250 are per group)", "InstrumentValid"),
    ["model_family", "corpus"], ["known-bad rule: shuffled readout < 10% of selected (8.7%)"], "b1: passed the preregistered floor narrowly.")
v49("v49-r4-b2-qwen4b-balanced", "b2.json",
    cond("Qwen3-4B-Base", "4B", 4.0, "H100 (same node)", NOC, "Qwen3-4B-Base (layers 13/21/29)", "1500 class-balanced natural wikitext sentences", "InstrumentValid"),
    ["scale", "corpus"], ["known-bad rule: shuffled readout < 10% of selected (9.1%)"], "b2: passed the preregistered floor narrowly.")

# ---------------------------------------------------------------- R6 (second family)
R6_ARM = cond("Qwen3-1.7B-Base", "1.7B", 1.7, "CUDA: gate on A10G, later stages H100 (TF32 off)", "float32 (bf16 weights cast)",
              "Qwen3-1.7B-Base, layers 8/15/22", "V54 corpus per R6 protocol section 2 (unchanged)", "InstrumentValid (gate v3)")
for tid, tname, bptr, fam in [("t0", "T0 all-position swap", "t0_all_position_swap", "class_swap_sufficiency"),
                              ("l", "L last-position swap", "l_last_position_swap", "class_swap_sufficiency"),
                              ("p", "P donor patch", "p_donor_patch", "donor_patch_sufficiency")]:
    item(id=f"r6-{tid}-qwen3", cluster_id="r6-second-family",
         claim={"text": f"Task {tname}: swapping/patching the selected units drops the verb margin far beyond 1000 matched-random sets on a model of a different family.", "family": fam},
         base=V61L, base_run=f"V61-local {tid.upper()}",
         transfer=dict(run_id="R6", **R6_ARM), axes=["model_family", "scale", "device"],
         confounds=["1000 controls vs 100 in V61-local", "gate computed on A10G, later stages on H100 (no cross-device stability claim)", "single run resumed twice"],
         kind="d201_task", ceiling=LDU,
         ev=[E("base_flag", "base", "repo", f"{A}/v61-local-run/results/out_v61_local.json", J, f"/{bptr}/supported"),
             E("tgt_sing", "transfer", "repo", "docs/protocols/R6-result.json", J, f"/{bptr}/selected/sing"),
             E("tgt_plur", "transfer", "repo", "docs/protocols/R6-result.json", J, f"/{bptr}/selected/plur"),
             E("tgt_all", "transfer", "repo", "docs/protocols/R6-result.json", J, f"/{bptr}/selected/all"),
             E("tgt_ctl_sing", "transfer", "repo", "docs/protocols/R6-result.json", J, f"/{bptr}/ctl_max/sing"),
             E("tgt_ctl_plur", "transfer", "repo", "docs/protocols/R6-result.json", J, f"/{bptr}/ctl_max/plur"),
             E("tgt_flag", "transfer", "repo", "docs/protocols/R6-result.json", J, f"/{bptr}/supported"),
             E("tgt_gate", "transfer", "repo", "docs/protocols/R6-result.json", J, "/gate/status"),
             E("rule_text", "rule", "repo", "docs/protocols/R6-second-model-agreement-gate.md", R,
               r"\"Replicates on second family\" iff the gate is valid and T0, L, P are all supported")],
         notes="R6 result: single run, owner-unreviewed; devices differ between gate and later stages.")

# ---------------------------------------------------------------- R4 offshoot (in-repo experiments)
R4D = cond("Pythia-160m", "160M", 0.16, "local (not recorded in result JSON)", NOC, "13 Pythia checkpoints step1000-step143000", "wikitext-2-raw-v1 train", "n/a (statistic with 1000-permutation null)")
for iid, arm, tgt, axes, conf, note in [
    ("r4d-r4e-fresh-text", "r4e_armA_pythia160m_fresh_text.json",
     cond("Pythia-160m", "160M", 0.16, "local (not recorded)", NOC, "same checkpoint grid", "fresh WikiText-2 slice", "n/a"), ["corpus"], ["R4d was designed after R4c near-miss (post-hoc power increase)"],
     "Arm A: same model, fresh text slice."),
    ("r4d-r4e-pythia70m", "r4e_armB_pythia70m_fresh_text.json",
     cond("Pythia-70m", "70M", 0.07, "local (not recorded)", NOC, "12-checkpoint grid (step1000 omitted), fp16-sourced weights", "fresh WikiText-2 slice", "n/a"), ["scale", "checkpoint", "corpus"],
     ["different checkpoint grid", "fresh text slice"], "Arm B: smaller model, fresh text slice, different checkpoint grid."),
]:
    item(id=iid, cluster_id="r4-offshoot",
         claim={"text": "Among a neuron's persistent top tokens, tokens that joined its top set closer in time are more embedding-similar than a frequency-stratified permutation null predicts (token-level directed growth).",
                "family": "offshoot_growth"},
         base=R4D, base_run="R4d", transfer=dict(run_id="R4e " + arm.split("_")[1], **tgt), axes=axes, confounds=conf,
         kind="r4e_arm", ceiling=LDU,
         ev=[E("base_layers", "base", "repo", "results/r4d_powered_birth_order.json", J, "/layers"),
             E("tgt_layers", "transfer", "repo", f"results/{arm}", J, "/layers"),
             E("rule_text", "rule", "repo", "docs/protocols/R4e-replication.md", R,
               r"Layer growth-pass: mean rho <= -0\.05, p <= 0\.01 \(1000 frequency-stratified permutations\), >= 30 qualifying neurons\. Arm passes iff growth-pass in >= 2 of 3 layers")],
         notes=note + " Effect sizes are small (rho -0.05 to -0.11); several layers clear the -0.05 bar by 0.003-0.008.")

# ---------------------------------------------------------------- theseus-lab (affine replacement)
V8_BASE = cond("OPT/ReLU, GPT-2/GELU, Llama/SiLU-gated checkpoints (one each)", "<=500M cap (OPT 98.2M, AMD-Llama 135M; GPT-2-family size not stated)", None,
               "CPU (macOS arm64)", NOC, "kanishka/opt-babylm2-... rev baa6df5e; ai-forever/rugpt3small rev a9307e69; amd/AMD-Llama-135m rev 9a224a07",
               "synthetic template prompts (adaptation 8,020-9,191 token rows per case)", "all frozen treatment and control gates passed (packet PASS)")
V8_EV = E("base_ok", "base", "repo", f"{T}/replacement-loss-v8-results.md", R, r"The frozen v8 primary run passed all nine checkpoint-by-seed cases")


def theseus(id_, tgt, axes, conf, ev_extra, note, tgt_run):
    item(id=id_, cluster_id="theseus-affine-replacement",
         claim={"text": "A newly fitted affine MLP output projection, estimated from disjoint adaptation activations without copying target weights, predicts held-out module and logit intervention effects, rescues activations and preserves next-token behaviour (frozen v8 gates).",
                "family": "affine_replacement_prediction"},
         base=V8_BASE, base_run="V8", transfer=dict(run_id=tgt_run, **tgt), axes=axes, confounds=conf,
         kind="theseus_arm", ceiling=LDU, ev=[V8_EV] + ev_extra, notes=note)


BAND_A = cond("facebook/opt-2.7b; openai-community/gpt2-xl; HuggingFaceTB/SmolLM2-1.7B", "1.7B-2.7B", None, "CPU (cloud VM, no paid GPU)", NOC,
              "three checkpoints selected after freeze, three seeds each", "synthetic template prompts, 128 adaptation prompts", "frozen gates unchanged from v3/v8")
theseus("theseus-v8-bandA", BAND_A, ["scale", "checkpoint"], ["first execution stopped at report step and was re-run over sealed predictions (tooling failure, disclosed)"],
        [E("tgt_status", "transfer", "repo", f"{T}/replacement-loss-v8-scale-bandA-report.json", J, "/outcome_status"),
         E("all_cases_pass", "transfer", "repo", f"{T}/replacement-loss-v8-scale-bandA-report.json", J, "/all_treatment_case_gates_pass")],
        "Band A: 0/9 cases pass the primary gates; behaviour preserved and effect direction predicted, pooled median cosine under the 0.90 gate.", "v8-scale Band A")
theseus("theseus-v8-v28-fresh", cond("fresh OPT/ReLU, GPT-2/GELU, Llama/SiLU-gated checkpoints", "unrecorded in V28 results", None, "local (no paid compute or remote worker)", NOC,
                                      "three fresh checkpoints, three seeds each", "synthetic template prompts", "case-level gates passed 9/9; global control gate failed"),
        ["checkpoint"], ["frozen global mapping-aligned permutation gate (<=1e-4) missed: module-effect max diff 4.12e-4, logit-effect 2.21e-4"],
        [E("tgt_status", "transfer", "repo", f"{T}/replacement-loss-v28-results.md", R, r"\*\*Outcome: `([A-Z_]+)`\. Assurance"),
         E("case_level_all_pass", "transfer", "repo", f"{T}/replacement-loss-v28-results.md", R,
           r"(All nine case-level treatment, held-out behavior, causal-effect, and activation-rescue checks passed)")],
        "V28: packet FAIL on a frozen float32 accumulation-order control gate; the discrepancy is consistent with numeric sensitivity but V28 did not adjudicate that. Note (F1 spot-check): the source reports all nine case-level treatment, held-out behavior, causal-effect and activation-rescue checks as passed; `failed` here follows the packet outcome rule, not a failure of the intervention predictions. See LABEL_RULES section 5.", "V28")
theseus("theseus-v8-cloudrun", cond("same frozen v8 packet", "same", None, "Google Cloud Run CPU (Linux/amd64, 4 vCPU, 16 GiB)", NOC, "same", "same", "n/a"),
        ["device"], ["replay container, different OS and CPU"],
        [E("tgt_status", "transfer", "repo", f"{T}/replacement-loss-v8-cloud-run-replay-v1.md", R, r"The scientific result is `([A-Z_]+)`, not `INTERNAL_PASS`")],
        "V3 replay: INCONCLUSIVE; the evaluator did not produce case outcomes within the frozen resource packet.", "v8 Cloud Run replay V3")
theseus("theseus-v8-r2-c4", BAND_A, ["scale", "checkpoint"], ["adaptation prompts 4x (512) vs 128", "resource-limit stop at 25.3 GiB peak RSS vs 24 GiB frozen limit"],
        [E("tgt_status", "transfer", "repo", f"{T}/replacement-loss-v8-scale-r2-C4-report.json", J, "/outcome_status"),
         E("all_cases_pass", "transfer", "repo", f"{T}/replacement-loss-v8-scale-r2-C4-report.json", J, "/all_treatment_case_gates_pass"),
         E("resource_blockers", "transfer", "repo", f"{T}/replacement-loss-v8-scale-r2-C4-report.json", J, "/resource_blockers")],
        "r2 C4: all nine case gates pass but the arm is INCOMPLETE by the frozen resource rule; recorded as inconclusive, not PASS.", "v8-scale r2 C4")
for arm, n, note in [("C2", 256, "r3 confirmatory coverage packet, 256 adaptation prompts."), ("C4", 512, "r3 confirmatory coverage packet, 512 adaptation prompts; memory limit 30 GiB preregistered in that packet.")]:
    theseus(f"theseus-v8-r3-{arm.lower()}", BAND_A, ["scale", "checkpoint"], [f"adaptation prompts {n} vs 128 in Band A", "no concurrent 128-prompt control in this packet"],
            [E("tgt_status", "transfer", "repo", f"{T}/replacement-loss-v8-scale-r3-{arm}-report.json", J, "/outcome_status"),
             E("all_cases_pass", "transfer", "repo", f"{T}/replacement-loss-v8-scale-r3-{arm}-report.json", J, "/all_treatment_case_gates_pass")],
            note, f"v8-scale r3 {arm}")
for arm, n, note in [("N128", 128, "r5 natural-text adaptation, 128 paragraphs; 3/9 cases pass."), ("N256", 256, "r5 natural-text adaptation, 256 paragraphs (concurrent template control T256 also passed)."),
                     ("N512", 512, "r5 natural-text adaptation, 512 paragraphs.")]:
    theseus(f"theseus-v8-r5-{arm.lower()}",
            cond("facebook/opt-2.7b; openai-community/gpt2-xl; HuggingFaceTB/SmolLM2-1.7B", "1.7B-2.7B", None, "CPU (cloud VM)", NOC, "same three checkpoints, three fresh seeds", f"natural-text adaptation, {n} paragraphs", "frozen gates unchanged"),
            ["scale", "checkpoint", "corpus"], [f"adaptation size {n} paragraphs"],
            [E("tgt_status", "transfer", "repo", f"{T}/replacement-loss-v8-scale-r5-{arm}-report.json", J, "/outcome_status"),
             E("all_cases_pass", "transfer", "repo", f"{T}/replacement-loss-v8-scale-r5-{arm}-report.json", J, "/all_treatment_case_gates_pass")],
            note, f"v8-scale r5 {arm}")

# ---------------------------------------------------------------- jspace-pilot (OLMo-2-7B checkpoints)
OLMO = "OLMo-2-1124-7B Base / SFT / DPO / Instruct"
for iid, a, b, axes, conf, tgt, note in [
    ("jspace-pilot-main", "gateA", "gateM", ["scale", "device"], ["pilot 20 templates x 50 prompts vs main 40 templates x 100 prompts", "main run on Modal, device of pilot not recorded"],
     cond(OLMO, "7B", 7.0, "Modal (gateM)", NOC, "four OLMo-2 checkpoints", "40 valence templates x 100 tasks", "additivity gate failed"),
     "Both runs read inconclusive / H_null under the pre-specified rule."),
    ("jspace-main-locked", "gateM", "gateN", ["device"], ["locked config differs (clamp L13-18, k=25, closure 2) vs config A (closure 0)", "tuning sweep on held-out frames between runs"],
     cond(OLMO, "7B", 7.0, "gman H100", NOC, "four OLMo-2 checkpoints", "40 valence templates x 100 tasks, locked config", "additivity gate failed for all four"),
     "Both runs read inconclusive / H_null; post-hoc damage-free bound and graded clamp are not part of the pre-specified rule."),
]:
    item(id=iid, cluster_id="jspace-olmo-route",
         claim={"text": "Preference tuning (DPO minus SFT) raises the workspace-route effect T_J by more than the pre-specified 0.5 logit with the 2.5% bound above 0.",
                "family": "route_bypass"},
         base=cond(OLMO, "7B", 7.0, "unrecorded" if a == "gateA" else "Modal (gateM)", NOC, "four OLMo-2 checkpoints", "20 templates x 50 prompts (pilot)" if a == "gateA" else "40 valence templates x 100 tasks", "additivity gate failed"),
         base_run=a, transfer=dict(run_id=b, **tgt), axes=axes, confounds=conf, kind="decision_read", ceiling=LDU,
         ev=[E("base_read", "base", "jspace", f"results/{a}_decision.json", J, "/own dose*/read"),
             E("base_additivity_ok", "base", "jspace", f"results/{a}_decision.json", J, "/own dose*/additivity_ok"),
             E("tgt_read", "transfer", "jspace", f"results/{b}_decision.json", J, "/own dose*/read"),
             E("tgt_additivity_ok", "transfer", "jspace", f"results/{b}_decision.json", J, "/own dose*/additivity_ok")],
         notes=note + " dT_J DPO-SFT about +0.35 logit with CI excluding 0 but below the 0.5 threshold in all three runs.")
