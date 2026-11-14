# F1 label rules (preregistration, committed before any label exists)

Ceiling: `LocalDevelopmentUnreviewed`. Governs `bench/transfer/items.jsonl`.
This file is committed before `items.jsonl`, the derivation code, the split or
any baseline exists in the repository. Changing a rule or threshold after labels
exist requires a new file (`LABEL_RULES_v2.md`) and a new bench version; labels
produced under this file are never relabelled in place.

## 0. Disclosure about what the author had already read

Before writing this file the author (an AI agent, same-owner, not independent)
surveyed the source repositories to see what exists: run READMEs, `out_*.json`
result files and result write-ups for the Astral lane (V49-V61-local), the
theseus-lab reports (V7-V28, Band A, r2-r5), `R4d/R4e/R6` results, and the
`jspace-pilot` RESULTS/decision files. The author therefore **knows most
outcomes**. This preregistration is consequently not blind. What is fixed
instead is the mapping from evidence to label:

1. every numeric threshold below is copied from a protocol that the source
   authors committed before their own run (location given in section 3), none
   is chosen here;
2. which rule applies to an arm is decided by which source protocol governed
   that arm, listed in section 4 before any label is computed;
3. the two judgement boundaries that remain are resolved here, in advance, by
   following the source's own status word (section 5).

A reader should treat the labels as "what the sources' own frozen rules say",
re-derived mechanically, not as the author's opinion of what replicated.

## 1. Definitions

* **Claim C**: a causal or predictive statement with a frozen rule in its source
  protocol. **Base run B**: the run where C was first evaluated.
  **Transfer arm T**: a run that applied the same frozen procedure with at least
  one axis changed (`scale`, `device`, `dtype`, `checkpoint`, `model_family`,
  `corpus`, `task`; see `SCHEMA.md`).
* **Base verdict class**: `supported` iff B's own source status is one of
  `Candidate` (with instrument gate `InstrumentValid` where a gate exists),
  `PASS`, `supported: true`, `headline_reproduced_and_hardened: true`,
  `growth_pass` true, `Replicates`, or the source's text states the claim held.
  `inconclusive` iff the source status is inconclusive/`InstrumentNotValid`/
  `H_null`. `not_supported` otherwise.
* **Arm status**: `completed`, `completed_instrument_invalid` (the source records
  `InstrumentNotValid`), `closed_before_measurement` (guard abort, setup stop,
  budget stop before the claimed quantity was measured), `inconclusive` (source
  status `INCONCLUSIVE`/`INCOMPLETE`, including resource-limit stops), `not_run`.

## 2. Rules, applied in this order; the first that fires labels the item

| id | condition | label | reason |
|---|---|---|---|
| L5 | base verdict class is not `supported` | `not_tested` | `base_not_established` |
| L0 | no transfer arm with results exists on disk (including arms prepared but not run) | `not_tested` | `no_arm_run` |
| L1 | arm status is `closed_before_measurement` or `inconclusive` for the quantity the claim is about | `not_tested` | `arm_closed_before_measurement` or `arm_inconclusive_by_source_rule` |
| L2:`<S-id>` | the arm was governed by a source replication rule listed in section 3 | `replicated` iff the source rule's positive condition holds on the extracted numbers, else `failed` | `rule_met` / `rule_not_met` |
| L4 | arm status is `completed_instrument_invalid` and no L2 rule governs the arm | `not_tested` | `instrument_invalid_no_rule` |
| L3 | the arm's frozen protocol returns a primary verdict and no L2 rule governs it: `PASS`, or `Candidate` with gate `InstrumentValid` -> `replicated`; `FAIL` or `NoCandidate` -> `failed`. If the arm produced only descriptive numbers with no frozen verdict for the quantity | `replicated`/`failed` as stated, or `not_tested` with reason `no_preregistered_rule` | `rule_met`/`rule_not_met`/`no_preregistered_rule` |

L4 sits before L3 and after L2 on purpose: an effect read through an instrument
that its own gate rejected is not counted for or against the claim (AGENTS rule
6), unless the source protocol itself declared that case a non-replication
(L2:S-V54R does, for V55).

Labels never use a threshold other than those in section 3. A numeric input that
has no section-3 threshold gets `no_preregistered_rule`, not a threshold chosen
now.

## 3. Source rules (L2) and every threshold, with provenance

All paths are repository-relative; the quoted condition is paraphrased only to
make it mechanical, the verbatim text is pinned by `regex` evidence in each item.

| S-id | Applies to | Positive condition (everything else = not met) | Source |
|---|---|---|---|
| S-V54R | Astral arms V54 (vs V53) and V55 (vs V54) | `status == "Candidate"` and gate `== "InstrumentValid"` and matched-random `p <= 0.01` on tune and on assess and specificity `pass == true`. Doc 137 states anything else is "a non-replication on this model". Frozen sub-thresholds inside the gate and specificity (gate v3 `Z_NULL 4.0`, `Z_SIGNAL 10.0`, min effect `0.5` logits; constant control `< 0.5`, gap `>= 0.5`, `z >= 4.0`) are applied by the source runner and read from its output flags, not recomputed. | `imports/astral-neuron/protocols/136-neuron-level-causal-lane-v54-protocol.md` (Interpretation rule fixed now), `.../137-neuron-level-causal-lane-v55-protocol.md` (Replication rule) |
| S-D201 | Astral V61-local arms T0, L, P | task "supported" iff singular-drop `>` singular control max, plural-drop `>` plural control max, and combined drop `>= 0.1` logits (100 matched sets, p floor 0.0099). A task replicates iff supported. | `.../201-neuron-level-causal-lane-v61-local-protocol.md` step 4 and Decision rule |
| S-R6 | R6 arms T0, L, P on Qwen3-1.7B-Base | S-D201 on the task **and** instrument gate `InstrumentValid` (R6 decision rule: "Replicates ... iff the gate is valid and T0, L, P are all supported"; applied per task so the three arms get three items). Control draws 1000. | `docs/protocols/R6-second-model-agreement-gate.md` section 3 |
| S-D141R | V59 reflexive-agreement transfer (`task`) | transfer supported iff the 8-unit ablation drop exceeds every one of the 100 matched controls and is `>= 0.1` logits | `.../141-neuron-level-causal-lane-v59-protocol.md` Task R |
| S-D200EQ | V60-local selection reproduction (`device`, `dtype`) | replicated iff the fresh k=32 reselection equals the locked V54 selection; "A selection mismatch ... closes V60 without retuning" | `.../200-neuron-level-causal-lane-v60-protocol.md` (Units and items; Budget and rules) |
| S-D202J | swap-set stability V59 -> V61-local | stable (replicated) iff swap-set Jaccard `>= 0.5`; the protocol's "selection precision-sensitive" flag is Jaccard `< 0.5`. Jaccard = overlap / (n_base + n_target - overlap). Extending the 0.5 line from V62's arms to this pair is disclosed, because it is the same quantity. | `.../202-neuron-level-causal-lane-v62-robustness-protocol.md` Decision rule |
| S-R4E | R4e arms A (fresh text) and B (Pythia-70m) vs R4d | layer growth-pass iff mean rho `<= -0.05`, p `<= 0.01` (1000 permutations) and `>= 30` qualifying neurons; arm passes iff growth-pass in `>= 2` of 3 layers; item replicates iff its arm passes | `docs/protocols/R4e-replication.md` Decision rules |

Arm-verdict sources for L3: theseus-lab arm reports expose `outcome_status`
(`PASS`, `FAIL`, `INCOMPLETE`) whose conditions are the frozen gates of V8
(`imports/theseus-lab/docs/replacement-loss-v8-results.md`: causal direction
accuracy `>= 0.80`, median cosine `>= 0.90`, normalized error `<= 0.20`, rescue
cosine `>= 0.90`, paired logit error `<= 0.05`, continuation CE increase
`<= 0.05` nats/token; gates unchanged in v8-scale r2/r3/r5). Astral V49 arms
expose `v49_run.status` and `gate.status`. The packet-level status word is used
as recorded; per-case sub-results do not override it.

No threshold in this file was derived from the labels.

## 4. Planned items and the rule that will label each

Declared before labelling. Count: 36 planned. Unless stated, the arm is the
literal run named; the base is the run named first.

| id | base -> arm | axes changed | rule |
|---|---|---|---|
| astral-v53-v54-corpus | V53 -> V54 (Pythia-2.8b, fresh books) | corpus | L2:S-V54R |
| astral-v54-v55-scale | V54 -> V55 (Pythia-1.4b) | scale | L2:S-V54R |
| astral-v50-scale-1.4b-2.8b | V50 1.4b -> V50 2.8b (gate v1) | scale | L5 (base `InstrumentNotValid`) |
| astral-v59-t0-v61local | V59 all-position swap -> V61-local T0 | device, dtype | L2:S-D201 |
| astral-v60-l-v61local | V60 Task L -> V61-local L | device, dtype | L2:S-D201 |
| astral-v60-p-v61local | V60 Task P -> V61-local P | device, dtype | L2:S-D201 |
| astral-v59-swapset-v61local | V59 7-unit swap set -> V61-local 8-unit swap set | device, dtype | L2:S-D202J |
| astral-v54-selection-v60local | V54 locked selection -> V60-local reselection (MPS bf16) | device | L2:S-D200EQ |
| astral-v54-candidates-v61local | V54 selection -> V61-local candidate list | device, dtype | L3 (no frozen threshold for candidate-list overlap) |
| astral-v60-sufficiency-v60local | V60 L/P effect -> V60-local | device | L1 (closed before measurement) |
| astral-v59-reflexive-task | V59 swap-sufficiency units -> V59 Task R | task | L2:S-D141R |
| astral-v57-mean-ablation | V57 mean-ablation (2.8b) -> none | none | L0 |
| astral-v58-concentration | V58 top-4 concentration (2.8b) -> none | none | L0 |
| astral-v61-minimal-pair | V61 minimal-pair patch (2.8b) -> none | none | L0 |
| astral-v62-device-dtype | V61-local -> V62 arms A-D (prepared, not run) | device, dtype | L0 |
| v49-r4-g1-templates | V49 r4 -> g1 (Qwen3-1.7B, templates B) | corpus | L4 or L3 by recorded gate |
| v49-r4-g2-qwen4b | V49 r4 -> g2 (Qwen3-4B-Base) | scale, corpus | L4 or L3 by recorded gate |
| v49-r4-c1-smollm2 | V49 r4 -> c1 (SmolLM2-1.7B, natural text) | model_family, corpus | L4 or L3 by recorded gate |
| v49-r4-b1-smollm2-balanced | V49 r4 -> b1 | model_family, corpus | L4 or L3 by recorded gate |
| v49-r4-b2-qwen4b-balanced | V49 r4 -> b2 | scale, corpus | L4 or L3 by recorded gate |
| r6-t0-qwen3 | V61-local T0 -> R6 T0 | model_family, scale, device | L2:S-R6 |
| r6-l-qwen3 | V61-local L -> R6 L | model_family, scale, device | L2:S-R6 |
| r6-p-qwen3 | V61-local P -> R6 P | model_family, scale, device | L2:S-R6 |
| r4d-r4e-fresh-text | R4d -> R4e arm A (Pythia-160m fresh text) | corpus | L2:S-R4E |
| r4d-r4e-pythia70m | R4d -> R4e arm B (Pythia-70m) | scale | L2:S-R4E |
| theseus-v8-bandA | V8 (<=500M) -> Band A (opt-2.7b, gpt2-xl, SmolLM2-1.7B) | scale, checkpoint | L3 |
| theseus-v8-v28-fresh | V8 -> V28 fresh checkpoints | checkpoint | L3 (packet status word) |
| theseus-v8-cloudrun | V8 -> Cloud Run replay V3 | device | L1 (replay `INCONCLUSIVE`) |
| theseus-v8-r3-c2 | V8 -> r3 arm C2 (256 adaptation prompts) | scale, checkpoint | L3 |
| theseus-v8-r3-c4 | V8 -> r3 arm C4 (512) | scale, checkpoint | L3 |
| theseus-v8-r2-c4 | V8 -> r2 arm C4 | scale, checkpoint | L1 (`INCOMPLETE`, RSS limit) |
| theseus-v8-r5-n128 | V8 -> r5 N128 natural text | scale, checkpoint, corpus | L3 |
| theseus-v8-r5-n256 | V8 -> r5 N256 | scale, checkpoint, corpus | L3 |
| theseus-v8-r5-n512 | V8 -> r5 N512 | scale, checkpoint, corpus | L3 |
| jspace-pilot-main | jspace gateA -> gateM (DPO-SFT route effect) | scale (design size), device | L5 (base `H_null`) |
| jspace-main-locked | jspace gateM -> gateN | device | L5 (base `H_null`) |

Excluded sources and why are in `EXCLUDED.md` (written with the items, from the
same survey). `composed-zk-benchmark-os` contributes no planned item: its
interpretability results are single-setting gate verdicts or non-causal
qualifications, and its one rerun thread (V41R30) changes no primary axis.

## 5. Judgement boundaries fixed in advance

1. **Theseus V28.** The packet status is `FAIL` because a frozen global control
   gate missed; all nine case-level gates passed. Rule: L3 uses the packet status
   word, so the item is `failed`. The case-level result is recorded in
   `label_inputs`. This differs from V55-style `InstrumentNotValid` (L4,
   `not_tested`) only because the source used the word `FAIL`.
2. **Candidate-list overlap V54 -> V61-local** has numbers but the source
   declares them descriptive, so `no_preregistered_rule` -> `not_tested`.

## 6. Re-derivation

`scripts/verify.py` recomputes every `label_inputs` value from the cited
`evidence`, re-applies this table, and fails on any difference with
`items.jsonl`. It also recomputes the split. Evidence of kind
`absence_assertion` is the one thing it cannot re-derive.

## 7. Abort / reporting rules

* A planned item whose evidence cannot be extracted is dropped and reported as a
  gap, never replaced with a different arm.
* Negative and `not_tested` items are reported with the same prominence as
  `replicated` (AGENTS rule 7). The count of each label is stated wherever the
  item count is stated.
* No run, download, GPU, Modal or gman use is part of this bench phase.
