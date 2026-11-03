# R6 protocol: does the Pythia-2.8B agreement-unit result replicate on a second model family?

Status: **FROZEN 2026-11-03 (protocol only), nothing run.** Slice `mech-interp-unified-r6`. Ceiling `LocalDevelopmentUnreviewed`. The runner is not written yet; it must be committed, and its tests pass, before any model is loaded. Freeze decisions (owner chat, 2026-11-03): subject Qwen3-1.7B-Base; 1000 control draws; R6-B, R6-C and R7 deferred (no downloads). Layers fixed at **9, 16, 23** (0-indexed positions of V61's 10/18/26 of 32 scaled to Qwen3-1.7B's 28 layers, `num_hidden_layers` = 28 from the cached config, rounded to nearest). No other layer search. Written 2026-11-03 after triaging public X claims about "neuron-level" findings in Qwen models (abstraction neurons as embeddings, a single "Einstein neuron", Neuron Tuning). Those claims report no matched-random or known-bad control. This repo has the instrument that supplies both.

Freezing requires the owner decisions in section 8. Until frozen, no model is loaded and no data is read.

## 1. Question
The V53-V61 result (a small set of fit-selected MLP units is necessary and partly sufficient for subject-verb number agreement, beyond matched-random units) is shown only on Pythia-2.8B. `docs/INDEX.md` lists "not shown on second model" and V55 showed the effect does not transfer to Pythia-1.4B under the frozen gate. Does it hold on a different family, Qwen3-1.7B-Base, using the V61-local pipeline unchanged?

## 2. Design (to be frozen)
The V61-local pipeline (`imports/astral-neuron/protocols/201-neuron-level-causal-lane-v61-local-protocol.md`) is applied to a new subject model with no change to logic, templates, corpus, split rule or decision rule. Only model-dependent parameters change, and each is fixed in the freeze step before any tune/assess measurement:
- Subject: **Qwen3-1.7B-Base**, bf16 weights cast to float32, Apple MPS. Already in the HF cache (no download, the cache copy is 6.4 GB; digest recorded before use).
- Layers: V61 used layers 10/18/26 of a 32-layer model. Qwen3-1.7B has 28 layers; use the layers at the same relative depth (rounded), stated in the freeze record. No other layer search.
- Candidates: per-layer top-32 units by fit-split score, as V61 (locked list + digest before tune/assess).
- Controls: **1000** matched-random draws (AGENTS rule 6; V61 used 100, p floor 0.0099; with 1000 the floor is 0.000999).
- Gate: instrument-validity gate v3 (null-calibrated), including the known-bad control. A failed gate closes the run `InstrumentNotValid`; no retuning (cf. V50, V55).
- Tasks T0 (zero-ablation drop), L (mean-ablation), P (same-number/minimal-pair donor patch and swap sufficiency), each "supported" under the doc 201 rule (both directional drops exceed control maxima and combined drop >= 0.1 logits). The 0.5-logit instrument floor from V55 is applied as in doc 201.
- Selection stability: report candidate-list Jaccard across a repeat on the same device (determinism) before any effect is read. V60 showed device/dtype can change selection; this run has one device, so it is reported, not controlled.

## 3. Decision rule (inherited, not new)
- "Replicates on second family" iff the gate is valid and T0, L, P are all supported.
- "Partially replicates" iff the gate is valid and T0 is supported but L or P is not; name which.
- "Does not replicate" iff the gate is valid and T0 is not supported.
- "InstrumentNotValid" otherwise (no effect read as evidence).
All four outcomes are reported with equal prominence. Nothing here raises the result to "circuit", "mechanism" or "general".

## 4. Pre-registered descriptive outputs
Per task: selected combined drop, control max, one-sided rank p, same-class-donor diagnostic, concentration (share of the joint effect carried by the top-4 units, as V58), layer distribution of selected units, and cross-model comparison to Pythia-2.8B as a table only (no pooled test).

## 5. Abort conditions
Weight or corpus digest mismatch; locked candidate list changed after tune/assess started; gate v3 known-bad control not rejected; wall-clock cap 8 h per stage exceeded (reported NotCompleted, not retried); any model/dataset download not covered by section 8.

## 6. Follow-on arms (separate protocols, not part of R6 unless frozen here)
These come from the triaged posts and each needs its own question, controls and approvals.
- **R6-B, abstraction-neuron embeddings.** Compare a 16k-neuron selected embedding against 1000 matched-random 16k subsets and a full-hidden-state baseline on word similarity and on sentence/retrieval (the claim reports it wins one, loses the other). Needs a small word-similarity benchmark download (owner approval).
- **R6-C, single-neuron "concept" claim.** Same gate on a named unit in a model that has HF weights. The posts use Qwen3.5-4B; only a GGUF (`openresearchtools/Qwen3.5-4B-GGUF`) is cached, and GGUF does not expose editable activations in this harness. Needs a Qwen3.5-4B HF download (~8-9 GB, owner approval and budget change; current download budget 10 GB total, 3.2 GB already used for Qwen3-1.7B).
- **R7, Neuron Tuning vs LoRA with an anchor control.** Claim: 3-19x less drift on unrelated text. `temporal-grid-safety` v2 found a protected-unit benefit that was almost entirely a lower learning rate, so any comparison needs an equal-effective-rate control. Training run; out of scope for R6.

## 7. Limits
One new model, one task family (templated subject-verb agreement), one corpus, one machine, single device and dtype, post-hoc relative to V54-V62, units chosen by this pipeline, no independent review. A "replicates" outcome supports a unit-level causal claim on two models, not a circuit and not agreement-selectivity (V58 found agreement-selective false). No claim about Qwen3.5 or the posts' specific units. Spend: USD 0.00; no provider call.

## 8. Owner decisions (resolved at freeze; items 1, 2 and 3 decided, item 4 open)
1. Confirm the subject is Qwen3-1.7B-Base (no download) rather than waiting for a Qwen3.5-4B HF download.
2. Confirm the 1000-draw control count (about 10x V61's compute for the control stage) or keep 100 for comparability with V61 and accept the 0.0099 p floor. Recommendation: 1000, because AGENTS rule 6 requires it.
3. Approve or defer R6-B (benchmark download) and R6-C (model download, budget increase).
4. Note the working tree currently has uncommitted R1b run artifacts (`experiments/r1b/*`, `results/r1b*`); R6 should start from a clean commit after those are resolved, and must not touch them.
