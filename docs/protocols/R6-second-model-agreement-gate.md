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

## Addendum 1 (2026-11-03, written while implementing the runner; no model run, no data read)
This corrects the frozen text where it misdescribed the imported pipeline. Original text above is left as committed (`3339a2a`); where it conflicts with this addendum, the addendum governs. Changes are made before any code touched a model or corpus.
1. **Layers are 8, 15, 22, not 9, 16, 23.** The repo's own rule (`pythia_plan.layers_for`, depths 0.3/0.55/0.8, rounded) gives [10, 18, 26] for 32 layers (V61) and [8, 15, 22] for 28. My scaling of V61's layer numbers was wrong. Enforced in `experiments/r6/r6.py` and tests.
2. **T0 / L / P are class-swap and donor-patch tasks, not ablations.** As in doc 201: T0 = all-position class swap of the 8 swap units to the opposite-class fit mean; L = the same at the last position only; P = last-position donor activation patching (donor seed 6060, same-class diagnostic seed 6061). "Zero-ablation" and "mean-ablation" in section 2 are retracted. Decision rule, 0.1-logit combined-drop floor and the both-directions-exceed-control-max support rule are exactly doc 201's.
3. **Gate v3 composition.** Run on the 96 locked candidates, tune split, before any rank/assess read: 1000 matched-random draws (candidate-size sets) for margin and logprob p-values; `instrument_gate_v3` (v50 gate v2 checks: 0.5-logit effect floor, positive control, near-miss, activation shape, two readouts at alpha 0.05; plus V51 null-calibrated known-bad: 20 sign-flip draws, z_null 4.0, z_signal 10.0). Any failure closes the run `InstrumentNotValid` with no T0/L/P read. Thresholds are the imported frozen values, unchanged.
4. **Determinism report.** Selection is run twice on fit; Jaccard of the two candidate lists is reported (same device and dtype, so this is a determinism check, not a robustness test).
5. **Cost and caps.** About 2 x 1000 gate evaluations plus 3 x 1001 task evaluations (about 5000 passes over 800 items). The 8 h cap applies per stage (gate, T0, L, P each separately); a stage over cap yields `NotCompleted`, not a retry. Estimated wall clock is many hours on this Mac; run unattended.
6. **Known blocker, owner decision needed.** The V54 corpus items (raw-text `split_items`) are not in this repo and were not found on this machine. `imports/astral-neuron/runs/v54-run/corpus-public.json` holds only public aggregates. Rebuilding it fetches about 38 Project Gutenberg books (`build_corpus_v54.py`, curl, spaCy), which `docs/AUTHORIZATION.md` does not cover. If the original corpus exists elsewhere (owner-only root or the backup volumes), point the runner at it and verify its manifest digest against `59c514f0...` (the V61 digest). Otherwise the owner must approve the Gutenberg download. Until then R6 cannot run.
7. **Unused fields.** `unit_stats` also returns DLA fields that assume LayerNorm centering and are invalid for Qwen's RMSNorm; R6 reads only `act_sing`/`act_plur`.
Implementation: `experiments/r6/` (`r6.py`, `qwen3_backend.py`, `run_r6.py`), tests `tests/r6/test_r6.py` (15 hermetic tests, including a tiny random Qwen3 that checks the hook point equals the input of `down_proj`).

## Addendum 2 (2026-11-03, after first Modal run crashed in the gate; no result read)
Run 1 (Modal ap-CSyi1u..., 16:37-18:33 EST) completed 1000 gate controls, then crashed in `instrument_gate_v3` -> `activation_values`: the imported Pythia version hooks `mlp.dense_h_to_4h` (absent in Qwen3). No effect was read; no outcome exists. A second launch (ap-uEom6y..., 17:38) was a duplicate of the same buggy code and was left running by mistake of mine (I launched without checking for a run in flight); it is expected to crash at the same point.
Fix (`experiments/r6/qwen3_backend.py`): `activation_values` overridden to read the last-position activation at the measurement hook point (input of `down_proj`, silu(gate)*up), because SwiGLU has no single pre-activation. This is a deviation from the Pythia gate's "activation shape" input (pre-GELU value); it is signed, so the both-signs check stays meaningful. Frozen config, thresholds, layers, draws, decision rule unchanged. Regression test added (16 hermetic tests pass) and a tiny-model smoke of the full run path (real gate, then forced-valid gate through T0/L/P) passes.
Spend note: runs 1 and 2 consumed roughly USD 4 of the Addendum 3 allocation of USD 7.00 (Modal billing is the receipt). A full rerun (about 6 h worst case) would exceed that allocation, so it needs a new owner statement; none has been given.

## Addendum 3 (2026-11-03): partial-result checkpoints, runtime estimate
- Runner change (no frozen value touched, digest guard unchanged): `run_r6` accepts an optional `checkpoint` callback, called after the gate, after rank, after T0, after L and after P. `run_r6.py` writes `<out>.partial` each time and the Modal launcher commits the Volume, so a timeout or crash after the gate keeps every finished stage. The final result file is unchanged.
- Runtime estimate from run 1 (gate = 2000 forwards over 800 tune items in about 6600 s, so about 3.3 s per forward on A10G): rank about 0.3 h, T0/L/P about 3000 forwards over 800 assess items, about 2.8-4.2 h. Total about 5-6.3 h against the 6 h function timeout. A timeout is reported `NotCompleted` with the finished stages saved; it is not extended.
