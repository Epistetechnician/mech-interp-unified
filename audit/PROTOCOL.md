# F2 Phase-1 protocol: can an auditor catch a planted-flaw instrument?

Status: preregistration, committed before any auditor is run on any evaluation case.
Slice: `mech-interp-unified` F2 Phase 1. Ceiling: `LocalDevelopmentUnreviewed`. Spend USD 0, no network,
no download, CPU only (numpy). Owner self-attestation is not claimed. Nothing here is independently reviewed.

## 1. Why the existing audit failed (read from committed results; no new measurement)

**R5 was blind** (`results/r5_planted_superposition.json`, `docs/protocols/R5-result.md`).
Its job audit counts features whose Pearson correlation with a neuron's linear pre-activation is >= 0.2.
For a linear neuron on i.i.d. standardized features that correlation equals the normalized weight
`w_jf / ||w_j||`. With n=20 features a typical normalized weight has sd 1/sqrt(20)=0.224, so about 19% of
weights clear 0.2 and `P(count >= 2)` is about 0.91 (computed from the normal approximation, matches the
observed 0.87-0.97). The metric therefore measures neuron width, not superposition: it is saturated, and it
reports random-weight nets (0.95-0.97) the same as trained ones. Also d=10 < n=20 meant the S=0 "control" was
never a no-superposition case. Result: M(0.95)-M(0.0)=0.13, H1 fails; threshold 0.4 collapses to 0.03-0.24.

**R5b is not validated** (`results/r5b_causal_audit.json`, `docs/protocols/R5b-result.md`).
(i) The share rule (job iff damage share >= 0.2) sends neurons whose damage is spread over many features
to "zero jobs": at S=0.9/0.95 audit-multi is 0.01/0.03 while ground-truth multi is 0.28/0.26 (neuron-level
sensitivity 0.61, specificity 0.94). (ii) The cutoff is arbitrary: 0.1 gives about 1.0, 0.3 gives about 0
on the same models. (iii) The "clean" control is not clean: trained n=d=10 dense nets have ground-truth multi
0.49, so the H3 bound 0.20 was unreachable. (iv) The ground truth (second-largest squared-weight share
>= 0.15) is a weight proxy, not a causal quantity, so part of "not validated" is that the yardstick is soft.
(v) H2 passed on specificity alone. Verdict stands: audit not validated, thresholds unchanged.

**What F2 adds.** Both failures are instances of one question the repo never measured directly: given
instruments with a *known* planted defect and a *known-good twin*, how often does each audit flag the
defect (TPR) and how often does it flag the twin (FPR)? Phase 1 measures that on synthetic nets only.

## 2. Unit of analysis

A **case** = one instrument run, with ground truth `flawed in {0,1}` fixed by construction.
Each flaw class has a known-bad instrument and a known-good twin built from the same seed (same world,
same data, same controls) differing only in the planted defect.
An **auditor** receives the case and returns `FLAG` (do not accept this instrument's conclusion as evidence),
`PASS`, or `N/A` (outside its interface; excluded from rates, never counted as a miss silently).
Honest negative conclusions (an instrument correctly reporting `NoCandidate`) must PASS.

Instrument output (`Report`): claimed verdict (`Supported` / `NoCandidate`), selected unit ids, selected
effects `sel{sing,plur,all}`, matched-random control draws `ctrl{sing,plur,all}` (n=1000, size-matched,
drawn from non-selected units), reported p, floor, author-supplied known-bad control, provenance
(fit/assess item ids, applied criterion), baseline metric and metric ceiling. An instrument is
re-executable: `run(probe)` accepts probe overrides (shuffled readout, dtype, function-preserving
transform, fresh assess items, fresh control seed, prereg criterion). Only auditors that are *defined* to
re-execute may call it.

## 3. Flaw taxonomy (known-bad / known-good twin)

Substrate: calibrated one-hidden-layer ReLU nets (D=16 inputs, H=32 units, logit-margin readout,
two item classes `sing`/`plur` = label sign), per-unit effects set by construction. Effects are additive in
the ablated set, so ground truth is analytic. Selection = top-k (k=8) by fit-set single-unit ablation score,
assessed on held-out items against 1000 size-matched random unit sets.

| # | Class | Known-bad | Known-good twin | Inspired by (composed-zk mutation class) |
|---|---|---|---|---|
| 1 | `label_independent_effect` | metric and selection use label-free output shift; decoy units look causal; shuffled-label readout moves the metric as much | label-conditional margin; shuffled readout collapses | `semantic_no_op_drift` inverted (change that should matter does not alter the output) |
| 2 | `selection_leakage` | assess items = fit items on a null model with small n. Variant `visible`: provenance shows overlap. Variant `hidden`: provenance claims disjoint | held-out assess items | `stale_state_reads`, `witness_aliasing` |
| 3 | `device_dtype_selection` | near-tied unit effects (1e-4 relative), selection flips under reduced precision | well-separated effects (6% gaps), stable | `nondeterministic_transition_injection` |
| 4 | `representation_relative_identity` | selection by incoming weight norm (not invariant to function-preserving rescale + permutation) | selection by ablation effect (invariant) | `public_private_boundary_mismatch` |
| 5 | `planted_superposition` | hand-built privileged-basis nets with exact multi-job neurons. Variant `pairs`: 2 features/neuron. Variant `diffuse`: 8 features/neuron (damage share about 0.125 < 0.2) | hand-built exact monosemantic net (scaled permutation) | `missing_constraints` |
| 6 | `saturated_behavior_metric` | accuracy metric with baseline accuracy exactly 1.0 (rank-1 task), effect invisible | same metric, non-saturated baseline, effect visible | `invariant_weakening` |
| 7 | `post_hoc_criterion_change` | weak-effect world fails the preregistered floor. Variant `relax_floor`: floor lowered after seeing result, logged. Variant `switch_metric`: reports a more favourable metric on the same scale, provenance claims the preregistered one | honest `NoCandidate` under the preregistered criterion | `invariant_weakening`, `corrupted_guards` |
| 8 | `control_omission` | top 5% of control draws dropped and the array padded back to n=1000 with copies of retained draws | full controls | `observation_omission`, `missing_constraints` |

Each good twin also receives a semantic no-op perturbation (item order shuffled, control seed changed),
so FPR is measured under nuisance variation, not only on identical inputs.
Variants within a class are pooled with equal weight (equal n). Instruments that are defective but whose
claim is nonetheless correct are still `flawed`; a secondary metric restricts to instances whose claim is
wrong (`claim_correct == False`) where that is defined by construction.

Not covered in Phase 1 (stated so absence is not read as coverage): real language models, real device/MPS
differences (dtype is emulated), multi-layer effects, trained SAEs, adversarial agents (Section 8,
`audit/GOODHART.md`), flaws not in this list.

## 4. Auditors under test (frozen)

- **A1 `rules_gate`** (INDEX.md section B; Lean `Gate.lean`). Uses only the report and the frozen
  preregistered floor (`0.5` margin units; `0.05` accuracy for class 6; never the report's own floor).
  FLAG iff any of: (a) control arrays not length n_draws or n_draws < 1000; (b) reported p differs from
  `(1+#{ctrl.all >= sel.all})/(1+n)` or is below `1/(1+n)`; (c) the author-supplied known-bad control is
  `Supported`; (d) recomputed verdict differs from claimed verdict. Recomputed verdict: `Supported` iff
  `sel.sing > max(ctrl.sing)` and `sel.plur > max(ctrl.plur)` (strict) and `sel.all >= floor`, else `NoCandidate`.
  This checks decision arithmetic, not instrument validity (INDEX.md: "does not prove ... intervention
  correctness"). Expected to be blind except where a defect changes the arithmetic.
- **A2 `r5b_audit`** (`experiments/r5b_causal_audit.py::run_model`, imported unchanged). Applicable only to
  class 5 (needs a `(W,b,n_feat,S)` toy model). Reports the share-audit multi-job fraction over used neurons
  at share 0.2; FLAG iff that fraction > 0.20 (R5b's own H3 bound; the instrument claims a clean basis).
  `N/A` for all other classes. Not retuned.
- **A3 `probe_battery`**: re-executes the instrument. FLAG iff any check fires; thresholds frozen here:
  1. A1 flags.
  2. prereg consistency: applied metric id != preregistered, or applied floor < preregistered, or applied statistic != `max`.
  3. split disjointness: provenance fit ids intersect assess ids.
  4. shuffled readout (only if claimed `Supported`): shuffled-label `sel.all` >= 0.5 x claimed `sel.all`.
  5. fresh assess (only if claimed `Supported`): rerun on fresh items; fail strict exceedance or floor.
  6. dtype stability: selection under float16 or float64 has Jaccard < 0.75 with the claimed selection.
  7. equivariance: under a random function-preserving permutation + per-unit positive rescale (alpha in [0.25, 4]),
     the selection mapped back has Jaccard < 0.75 with the claimed selection.
  8. control redraw: reported control max < the 99th percentile of 1000 freshly redrawn controls, auditor-drawn with the standard matched-random procedure (no instrument-side trimming), for any of sing, plur, all [amended, see section 9].
  9. headroom: metric has a ceiling and baseline >= ceiling - 0.01.
  10. preregistered-criterion rerun: rerun under the preregistered metric and floor gives a different verdict than claimed.
  The probes were written *after* the flaw list and cover its classes; the battery is therefore a
  reference-style auditor whose TPR on these classes is upper-bounded by design, not evidence of general
  audit capability. Unprobed flaws are by definition untested.
- **A4 `llm_judge`**: prompt and harness are implemented, not run (no model call is possible without a paid
  API). A scripted stub exists only to unit-test parsing. No TPR/FPR is reported for A4.
- **Null auditors (instrument-validity controls for the harness)**: `always_pass`, `always_flag`,
  `coin_flip(p=0.5, seeded)`. Harness is valid only if always_pass has TPR=FPR=0, always_flag has
  TPR=FPR=1, and coin_flip's pooled TPR and pooled FPR Wilson 95% CIs contain 0.5 with every per-class point
  estimate within 0.5 +/- 0.15 [amended from per-class CIs, see section 9]. Otherwise all results are void.

## 5. Primary metrics and decision rules

Per (auditor, class): TPR = flagged / flawed cases; FPR = flagged / good cases; 95% percentile bootstrap CI
(B=10000, fixed rng seed, resampling cases, stratified by variant), Wilson 95% reported alongside because
bootstrap intervals degenerate at 0 and 1. Secondary: per-variant TPR; TPR restricted to
`claim_correct == False`; balanced accuracy; macro mean over applicable classes (descriptive only).

Preregistered labels per (auditor, class):
- **DETECTS**: TPR lower 95% bound >= 0.80 and FPR upper 95% bound <= 0.10.
- **BLIND**: TPR point estimate < 0.30.
- **PARTIAL**: everything else with N/A excluded. A cell with FPR upper bound > 0.10 is additionally marked **NOISY**.

No hypothesis tests, so no multiplicity correction; labels use uncorrected 95% intervals and are descriptive.

**What counts as a negative.** Any applicable (auditor, class) not labelled DETECTS is a reported miss with its
interval. The F2 hypothesis "an auditor available in this repo catches broken instruments" is *negative* for
A1 if A1 is DETECTS on fewer than 6 of 8 classes, negative for A2 if it is not DETECTS on class 5 pooled
over variants, and the combined statement "no single auditor covers all classes" is negative if no auditor
is DETECTS on all 8. Failure to detect is a valid result and is reported with the same prominence.
Predictions, recorded before running and not decision-bearing: A1 DETECTS only `post_hoc_criterion_change`
variant `relax_floor`, otherwise BLIND with FPR near 0; A2 high TPR on `pairs`, low on `diffuse` (the R5b
share-spreading blind spot, now on exact ground truth); A3 DETECTS most classes by construction.

## 6. Design and seeds

N = 200 seeds per (class, variant, arm) initially (amendable only before the first evaluation run, see
section 9). Evaluation seeds 0..N-1. Development seeds are >= 100000 and are never used for evaluation.
Case randomness: `numpy.random.SeedSequence([class_id, variant_id, arm_id, seed])`; twins share the world seed.
Matched-random controls: 1000 draws (AGENTS rule 6), size k, uniform over non-selected units.
Instrument validity of the audit harness itself is checked by the null auditors above.
`audit/verify` re-derives every count from seeds and compares to `audit/results/` byte-for-byte on counts.

## 7. Governance

No GPU, no gman, no Modal, no download. Raw per-case arrays are not committed (digests and aggregates only).
No claim above `LocalDevelopmentUnreviewed`. `imports/`, `docs/INDEX.md`, `bench/transfer/` untouched.
Thresholds in sections 4-5 are frozen: after any evaluation result is seen, a failed gate closes the run
and is reported; it is not retuned. Abort rules: runtime over 60 minutes, or harness-validity check fails.

## 8. Adversarial-agent condition

Specified separately in `audit/GOODHART.md`, preregistered there.

## 9. Amendment log

Development calibration of the *instrument side* (world parameters in section 3) may use development seeds
and may adjust world parameters, never auditor definitions or thresholds, and is recorded here with its commit
before the first evaluation run. No auditor TPR/FPR is computed on development seeds for tuning.
Amendments made after dev-seed runs (seeds >= 100000, n=40) and BEFORE the first evaluation run. Disclosure: items
2-3 touch auditor-side definitions and were motivated by dev-seed numbers, which section 9 forbids for tuning; they
are logged here instead of hidden, and the REPORT discusses them.
1. World parameters (allowed): ablation is mean-ablation with per-unit effect calibration except class 6 (zero
   ablation, sigma_good=1.2); class 1 uses 8 decoys with decoy_d=8; class 2 n_fit=n_assess=16; class 7 weak profile
   0.06 x STRONG; class 8 one weak carrier per class (effect ~U(0.02,0.10) per seed) plus 30 noisy inert units,
   prereg floor 0.05; selection is quota (k/2 units by class-0 score, k/2 by class-1 score).
2. Author-supplied known-bad control is a random size-k set drawn from the non-selected pool (previously drawn from all
   units, which sometimes contained true carriers and made the gate FPR 0.35-0.57 for an instrument-side bug).
3. Probe 8 redraws controls via a plain auditor-side procedure (the instrument's own re-execution reproduced the trim)
   and also checks the `all` statistic (the p-value statistic); with sing/plur only, class-8 TPR was 0.2 on dev seeds.
4. Harness-validity coin-flip rule changed from 16 per-class CIs (all-contain-0.5 has probability about 0.44 for a fair
   coin) to the pooled rule above.
