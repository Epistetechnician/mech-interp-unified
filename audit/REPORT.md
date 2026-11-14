# F2 Phase-1 report: can an auditor catch a planted-flaw instrument?

Claim ceiling `LocalDevelopmentUnreviewed`. Synthetic nets only, CPU, USD 0, no download. Owner-run, not independently
reviewed. Protocol: `audit/PROTOCOL.md` (committed before any measurement; amendments logged in its section 9).
Results: `audit/results/f2_phase1.json` (evaluation, seeds 0-199, 200 per class/variant/arm),
`audit/results/f2_goodhart_scripted.json`. Reproduce: `python3 -m audit.verify` (re-derived exactly, run twice, exit 0),
`python3 -m pytest audit/tests`.

## Why R5 was blind and R5b is not validated (from committed results, nothing re-run)
- **R5**: the job count is the number of features with |normalised weight| >= 0.2. For n=20 features that is true of about
  19% of weights per feature, so P(>=2 jobs) is about 0.91 for any weight vector, including random ones (observed 0.95-0.97).
  The metric measures neuron width, not superposition; the d=10 < n=20 control was not a no-superposition case.
- **R5b**: the share rule (job iff damage share >= 0.2) sends neurons whose damage is spread over many features to
  "zero jobs", so at S=0.9/0.95 it reports 0.01/0.03 multi-job against 0.28/0.26 ground truth; the cutoff is arbitrary
  (0.1 gives ~1.0, 0.3 gives ~0); the n=d=10 control already has ground-truth multi 0.49; H2 passed on specificity alone.
  F2 re-measures the R5b blind spot below on exact, constructed ground truth.

## Setup in one paragraph
Eight flaw classes, each a known-bad instrument with a known-good twin from the same seed. Twins also get semantic no-ops
(item order, control seed). Auditors: A1 rules gate (strict exceedance of 1000-draw matched-random maxima + floor + p-floor
1/(1+n), against the preregistered floor), A2 the unchanged R5b audit, A3 a probe battery that re-executes the instrument,
plus null auditors for harness validity. A4 LLM judge: prompt, parser and wrapper exist (`audit/auditors/llm_judge.py`),
**not run** (needs a paid model call; no USD ceiling granted). Labels: DETECTS = TPR lower bound >= 0.80 and FPR upper
bound <= 0.10; BLIND = TPR < 0.30; else PARTIAL; +NOISY if FPR upper bound > 0.10. Intervals are percentile bootstrap (B=10000);
Wilson intervals are in the JSON and agree to within about 0.03 except at 0/1, where the bootstrap is degenerate.
Harness validity: always_pass TPR=FPR=0, always_flag TPR=FPR=1, coin_flip pooled CIs contain 0.5: all passed.

## Headline (preregistered decision rules)
- **A1 rules gate is negative**: DETECTS on 0 of 8 classes (rule: negative if < 6). It flags only the `relax_floor` variant
  of the post hoc change (TPR 1.0, via verdict_mismatch against the preregistered floor) and nothing else. FPR 0 everywhere.
  It checks decision arithmetic, which the flaws leave consistent; this matches INDEX.md section B ("does not prove ...
  intervention correctness").
- **A2 R5b audit is negative on class 5**: TPR 0.65 [0.62, 0.68], FPR 0.00, label PARTIAL. Pairs (2 features per neuron) 1.00;
  diffuse (8 features per neuron, share ~0.125) 0.30. It catches shared neurons and misses diffuse ones, confirming on
  exact ground truth the failure mode diagnosed in R5b. The good twin never trips it.
- **A3 probe battery DETECTS 3 of 8 classes** (label-independent, representation-relative identity, post hoc change), is
  PARTIAL on device/dtype (0.84), selection leakage (0.80), and PARTIAL+NOISY on saturated metric (FPR 0.10) and control
  omission (FPR 0.23). N/A on class 5.
- **No auditor DETECTS all 8**: negative for the combined statement "an auditor available here catches broken instruments".
  The union of A1-A3 covers 3 of 8 classes at the DETECTS bar (A3's three; A1 and A2 add none).
- A3 was written after the flaw list and has one probe per class, so its TPR here is an upper bound on its general ability.
  It is a reference point, not evidence that the problem is solved.

## Results (TPR/FPR with bootstrap 95% CI)

**A1 rules gate** (n per cell: 200 seeds x variants; bootstrap 95% CI, Wilson in JSON)

| class | TPR | FPR | label | TPR where claim is wrong | n/a |
|---|---|---|---|---|---|
| control_omission | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | BLIND | 0.00 (0/10) | 0 |
| device_dtype_selection | 0.005 [0.000, 0.015] | 0.000 [0.000, 0.000] | BLIND | n/a | 0 |
| label_independent_effect | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | BLIND | 0.00 (0/200) | 0 |
| planted_superposition | n/a | n/a | N/A | n/a | 800 |
| post_hoc_criterion_change | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | PARTIAL | 0.50 (200/400) | 0 |
| representation_relative_identity | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | BLIND | n/a | 0 |
| saturated_behavior_metric | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | BLIND | 0.00 (0/200) | 0 |
| selection_leakage | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | BLIND | 0.00 (0/400) | 0 |

**A2 R5b audit** (n per cell: 200 seeds x variants; bootstrap 95% CI, Wilson in JSON)

| class | TPR | FPR | label | TPR where claim is wrong | n/a |
|---|---|---|---|---|---|
| control_omission | n/a | n/a | N/A | n/a | 400 |
| device_dtype_selection | n/a | n/a | N/A | n/a | 400 |
| label_independent_effect | n/a | n/a | N/A | n/a | 400 |
| planted_superposition | 0.650 [0.618, 0.682] | 0.000 [0.000, 0.000] | PARTIAL | n/a | 0 |
| post_hoc_criterion_change | n/a | n/a | N/A | n/a | 800 |
| representation_relative_identity | n/a | n/a | N/A | n/a | 400 |
| saturated_behavior_metric | n/a | n/a | N/A | n/a | 400 |
| selection_leakage | n/a | n/a | N/A | n/a | 800 |

**A3 probe battery** (n per cell: 200 seeds x variants; bootstrap 95% CI, Wilson in JSON)

| class | TPR | FPR | label | TPR where claim is wrong | n/a |
|---|---|---|---|---|---|
| control_omission | 1.000 [1.000, 1.000] | 0.230 [0.175, 0.290] | PARTIAL+NOISY | 1.00 (10/10) | 0 |
| device_dtype_selection | 0.840 [0.790, 0.890] | 0.005 [0.000, 0.015] | PARTIAL | n/a | 0 |
| label_independent_effect | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | DETECTS | 1.00 (200/200) | 0 |
| planted_superposition | n/a | n/a | N/A | n/a | 800 |
| post_hoc_criterion_change | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | DETECTS | 1.00 (400/400) | 0 |
| representation_relative_identity | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | DETECTS | n/a | 0 |
| saturated_behavior_metric | 1.000 [1.000, 1.000] | 0.100 [0.060, 0.145] | PARTIAL+NOISY | 1.00 (200/200) | 0 |
| selection_leakage | 0.797 [0.765, 0.830] | 0.007 [0.000, 0.018] | PARTIAL | 0.80 (319/400) | 0 |

**Per-variant TPR (FPR)**

| auditor | class/variant | TPR | FPR |
|---|---|---|---|
| rules_gate | control_omission/trim5pct | 0.00 | 0.000 |
| rules_gate | device_dtype_selection/near_ties | 0.01 | 0.000 |
| rules_gate | label_independent_effect/decoy | 0.00 | 0.000 |
| rules_gate | post_hoc_criterion_change/relax_floor | 1.00 | 0.000 |
| rules_gate | post_hoc_criterion_change/switch_metric | 0.00 | 0.000 |
| rules_gate | representation_relative_identity/weight_norm | 0.00 | 0.000 |
| rules_gate | saturated_behavior_metric/accuracy_ceiling | 0.00 | 0.000 |
| rules_gate | selection_leakage/hidden | 0.00 | 0.000 |
| rules_gate | selection_leakage/visible | 0.00 | 0.000 |
| r5b_audit | planted_superposition/diffuse | 0.30 | 0.000 |
| r5b_audit | planted_superposition/pairs | 1.00 | 0.000 |
| probe_battery | control_omission/trim5pct | 1.00 | 0.230 |
| probe_battery | device_dtype_selection/near_ties | 0.84 | 0.005 |
| probe_battery | label_independent_effect/decoy | 1.00 | 0.000 |
| probe_battery | post_hoc_criterion_change/relax_floor | 1.00 | 0.000 |
| probe_battery | post_hoc_criterion_change/switch_metric | 1.00 | 0.000 |
| probe_battery | representation_relative_identity/weight_norm | 1.00 | 0.000 |
| probe_battery | saturated_behavior_metric/accuracy_ceiling | 1.00 | 0.100 |
| probe_battery | selection_leakage/hidden | 0.59 | 0.015 |
| probe_battery | selection_leakage/visible | 1.00 | 0.000 |

**Scripted adversary (GOODHART C0)**

| target/adversary | successes | rate [boot 95% CI] | median queries | component | transfer flag rate |
|---|---|---|---|---|---|
| probe_battery/es | 4/50 | 0.080 [0.020, 0.160] | 121.0 | partial | 0.0 |
| probe_battery/random | 1/50 | 0.020 [0.000, 0.060] | 67.0 | robust | 0.0 |
| rules_gate/es | 50/50 | 1.000 [1.000, 1.000] | 17.5 | broken | 1.0 |
| rules_gate/random | 50/50 | 1.000 [1.000, 1.000] | 5.0 | broken | 1.0 |

Decision: {'probe_battery': 'partially_robust', 'rules_gate': 'broken'}


## Reading the numbers
- **Selection leakage, hidden variant (A3 TPR 0.59)**: probes 4/5 only run when the claim is `Supported`; 81 of 200 leaky
  instruments honestly claimed `NoCandidate`, and a flawed instrument whose conclusion happens to be negative is not
  detected. The visible variant (overlapping ids in provenance) is caught 200/200 by a set intersection.
  Hidden-variant TPR 0.59 = 119/200, exactly the leaky instruments whose claim was `Supported` and failed the fresh-assessment rerun.
- **Device/dtype (0.84)**: dtype probe fires on 167/200; near-tied worlds where float16 rounding happens not to flip the
  selected set are missed. One rules-gate FLAG on a bad case is a chance exceedance of the known-bad control.
- **Control omission**: probe 8 (auditor-side control redraw) flags 200/200 trimmed instruments, but FPR is 0.23, all from probe 5
  (fresh-assessment rerun) on good twins whose weak effects genuinely do not replicate on fresh items. That is an honest
  fresh-data failure, but it means the twin is not a clean positive; treat the 0.23 as the cost of a marginal-effect world.
- **Saturated metric FPR 0.10**: same cause (probe 5 on a good twin whose accuracy effect is marginal; only 43/200 good
  twins claim support). The good twin is not reliably a positive instrument.
- **Post hoc, `switch_metric` (A1 TPR 0)**: the gate recomputes the verdict from reported numbers, which already come from the
  switched metric; only a rerun under the preregistered metric (A3 probe 10) or the shuffled-readout probe exposes it.
- A2 is applicable to one class by construction and A1/A3 to seven; n/a cells are excluded, not counted as misses, but they
  are uncovered ground.

## Adversarial condition (GOODHART.md, scripted proxy only)
Black-box random search and a gray-box (1+1) evolution each had 200 queries per world, 50 worlds (seeds 900-949), searching
the instrument's configuration space for a `Supported` claim resting on mostly non-carrier units that the target auditor passes.
Results above: **A1 broken** (50/50 for both adversaries, median 5 / 17.5 queries); **A3 not broken** in this space (random 1/50,
gray-box 4/50 [0.02, 0.16], decision "partially robust"; all A1-beating configurations were flagged by A3, transfer 1.0, while
A3-beating configurations were passed by A1, transfer 0.0). Caveats: the adversary writes the preregistration, so A1's frozen
floor is whatever it registers; the gray-box adversary sees A3's fired-check count; the knob space is small and hand-built; the two A3-beating configurations inspected (stored in the JSON) both combined `assess=fit` with hidden overlap logging and a top-decile applied metric, which the probes cover only through the shuffled-readout and fresh-assessment checks. Some knob settings (n_fit_use=2) produce
empty classes and NaN effects; those count as adversary failures. The LLM-agent condition (C1) was **not run**.

## What the audit still misses
- Anything outside these eight classes and this substrate: real language models, real MPS/CUDA numerics (dtype is emulated by
  rounding), multi-layer effects, trained SAEs, dataset contamination, incorrect intervention code.
- Diffuse superposition for the existing R5b audit (70% missed), and any superposition for A1/A3 (no interface; class 5 is N/A).
- Leakage in instruments that report a negative result; flawed-but-honestly-negative instruments are invisible to claim-conditional probes.
- The probes are mine and were written against this list, so unknown flaws are untested by construction; the scripted adversary
  is not an LLM or trained agent.
- Worlds are linear-readout ReLU nets with additive effects; effect sizes and good-twin quality (class 6 and 8 twins are marginal)
  were set during development, and the amendments in PROTOCOL.md section 9 (items 2 and 3 touch auditor-side definitions after
  seeing development-seed numbers) weaken the preregistration. Results should be read with that disclosed.

## What remains
1. Run the LLM judge (A4) and C1 agent condition: needs an owner-stated USD ceiling and provider identity (AGENTS.md rule 4).
2. Port a subset of classes to a small cached real model (CPU/MPS) to test whether the planted-flaw TPR/FPR transfers.
3. Repeat with an auditor written by someone other than the flaw author, before any claim about auditor quality.
4. Fix good-twin quality for classes 6 and 8 (marginal positives) in a new preregistered round rather than retuning this one.
5. A2 variant with a share cutoff chosen from a null-calibrated control, as the R5b result recommends; not tried here to avoid retuning.
