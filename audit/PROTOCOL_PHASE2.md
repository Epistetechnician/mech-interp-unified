# F2 Phase-2 protocol: does the frozen probe battery catch flaw classes it was not written for?

Status: preregistration, committed before any Phase-2 flaw code exists and before any auditor is run on any
Phase-2 case. Slice: `mech-interp-unified` F2 Phase 2. Ceiling: `LocalDevelopmentUnreviewed`. Spend USD 0, no
network, no download, CPU only (numpy). Not independently reviewed. Drafted by an AI agent session at the
owner's request ("go" on the roadmap item "planted-flaw benchmark ... probe battery tested on flaw classes it
was not written for", 2026-11-14).

## 1. Why

Phase 1 (`audit/REPORT.md`) found that A3 `probe_battery` DETECTS 3 of 8 classes, and noted that A3 "was
written after the flaw list and has one probe per class, so its TPR here is an upper bound". Phase 2 measures
the other direction: A1 and A3, **unchanged**, on flaw classes written after them.

## 2. How the new classes were chosen (and the bias that remains)

Selection rule, fixed here: a class is admitted only if it restates a confound that actually occurred and is
documented in one of the owner's own write-ups dated before this protocol, and it can be expressed in the
Phase-1 substrate (`audit/worlds.py`, `audit/pipeline.py`) by adding instrument-side options whose defaults
leave every Phase-1 case byte-identical (`python3 -m audit.verify` must still pass).

Disclosed bias: the author of the classes had read the ten A3 probes before writing them. The rule above limits,
but does not remove, the freedom to pick classes by whether A3 would catch them. Three candidate confounds were
considered and dropped because the additive substrate cannot express them without a new world: the clean-margin
level shift (jspace `level_adjust.py`), proxy-versus-behaviour disagreement (jspace PCA arm), and selection that
depends on device (already Phase-1 class 3).

## 3. New classes (known-bad / known-good twin)

Evaluation seeds 0-199, 200 per (class, variant, arm); development seeds >= 100000, never used for evaluation.
Same twin and nuisance-perturbation rules as Phase 1 section 3 unless stated.

| # | Class | Source confound | Known-bad | Known-good twin |
|---|---|---|---|---|
| 9 | `unmatched_controls` | jspace-pilot: rank-matched PCA removed a different norm than J-ablation (`baseline_writeup.md`); hsi-lab E6: gain partly pool size (`docs/REPORT.md`); temporal-grid-safety v16: the adversary inherited each arm's own rates (`REPORT.md`) | Class-8 world (one weak carrier per class, 30 noisy inert units, floor 0.05). Controls are reported as 1000 size-matched random draws but are not. Variant `size`: control sets have k/2 units. Variant `pool`: control sets are drawn only from the half of non-selected units with the lowest fit-set score. Variant `condition`: control effects are computed under a shuffled readout while the selected effect uses the true one | standard matched-random controls |
| 10 | `underpowered_null` | jspace-pilot: ablation efficiency about 0.5, "a null at high ablation efficiency remains untested" (`baseline_writeup.md`); temporal-grid-safety v1 floor effect | STRONG world, floor 0.5. The intervention removes only a fraction `eff` of each unit's contribution (selected and control sets alike), so the instrument reports `NoCandidate` although the full effect clears the floor. The report has no efficiency field | A world whose carrier profile is `eff` x STRONG with full ablation: an honest `NoCandidate` with a statistically matched report. (Deviation from Phase 1: the twins differ in the world, not only in the instrument, so that the two reports look alike.) |
| 11 | `best_of_m` | temporal-grid-safety "gates were sometimes chosen after seeing earlier data" (`REPORT.md`); jspace 0.2-logit margin chosen after data | A world scaled so the full-data effect sits just under the floor. The instrument evaluates up to m=20 class-balanced random subsets of 64 assess items and reports the first that yields `Supported`; if none does it reports the full-data result. Provenance truthfully lists the items used. Re-execution repeats the same procedure | same world, full assess set, no subset search |

Ground truth: every bad case is `flawed`. `claim_correct` / `honest_claim` is set as in Phase-1 class 8, by
rerunning with the defect removed, and the secondary TPR is restricted to cases whose claim is wrong. Class 10
bad cases have `claim_correct = False` by construction.

Instrument-side calibration allowed on development seeds, using instrument claims only and never an auditor
decision: `eff` for class 10 (target: bad arm `NoCandidate` in at least 0.95 of seeds while the same world at
full ablation is `Supported` in at least 0.95); the profile scale for class 11 (target: bad arm `Supported` in
0.4 to 0.8 of seeds, good arm `Supported` in at most 0.05). Chosen values are logged in section 7 before the
evaluation run.

## 4. Auditors (frozen, unchanged)

A1 `rules_gate` and A3 `probe_battery` exactly as in `audit/PROTOCOL.md` section 4, same thresholds, same code.
A2 is not applicable. A4 is not run. Null auditors and the harness-validity rule are those of Phase 1.

One interpretation is fixed here because it decides probe 8. Phase 1 defines the probe-8 redraw as
"auditor-drawn with the standard matched-random procedure (no instrument-side trimming)". In Phase 2
`plain_controls` therefore means: size k, drawn from all non-selected units, effects computed under the same
readout as the selected set. An instrument-side control defect of class 9 is not reproduced in that redraw.
This makes probe 8 a strong check on class 9 by definition, and class 9 should be read with that in mind.

## 5. Metrics and decision rules

TPR, FPR, intervals and the labels DETECTS / PARTIAL / BLIND / NOISY are those of Phase 1 section 5, per
(auditor, class), variants pooled with equal weight; per-variant rates are reported.

- **Primary.** "The frozen A3 generalises to held-out flaw classes" is supported iff A3 is DETECTS on at least
  2 of the 3 new classes. Otherwise it is negative, reported with the same prominence.
- **Secondary.** A1 is negative if it is DETECTS on fewer than 2 of 3.
- Any class where the bad arm's claim is wrong in fewer than 0.2 of seeds is reported as "defect rarely changes
  the claim" and its secondary TPR is not interpreted.

Predictions, recorded before any code and not decision-bearing: A1 BLIND on all three with FPR near 0. A3:
class 9 DETECTS or PARTIAL through probe 8; class 10 BLIND (no probe looks at power, and a `NoCandidate` claim
skips probes 4 and 5); class 11 BLIND or PARTIAL (the fresh-assess probe re-executes the search). Predicted
primary outcome: negative.

## 6. Governance

No GPU, no paid compute, no download. Raw per-case arrays are not committed. `imports/`, `docs/INDEX.md`,
`bench/transfer/` and the Phase-1 result files are untouched; `python3 -m audit.verify` must pass after the
pipeline gains the new options. Thresholds above are frozen; a failed gate closes the run and is reported.
Abort rules: runtime over 60 minutes, or the harness-validity check fails. Results go to
`audit/results/f2_phase2.json`, reproduced by `python3 -m audit.run_phase2`.

## 7. Amendment log

(Calibration values and any change made before the first evaluation run are recorded here with their commit.)
