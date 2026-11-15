# F2 Phase-2 report: the frozen auditors on flaw classes written after them

Claim ceiling `LocalDevelopmentUnreviewed`. Synthetic nets, CPU, USD 0, no download. Not independently reviewed;
protocol, flaw classes, code and this report were drafted by an AI agent session at the owner's request.
Protocol: `audit/PROTOCOL_PHASE2.md` (commit `0fc9778`, before any Phase-2 code). Flaw code and calibration:
commit `90b9127`, before the evaluation run. Results: `audit/results/f2_phase2.json` (seeds 0-199, 200 per
class/variant/arm, runtime 146 s). Reproduce: `python3 -m audit.run_phase2 --check`.

## Headline (preregistered rule)

**Negative: the frozen probe battery does not generalise.** A3 is DETECTS on 0 of the 3 held-out classes (rule:
supported if at least 2). A1 `rules_gate` is BLIND on all three with FPR 0. This matches the prediction recorded
in the protocol. Harness validity passed (always_pass 0/0, always_flag 1/1, coin_flip pooled CIs contain 0.5).

| Class | A3 label | TPR [95% bootstrap] | FPR [95%] | TPR where the claim is wrong | A1 |
|---|---|---|---|---|---|
| 9 `unmatched_controls` | PARTIAL + NOISY | 0.74 [0.71, 0.78] | 0.13 [0.11, 0.16] | 77/92 = 0.84 | BLIND, 0/600 |
| 10 `underpowered_null` | BLIND | 0.00 (0/200) | 0.00 | 0/200 | BLIND, 0/200 |
| 11 `best_of_m` | PARTIAL | 0.42 [0.35, 0.49] | 0.00 | 59/121 = 0.49 | BLIND, 0/200 |

Class 9 by variant (TPR / FPR): `pool` 0.905 / 0.13, `size` 0.77 / 0.17, `condition` 0.555 / 0.095.

## What each result says

- **Class 10 is the clean finding.** An instrument whose intervention removes 8% of each unit's contribution
  reports `NoCandidate` in a world where full ablation gives an effect of about 5.7 against a floor of 0.5. No
  auditor flags it in any of 200 seeds. The report has no efficiency field, a `NoCandidate` claim skips probes 4
  and 5, and re-execution reproduces the same weak intervention. Nothing in A1 or A3 asks whether the instrument
  could have detected an effect. This is the jspace-pilot situation (ablation efficiency about 0.5) on exact
  ground truth.
- **Class 11: re-execution repeats the search.** The bad arm reports `Supported` in 121 of 200 seeds, always
  wrongly. A3 catches about half of those: probe 5 (fresh assess) fired in 49 cases and probe 8 (control redraw)
  in 42. The other half pass because the fresh-item rerun runs the same subset search and finds another subset.
- **Class 9: probe 8 does the work, as the protocol said it would by definition.** It fired in 181, 145 and 102
  of 200 bad cases for `pool`, `size` and `condition`. The preregistered caveat applies: the defect changed the
  claim in only 92 of 600 bad cases (0.15, below the 0.2 bar), so the secondary TPR of 0.84 is reported but not
  interpreted.
- **A3's false positives come from probe 5 on honest instruments.** In the class-9 world the effect sits near the
  floor, so an honest `Supported` claim often fails on fresh items: 78 of the 79 good-arm flags are probe 5. That
  is a correct observation about a fragile claim and a wrong verdict about the instrument. The label NOISY stands.

## Limits

- Three classes, chosen by an author who had read the probes (protocol section 2). The direction of that bias is
  not known.
- Class 9 depends on one interpretation fixed in the protocol: the auditor's control redraw never reproduces a
  control-side defect. An auditor without an independent control procedure would do worse.
- Class 10's twins differ in the world, not only in the instrument (disclosed deviation), so that the reports match.
- Additive one-layer substrate; no real model. Three confounds from the source write-ups could not be expressed
  in it and were dropped (protocol section 2).
- Process deviation: the commit `90b9127` was made while one new unit test was failing. The failure was in the
  test (an exact float comparison across a reordered sum, difference 1e-16), not in the pipeline; it is fixed in
  the commit that adds this report. No evaluation number depends on it.

## What would close the gaps

A power check for null claims: a planted positive control of known size that the instrument must recover before a
`NoCandidate` is accepted (jspace's probe arm is one). For search over subsets or configurations: the auditor
re-executes on fresh items with the procedure's degrees of freedom fixed in advance, not replayed. Neither is
implemented here; adding them would be a new auditor under a new protocol, tested on classes written after it.
