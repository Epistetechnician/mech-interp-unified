# F1 extension protocol: runs that would add items (preregistered, NOT run)

Ceiling `LocalDevelopmentUnreviewed`. Nothing in this file has been run, downloaded or provisioned. Each experiment
below needs its own owner go-ahead; per `AGENTS.md` rule 4 no paid action may start without an owner-stated USD
ceiling in chat, a recorded provider/node identity and an allocation receipt, and **no ceiling is currently granted**.
Do not use gman nodes `playbox-tp3` or `jspace-*` (other work). gman commands need the sandbox bypass.

Rate: gman H100 **$5.04/h** (measured, `$0.084/min`). Time figures are marked *measured* (read from a source file) or
*estimate* (scaled by the author, not run).

## Why these three

The bench has 36 items but only 3 model-family/size combinations at the causal-claim level and no item that varies the
precision of an already-supported claim (V62 prepared, never run). Items that would change label counts most:
device/dtype arms, a non-Qwen/non-Pythia family, and a corpus that is neither Gutenberg nor templates.

## Common rules for every experiment

* Same procedure as the source protocol, frozen thresholds unchanged (doc 201 supported rule; gate v3; 0.5-logit floor; Holm at the draw floor). Matched-random draws: 100 where the arm is compared to V61-local (floor 0.0099), else 1000 (AGENTS rule 6).
* Labels computed with `LABEL_RULES.md` v1 unchanged. A new arm gets an L2 rule only if its own protocol is committed first; otherwise L3/L4 apply as written. New items are appended as bench v1.1; the v1 holdout list in `splits/split.json` is never edited, new ids are assigned by the same hash rule.
* Raw traces stay outside the repo in an owner-only root and are deleted within 72 h; only digests and aggregates are committed.
* Abort (no retuning, report as NotCompleted/closed): weight or corpus digest mismatch; selection guard failure; gate v3 known-bad not rejected; spend cap reached. A stopped run is recorded as `not_tested`, not as a failure.
* Negative results are reported with the same prominence.

## E1: V62 device/dtype arms (protocol 202)

* Design: exactly protocol doc 202: arms A `mps_f32`, B `cpu_f32`, C `mps_bf16`, D `mps_fp16` on Pythia-2.8b, V54 corpus, 100 controls, decision rule "Robust / Effect robust, selection precision-sensitive / Not robust", 8 h per-arm cap. Protocol 202 specifies Apple MPS/CPU on the owner's Mac.
* Cost as written: **$0, zero GPU hours**. Time: arm A is a determinism replicate of V61-local, *measured* 6,806 s (`out_v61_local.json`, `wall_seconds`); B (CPU) is likely slower (*estimate* up to the 8 h cap), C/D similar to A. Order of 10-26 wall-clock hours on the Mac.
* Needs: Pythia-2.8b weights (5.6 GB) and the V54 Gutenberg corpus rebuild (about 300 MB), both already authorised in doc 202 text but not executed; confirm that authorisation still stands.
* Yield: up to 4 arms x 4 items (T0, L, P, swap-set stability, same rule S-D201/S-D202J) = up to 16, strongly clustered; B-D give 12 device/dtype items.
* Optional E1b (not in doc 202, needs its own protocol): the same pipeline on H100 CUDA at fp32, bf16, fp16. *Estimate* 0.5 h per arm with 100 controls (R6's measured 1000-draw H100 run took about 2.5 h, with L and P at about 3,300 s each; control cost is roughly linear in draws) plus 0.5 h setup/download: **2.0 H100-h, $10.1**.

## E2: second model family

* Design: the R6 pipeline (`experiments/r6`, `docs/protocols/R6-second-model-agreement-gate.md`) unchanged, on one open-weight family that is neither Pythia (GPT-NeoX) nor Qwen3, about 1-2B parameters (candidates: OLMo-2-1B, Llama-3.2-1B; pick one in the committed pre-run packet, do not switch after seeing outcomes). 1000 controls, CUDA fp32, TF32 off.
* Cost: *estimate* 2.5 H100-h from R6's measured profile (about 2.5 h for 1.7B, Addendum 9 observation) plus 0.5 h download/setup: **3.0 H100-h, $15.1**. Model weights about 2.5-6 GB need owner download approval.
* Yield: 3 items (T0, L, P) against V61-local base, axes `model_family, scale, device`.

## E3: non-templated corpus

* Design: V61-local pipeline on Pythia-2.8b with a fresh natural-text corpus that is not Project Gutenberg and not template-generated (for example class-balanced Wikipedia/wikitext sentences, as V49 b1/b2 used for other models), item-disjoint from V54, label audit 200 items (<=5% error else close unrun), 100 controls. A second arm on Qwen3-1.7B-Base with 1000 controls if E2 is skipped.
* Cost: *estimate* Pythia arm 0.6 H100-h + 0.5 h setup (**1.1 h, $5.5**); Qwen3-1.7B 1000-control arm 3.0 h (**$15.1**).
* Yield: 3 items per arm (T0, L, P), axis `corpus`.

## Budget table

| experiment | H100-h | USD at $5.04/h | basis |
|---|---|---|---|
| E1 as written (Mac) | 0 | 0 | measured Mac wall time |
| E1b CUDA arms | 2.0 | 10.1 | estimate |
| E2 second family | 3.0 | 15.1 | estimate from measured R6 profile |
| E3 Pythia arm | 1.1 | 5.5 | estimate |
| E3 Qwen arm (optional) | 3.0 | 15.1 | estimate |
| **Total with all of the above** | **9.1** | **45.9** | |
| Contingency +50% | 13.7 | 68.9 | AGENTS-style margin |

Prior spend context: the campaign ceilings recorded in `docs/INDEX.md` and `docs/AUTHORIZATION.md` are spent or expired; every figure above needs a new owner-stated ceiling. Minimum useful grant: E1 (free) first, then about USD 25 for E2 plus the E3 Pythia arm.

## Order and stopping

E1 first (free, answers the V62 question, adds most items). Then E2, then E3. After each, run `scripts/verify.py` and append items; if an experiment closes before measurement, record it as one `not_tested` item and do not retry under the same identity.
