# Reproduction and improvement plan: "Neural networks are already clean"

Source claims are author-reported, threshold-dependent, informal. This plan turns
each into a preregistered, falsifiable test, then adds what the essay lacks:
causal validation, null-calibrated instrument checks, threshold-sensitivity
curves, and matched controls. Nothing runs until its protocol is committed and
its spend gate is satisfied (`AGENTS.md`).

## Claims and tests

| # | Essay claim | Test | Compute | Improvement over essay |
|---|---|---|---|---|
| R1 | ~90% of Qwen3-32B neuron poles have one main job, ~7% two, 2-4% unknown | Audit of sampled poles: top-activating examples -> labeller proposes jobs -> held-out simulation score decides "explained". Ladder: Qwen3-0.6B, 1.7B, 8B local/cheap, 32B on rented GPU. | Local to H100 | Report the full curve over thresholds, not one point; add causal check: ablating a pole's labelled job vs matched-random poles (reuse V53-V59 machinery); inter-labeller agreement with a second model and a human-audited subsample. |
| R2 | Last-layer interpretable neurons 5% -> 50% (better tools) -> 99% (ConvNeXt) | Fix a written rubric and sampling scheme; score torchvision GoogLeNet/InceptionV1 with old vs transformation-robust feature visualization; score ConvNeXt-T last layer. | Small GPU | Blind scoring; pre-fixed rubric; confidence intervals. |
| R3 | Squeezing a layer (480 -> 10) yields coarser neurons, not superposition; SAE fails to recover clean detectors | Scaled-down replication on a small vision net and a small LM MLP width; train narrow vs wide; fit SAE; compare feature recovery to ground truth planted features. | Small GPU | Planted-ground-truth superposition control, where the answer is known. |
| R4 | Neurons grow offshoots (recursive generalization): "iy" -> "diy" -> "do it yourself" | Public Pythia intermediate checkpoints (160M-1.4B): track fixed neurons' activation neighbourhoods across checkpoints; test local growth vs unrelated attachment against shuffled-neighbourhood nulls. | Local/cheap | Quantified locality statistic with null; this was drafted as Lane D but never run. |
| R5 | No real-world superposition circuit found | Search for counter-example: planted superposition (known) vs real-model candidates under the same pipeline; positive control must be detected. | Local | Establishes the pipeline can see superposition at all before concluding absence. |
| R6 | Case studies (cat/car V-shape, "pr" sound neuron, Colorado comma neuron) | Re-derive with preregistered descriptions and near-miss tests where models are available (Pythia/Qwen language cases). Vision cases only if InceptionV1 weights are in scope. | Local | Positive-control and near-miss test per explanation. |

## Prior in-house evidence feeding this plan

- Neuron-level causal effect exists but is concentrated (top 4 units = 75% of joint effect), not agreement-selective in single-unit terms, and not shown on 1.4B (V55) or as a circuit (V56). This already cuts against a naive "each neuron has one clean job" reading and must be tested directly in R1.
- Unit selection changed with device/precision (V60); V62 (unrun) addresses it and should run first when budget exists.
- theseus-lab: neuron identity is only meaningful relative to a representation (permutation test). R1 must use function-preserving permutation checks.

## Phases

0. **Done:** consolidation, hashes, synthesis (`INDEX.md`).
1. **No-spend:** unify tooling (shared gate/control/lock library from the V50-V61 runners), port hermetic tests, re-check Lean proofs, write R1-R6 protocols as committed documents with all thresholds and aborts.
2. **Local/free-weights:** R4 and R5 on small public models, R1 at Qwen3-0.6B/1.7B. Needs model downloads (owner approval) but no spend.
3. **Paid:** R1 at 8B/32B, R2, R3. Needs a USD ceiling, provider/node identity, allocation receipt.
4. **Synthesis:** one report: what reproduced, what failed, effect sizes with intervals, all negatives. No claim above ceiling.

## Gates required from the owner before phases 2-3

- Permission to download named model weights and datasets (with hashes pinned).
- A hard USD ceiling for any rented compute.
- (Optional) an independent reviewer if any claim above `OwnerSelfAttestedLocalDevelopment` is wanted.

## Progress
- R5: run, **negative**. R5b: run, **audit not validated** (misses diffuse superposed neurons; share cutoff swings results; n=d control not clean). See `protocols/R5b-result.md`.
- R4: rule met but confounded. R4b: persistence test fails; frozen frontier-vs-core verdict was size-biased; size-matched addendum null. Offshoot claim **unresolved**. See `protocols/R4b-result.md`. Next: finer Pythia checkpoints + concept-level relatedness (R4c).
- R1, R2, R3, R6: not started. R1 must not use the R5 job definition.
- R4c: run, **inconclusive** (weak growth-direction hint at layer 3, p=0.014 vs 0.01 bar; layer 6 underpowered). See `protocols/R4c-result.md`.
- R4d: run (500 neurons/layer, post-hoc power increase), **directed growth supported** in 3/3 layers but weak (rho -0.06 to -0.11). Needs fresh-text and second-model replication (R4e). See `protocols/R4d-result.md`.
- R4e: run, **replicated** (Pythia-160m fresh text and Pythia-70m, 6/6 layers pass, effects rho -0.05 to -0.10). Token-level temporal clustering of token births is a small but replicated effect in Pythia; concept-level and cross-family tests open. See `protocols/R4e-result.md`.
- R1: run on cached Qwen3-1.7B-Base (0.6B skipped: download budget). **Not validated, not essay-consistent**: 68% token-unexplained, 3% one-job, calibration uninformative (only 6 one-job poles). Neither confirms nor refutes the essay; needs a context-aware labeller (R1b). See `protocols/R1-result.md`.
- R1b: **closed, instrument too weak.** Oracle descriptions only reach mean r 0.49 (bar 0.8); C1 failed. Negative on the instrument, silent on the essay. See `protocols/R1b-result.md`.
- R6 second-model agreement gate (`protocols/R6-second-model-agreement-gate.md`, distinct from the plan-table R6 case-studies row): run on Qwen3-1.7B-Base, **Replicates** (gate InstrumentValid; all-position swap, last-position swap and donor patch each p=0.000999 vs 1000 matched draws; selection Jaccard 1.0). Single run, gate on A10G and later stages on H100, owner-unreviewed, ceiling `LocalDevelopmentUnreviewed`. See `protocols/R6-result.md`.
