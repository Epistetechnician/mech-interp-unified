# Research index (synthesis of `imports/`)

All results are owner-run, unreviewed, local-development scale. Ceiling for
everything below: `LocalDevelopmentUnreviewed`, or
`OwnerSelfAttestedLocalDevelopment` where an owner attestation exists (V49, V53,
V54). Self-attested is not independent. Numbers are read from the imported
`out_*.json` files; protocols live in `imports/astral-neuron/protocols/`.

## A. Astral neuron-level causal lane (Pythia, subject-verb agreement)

Question: do a few fit-selected MLP units causally carry agreement, beyond
matched-random units (1000 draws, p floor 0.001), under a gate that must reject a
deliberately broken instrument?

| Run | Model | Outcome | Note |
|---|---|---|---|
| V49 | see `runs/v49-run` | owner-attested, scoped | Templated task family (12 templates), first-order gradient selection, three layers; gate v1 only. |
| V50 | Pythia-1.4B | `InstrumentNotValid` | Old gate failed known-bad control (0.028 vs selected 0.189 logits, but gate rule rejected). |
| V51 | 2.8B | budget stops | Protocol ran out of budget; no verdict. |
| V52 | 2.8B | `NoCandidate` | One-item constant control; gate v3 (null-calibrated) introduced. |
| V53 | 2.8B | `Candidate` | Assess effect 0.812 logits, p=0.001; constant-control gap z=39. |
| V54 | 2.8B, fresh books | `Candidate` | Effect 0.709; replication of V53, not blind to design. |
| V55 | 1.4B | `InstrumentNotValid` | Failed frozen 0.5-logit floor. Effect does not transfer to 1.4B under the gate. |
| V56 | 2.8B | H2' false | Path analysis: not a circuit/mechanism. |
| V57 | 2.8B | dose-response not monotone; mean-ablation supported | Mean-ablation effect 0.723 (ctl max 0.015, 1.02x of zero-ablation). |
| V58 | 2.8B | concentration true; agreement-selective false; direct-path false | Top-4 units carry 75% of the k=128 joint effect (0.824). |
| V59 | 2.8B | swap sufficiency | 7 number-selective units swapped: -1.31 logits vs control max 0.07. Post-hoc. |
| V60 | 2.8B | headline hardened | Last-position swap and donor patch survive. Also showed device/numerics can change unit selection. |
| V61-local | 2.8B, MPS fp32 | T0/L/P all supported | Headline reproduced and hardened. |
| V61 (cloud) | 2.8B | minimal-pair donor patch supported | 7 swap units, 800 items: drop 0.482 logits vs control max 0.019 (about 25x); same-number donor -0.033. Smaller than V60's 1.40, so part of that was non-number donor differences. |
| V62 | 2.8B | **prepared, not run** | Device/dtype robustness arms; protocol `202`. (Subsequent upstream lanes V64/V65 continued in `composed-zk-benchmark-os`.) |

Honest read (at import time): a necessary-and-partly-sufficient unit set exists for agreement in
Pythia-2.8B, concentrated in few units, not shown to be a circuit, and selection is device/precision-sensitive (V60).
Post-import runs in this repository (`docs/protocols/`): **R6** replicated the causal agreement unit lane out-of-family on `Qwen3-1.7B-Base` (`docs/protocols/R6-result.md`), and **F2** evaluated rules gates (`0/8`) and informed probes (`3/8`) against 8 planted-flaw organism classes (`docs/protocols/F2-result.md`).
Campaign spend per the V61 record: about USD 45.6 of the USD 50.00 ceiling, so roughly USD 4.4 remains and V62's budget is not guaranteed.

## B. Machine-checked harness logic (Lean 4.30.0)

`imports/astral-neuron/formal/neuron-gate-v1`: gate rule (support needs strict
exceedance of control maxima in both classes plus a floor), p-value floor
1/(1+n), quota/selection optimality, `p_uniform` (no-tie case), and a generated
certificate recomputing V61-local verdicts from logged numbers.
Does not prove the model results, selection stability, or intervention
correctness. `proof-carrying-nano-interp-v1` is a separate formal/tool package.

## C. theseus-lab (mechanism replacement; GPT-2, OPT-125M, SmolLM)

Behaviour and mechanism can come apart; raw labels/positions are not stable
causal identity (MLP permutation changed 100% of positions, preserved behaviour);
exact structural replacement is easy; learned replacement works only for a
narrow affine case (V8: 9/9); whole-block replacement did not replicate
(V9-V21, V39 diagnosis: matches values, not derivatives). Full audit:
`imports/theseus-lab/docs/theseus-all-runs-audit-v1.md`.
Relevance to Goodfire: caution that neuron/feature identity must be
representation-relative.

## D. interpretability-lab-sandbox (SAE/transcoder harness)

GPT-2 layer-6 SAE and Gemma Scope transcoder (V9/V10) adapters with custody and
fail-closed gates. SAE-centric, so it complements rather than tests the
"neurons are already clean" claim. MLP-out qualification closed negative.

## What this corpus does NOT contain

Any image-model work, any Qwen3 neuron audit, any training-trajectory
(offshoot) measurement, any squeezing experiment, any test of superposition.
Those are the reproduction targets (`REPRODUCTION_PLAN.md`).
