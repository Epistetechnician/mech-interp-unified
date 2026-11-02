# R4e result (run once, protocol unchanged)

Sources `results/r4e_armA_pythia160m_fresh_text.json`, `results/r4e_armB_pythia70m_fresh_text.json`, `results/r4e_verdict.json`; runner `experiments/r4e_replication.py`.

**Frozen verdict: `replicated`** (both arms pass the R4d rule in 3 of 3 layers).

| Arm | Layer | qualifying neurons | mean rho | null mean (sd) | p | output-embedding rho |
|---|---|---|---|---|---|---|
| A: Pythia-160m, fresh text | 3 | 374 | -0.070 | -0.010 (0.010) | 0.001 | -0.047 |
| | 6 | 255 | -0.099 | -0.013 (0.014) | 0.001 | -0.094 |
| | 9 | 294 | -0.055 | +0.001 (0.013) | 0.001 | -0.031 |
| B: Pythia-70m, fresh text | 1 | 323 | -0.053 | -0.004 (0.012) | 0.001 | -0.015 |
| | 3 | 335 | -0.059 | -0.011 (0.011) | 0.001 | -0.055 |
| | 4 | 389 | -0.091 | -0.011 (0.010) | 0.001 | -0.083 |

p = 0.001 is the 1000-permutation floor.

## What it shows
The R4d pattern appears again on text the analysis had not seen (arm A) and in a smaller model (arm B): among persistent tokens of a neuron, tokens that joined its top set closer in time are more embedding-similar than a frequency-stratified null predicts. Effect sizes are consistent across 2 models x 6 layers: rho about -0.05 to -0.10, 4-8 null standard deviations. Output-embedding similarity (secondary, descriptive) points the same way in all six layers, though weakly in the earliest layers (-0.015, -0.031, -0.047).

## What it does not show
- Marginal margins: arm B layer 1 (-0.053) and arm A layer 9 (-0.055) clear the -0.05 bar by 0.003-0.005. The decision rule was met, but the effect is small everywhere.
- Same model family (Pythia) and tokenizer, same corpus type (WikiText-2), different slice only. No other family, no vision model.
- Token-level temporal clustering only. This does not test the essay's concept-level "iy -> diy -> do it yourself" chains, nor that offshoots have any causal or functional role, nor that neurons are "clean".
- Arm B used a different checkpoint grid (12, step1000 omitted) and fp16-sourced weights, both frozen in advance.
- Spend USD 0; downloads this arm about 2 GB, within the budget. Ceiling `LocalDevelopmentUnreviewed`; not independently reviewed.

## Status of the essay's offshoot claim
Replicated at token level in two Pythia sizes on fresh text, with small effect sizes. Concept-level and cross-family tests remain open.
