# R4d result (run once, protocol unchanged)

Source `results/r4d_powered_birth_order.json`, runner `experiments/r4d_powered_birth_order.py`. 500 neurons per layer, otherwise identical to R4c.

**Frozen verdict: `directed_growth_supported`** (growth-pass in 3 of 3 layers; rule required 2).

| Layer | qualifying neurons | mean rho | null mean (sd) | p | output-embedding rho (descriptive) |
|---|---|---|---|---|---|
| 3 | 348 | -0.058 | -0.010 (0.010) | 0.001 | -0.010 |
| 6 | 274 | -0.113 | -0.018 (0.013) | 0.001 | -0.079 |
| 9 | 267 | -0.056 | -0.010 (0.013) | 0.001 | -0.059 |

p = 0.001 is the 1000-permutation floor. Excess over null: about 4.6, 7.5 and 3.4 null standard deviations.

## What this shows
In Pythia-160m, among tokens that persist in a neuron's top set to the final checkpoint, tokens that joined closer in time are more embedding-similar, beyond what token-frequency structure alone gives (frequency-stratified null). That is the pattern directed, chained growth predicts and stable coherence does not. R4b's size-biased test and R4c's underpowered test had left this open; with ~270-350 neurons per layer the effect is clear.

## Why this is modest, and not a confirmation of the essay
- **Small effect.** Mean rho is -0.06 to -0.11 (layers 3 and 9 clear the -0.05 rule by only 0.006-0.008). Statistically unambiguous, practically weak.
- **Post-hoc power increase.** R4d was designed after R4c's near-miss. Thresholds were not changed, but the same model, text and statistic were reused, so this is not an independent replication. A fresh text split and a second model (Pythia-410m or 1B) should confirm before relying on it.
- **Output-embedding corroboration is mixed.** Layer 3's secondary measure is ~0 (-0.010), so the layer-3 result rests on input-embedding similarity only.
- **Definition-dependent.** Birth = first appearance in a top-15 set; relatedness = token embedding cosine; neurons chosen at the final checkpoint (survivorship). It tests temporal clustering of similar tokens, not the essay's concept-level "iy -> diy -> do it yourself" chains (R4 showed the literal substring track is artifact-contaminated).
- **Promised descriptive item not produced.** The protocol said to report the first-100-neuron rho for comparison with R4c; the runner did not emit it. Not reconstructed after the fact.
- Ceiling `LocalDevelopmentUnreviewed`; spend USD 0.

## Status of the essay's offshoot claim here
Token-level temporal clustering supported (weakly, one model, post-hoc power). Not supported or tested: concept-level chains, other models, image models, the causal role of offshoots.
