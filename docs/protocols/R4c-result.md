# R4c result (run once, protocol unchanged)

Source `results/r4c_birth_order_similarity.json` (13 Pythia-160m checkpoints, hashes recorded), runner `experiments/r4c_birth_order_similarity.py`. One earlier launch crashed on an output-embedding accessor before any data was read; the fix is commit `38255a7` and changed no protocol element.

**Frozen verdict: inconclusive.** Neither the growth rule nor the stable-coherence rule was met.

| Layer | qualifying neurons | mean rho | null mean (sd) | p | growth-pass | stable-pass |
|---|---|---|---|---|---|---|
| 3 | 42 | -0.112 | -0.042 (0.031) | 0.014 | no (p > 0.01) | no |
| 6 | 22 | n/a | n/a | n/a | no (< 30 neurons) | no |
| 9 | 32 | -0.070 | -0.024 (0.041) | 0.142 | no | no |

Secondary (output-embedding similarity, descriptive): mean rho -0.069 (layer 3), -0.045 (layer 9).

## Reading
- The sign matches directed growth in both testable layers (tokens born closer in time are somewhat more similar), but the effect is weak, and layer 3 misses the frozen p <= 0.01 bar (0.014); layer 9 is far from it (0.14). Layer 6 lacks enough qualifying neurons (22 < 30) to be tested at all.
- The frequency-stratified null itself sits below zero (-0.04, -0.02), so part of the negative correlation comes from token frequency/birth structure that the null preserves. Excess over null is about -0.07 (layer 3) and -0.05 (layer 9), roughly 2 and 1 null standard deviations.
- Neither "directed growth" nor "stable coherence" is established. Combined with R4 (adjacent) and R4b (not separable), the offshoot claim in Pythia-160m remains **unresolved, with a weak hint toward growth in the earliest layer**.
- Why power is low: only 22-42 of 100 selected neurons per layer have >= 5 persistent tokens with >= 3 distinct birth checkpoints; token sets are small; the birth definition is crude.

Next, if pursued: larger pool of neurons (select 500 per layer), all 154 Pythia checkpoints densely early in training, and a relatedness measure based on activating contexts rather than token embeddings. Not preregistered yet.
