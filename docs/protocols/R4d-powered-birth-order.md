# R4d protocol: R4c with 5x more neurons (power increase)

Frozen before the R4d run. Slice `mech-interp-unified-r4d`. Ceiling `LocalDevelopmentUnreviewed`, spend 0, checkpoints already cached (no new downloads).

## Disclosure
R4d is a power increase chosen **after seeing R4c** (inconclusive: layer 3 p=0.014 vs a 0.01 bar; layer 6 had 22 of the 30 neurons required). It is not independent confirmation of R4c's layer-3 hint; the same model, text, checkpoints and statistic are reused.

## Single change from R4c
Neurons selected per layer: **500** (top 500 by max z-score on fit text at the final checkpoint) instead of 100. Everything else is identical to `R4c-birth-order-similarity.md`: 13 checkpoints, layers 3/6/9, text split, token-set definition, input-embedding cosine, birth index, frequency-tercile-stratified permutation null (1000 permutations), and the decision rules (growth-pass: mean rho <= -0.05, p <= 0.01, >= 30 qualifying neurons; directed growth supported iff growth-pass in >= 2 of 3 layers; stable coherence supported iff in >= 2 layers |mean rho| < 0.03 and p > 0.05 with >= 30 neurons; otherwise inconclusive). No threshold, seed rule or statistic is changed.

## Extra reporting (descriptive only)
Qualifying-neuron counts, mean rho for the first 100 selected neurons only (to compare with R4c), and the output-embedding rho.
