# R4c protocol: does temporal proximity of token birth predict token similarity? (fresh identity)

Frozen before R4c code runs. Slice `mech-interp-unified-r4c`. Ceiling `LocalDevelopmentUnreviewed`. Local compute, spend 0. Downloads authorized in `docs/AUTHORIZATION.md` (8 additional Pythia-160m revisions; total repo download budget 10 GB, disk checked before and after).

## What R4/R4b left open
R4 showed token sets are semantically coherent; R4b could not separate directed growth from a stable concept. R4c uses a test where stable coherence predicts a flat relationship by construction and size is controlled by using the same tokens throughout.

## Materials
- Model EleutherAI/pythia-160m, 13 revisions: step1000, 2000, 4000, 8000, 16000, 24000, 32000, 48000, 64000, 80000, 100000, 120000, 143000 (final). Layers 3, 6, 9; MLP `act` outputs.
- Text, fit/assess split, neuron selection (top 100 per layer by max z-score on fit at the final checkpoint), token-set definition T(n,c) (>=5 occurrences, up to 15 types, type-mean >= 0.5 x neuron max type-mean): identical to R4.
- Relatedness measure: cosine of final-checkpoint input-embedding rows (primary); cosine of final output-embedding (`embed_out`) rows (secondary, descriptive).

## Statistic
For neuron n, persistent tokens = T(n, final). Birth index b(t) = index (0..12) of the first checkpoint whose T(n,c) contains t. For all pairs (t,u) of persistent tokens with b(t) != b(u): similarity cos(t,u) vs temporal distance |b(t)-b(u)|. rho_n = Spearman correlation across those pairs. A neuron qualifies iff it has >= 5 persistent tokens, >= 3 distinct birth indices, and a defined rho. Layer statistic = mean rho_n over qualifying neurons; require >= 30 qualifying neurons.

Directed growth predicts rho_n < 0 (tokens born close in time are more similar, chained offshoots). A stable concept predicts rho ~ 0 (similarity independent of when a token joined).

## Null
Within each neuron, permute birth indices among its persistent tokens, stratified by frequency bin (tercile of fit-text token count; edges = 1/3 and 2/3 quantiles of fit counts over types with count >= 1; unseen types go to the lowest bin). This keeps the token set, the similarity matrix and the frequency-birth correlation fixed and destroys only the pairing of birth time to token identity within frequency strata. 1000 permutations; layer mean per permutation; p = (1 + #null_mean <= obs_mean)/1001.

## Decision per layer and overall
- Layer **growth-pass** iff mean rho <= -0.05 and p <= 0.01 and >= 30 qualifying neurons.
- **Directed growth supported** iff growth-pass in >= 2 of 3 layers.
- **Stable coherence supported** iff in >= 2 layers |mean rho| < 0.03 and p > 0.05 with >= 30 qualifying neurons (an informative null).
- Otherwise **inconclusive**. Limits: small model, one corpus, token-level relatedness, birth defined by top-set membership, 13 coarse-to-fine checkpoints. No threshold changes after data.
