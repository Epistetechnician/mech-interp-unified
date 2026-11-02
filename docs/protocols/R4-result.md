# R4 result (run once, protocol unchanged)

Source: `results/r4_offshoot_trajectory.json`, runner `experiments/r4_offshoot_trajectory.py`. Model/data file hashes are in the result file (5 Pythia-160m checkpoints + WikiText-2 train).

Frozen decision rule: **offshoot growth supported** (all 3 layers pass: gap >= 0.10, p <= 0.01, events >= 200).

| Layer | events | R_any | null mean | gap | R_sub | R_sub null | R_emb | R_emb null |
|---|---|---|---|---|---|---|---|---|
| 3 | 883 | 0.537 | 0.241 | 0.296 | 0.052 | 0.041 | 0.530 | 0.219 |
| 6 | 733 | 0.462 | 0.209 | 0.253 | 0.035 | 0.036 | 0.450 | 0.191 |
| 9 | 771 | 0.453 | 0.224 | 0.229 | 0.064 | 0.044 | 0.437 | 0.200 |

p = 0.001 (the 1000-permutation floor) in every layer.

## What this does and does not show
- Tokens that newly enter a neuron's top set are about twice as embedding-similar to that neuron's earlier set as to another neuron's earlier set. That is the "adjacent, not arbitrary" half of the essay's claim, at token level in one small model.
- The literal essay-type offshoot (substring: "iy" -> "diy") is **not** supported: R_sub is at the null in layer 6 and barely above it elsewhere. The layer-6 substring examples are mostly artifacts (case/whitespace variants of the same token such as "is"/"is", very short tokens like "re", "it"). The substring track in this protocol is contaminated and should not be cited as support.
- **Main confound, not controlled by the frozen null:** a neuron whose meaning is stable from early in training will have coherent token sets at every checkpoint, so new tokens would look "adjacent" even if no growth occurs. The neuron-shuffle null shows only that neurons are semantically coherent, not that sets *grow outward* over time. Distinguishing growth from stable coherence needs a churn control (new tokens vs. tokens that stayed/dropped, and vs. early-checkpoint-only sets). That is R4b, to be preregistered, and until it exists this result is "coherent token neighbourhoods", not "recursive generalization".
- Limits (as frozen): 160M model, one corpus, 5 coarse checkpoints, token-level, embedding cosine as a proxy, theta=0.094 from fit-text frequent tokens.

Ceiling `LocalDevelopmentUnreviewed`; spend USD 0.
