# R4b result (run once, protocol unchanged) plus post-hoc addendum

Source `results/r4b_growth_vs_coherence.json`; addendum `results/r4b_addendum_size_matched.json`.

## Frozen verdict: `stable_coherence_favored` (but biased, see below)
| Layer | Test A gap (persistent - transient) / p | Test C mean d / p | Result |
|---|---|---|---|
| 3 | 0.081 / 0.178 (fail) | -0.266 / 0.0001 | stable-coherence pass |
| 6 | 0.038 / 0.021 (fail) | -0.131 / 0.049 | neither |
| 9 | -0.063 / 0.657 (fail) | -0.259 / 0.0001 | stable-coherence pass |

Test A fails in every layer: tokens that persist to the final checkpoint are not more related to the prior set than transient ones. Directed growth (A and C-growth in >=2 layers) is not supported.

## Why the frozen Test C verdict cannot be trusted
Test C compares relatedness to a small frontier set (only the tokens added in the previous transition) against the larger core (up to 15 tokens). "Max cosine >= theta" is larger-set-favoring by construction. This design flaw was identified after the run; the frozen verdict stands as recorded and is not edited.

## Post-hoc size-matched addendum (descriptive, not decision-bearing)
Core subsampled to the frontier's size (200 draws/event): mean d = -0.016 (p=0.63), +0.055 (p=0.32), +0.029 (p=0.39) for layers 3/6/9. No difference from zero anywhere. So the apparent "core is closer" effect was the set-size artifact.

## Honest conclusion on the essay's offshoot claim (Pythia-160m, 5 checkpoints, token level)
- R4 found new tokens are embedding-adjacent to a neuron's earlier tokens (2x null).
- R4b found no evidence that this adjacency is *directed growth* (persistence test fails; frontier-vs-core indistinguishable once size is matched).
- Result: the offshoot claim is **unresolved** at this resolution, not supported and not refuted. Needs finer checkpoint spacing (Pythia has 154 checkpoints) and concept-level, not token-embedding, relatedness.
