# R1 result (run once, protocol unchanged)

Source `results/r1_qwen3_neuron_audit.json`, runner `experiments/r1_qwen3_neuron_audit.py`. Model Qwen3-1.7B-Base (cached; the requested Qwen3-0.6B was not run, see protocol). 100 random neurons x 2 poles; 198 of 200 poles passed the z >= 4 activity filter. No new downloads, spend USD 0.

**Frozen verdict: `validated = false`, `essay_consistent = false`.**

| Cut (theta percentile) | one | two | many | token-unexplained |
|---|---|---|---|---|
| P90 | 0.030 | 0.035 | 0.253 | 0.682 |
| P95 (primary) | 0.030 | 0.030 | 0.258 | 0.682 |
| P98 | 0.030 | 0.015 | 0.273 | 0.682 |

Instrument checks: test-retest agreement 0.87 / 0.89 / 0.91 (>= 0.80 passed). Planted-composite sensitivity **could not be measured**: only 6 poles were classified `one`, giving 0 distinct-job pairs (30 needed). So the audit failed validation by the protocol's own rule (sensitivity uninformative), independent of the percentages.

Per layer at P95, `token_unexplained`: layer 4 0.28, layers 10/16/22/27 0.74-0.80. Median token-lookup correlation r_tok across poles: 0.247 (63 of 198 poles >= 0.3).

## What this shows
- A token-identity audit explains about a third of Qwen3-1.7B-Base neuron poles (r_tok >= 0.3), mostly in layer 4; in layers 10-27 about 75-80% of poles are not predictable from the current token alone. That is consistent with mid/late MLP neurons being context-dependent. It is **not** evidence that those neurons are unexplained in the essay's sense, because the essay's jobs are context/concept descriptions that this proxy cannot see.
- Among token-explained poles, very few resolve to a single embedding cluster (3% of all poles), and about a quarter split into three or more clusters at these cuts. The clustering step is crude: Qwen input-embedding cosines are small (95th percentile 0.065), so cluster counts are very sensitive to the cut, and I would not read "many" as polysemanticity.
- Therefore R1 neither confirms nor contradicts the essay's "~90% one main job" figure for Qwen3-32B. The proxy is too weak and failed its own calibration. This is the same lesson as R5/R5b: threshold-style job audits need ground-truth calibration, and here the calibration had no population to run on.

## Open / next
R1b would need a context-aware labeller (a local LLM simulator scored on held-out activations, with an activation-prediction metric) and enough one-job poles to calibrate on planted composites. That needs a design and, for a larger model or a stronger labeller, a download/spend decision (the 10 GB download budget is exhausted; Qwen3-0.6B was not run). Ceiling `LocalDevelopmentUnreviewed`; not independently reviewed.
