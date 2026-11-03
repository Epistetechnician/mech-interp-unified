# R1 protocol: neuron-pole "jobs" audit on Qwen3-1.7B-Base (calibrated, token-level proxy)

Frozen before the R1 code runs. Slice `mech-interp-unified-r1`. Ceiling `LocalDevelopmentUnreviewed`, spend 0, **no new downloads**.

## Deviations from the request, stated up front
- Model: the owner asked for Qwen3-0.6B. That model is not cached, and its download (~1.2 GB) would exceed the 10 GB download budget in `docs/AUTHORIZATION.md` (about 10 GB already used by R4-R4e). **Qwen3-1.7B-Base** is already in the local HF cache, so R1 runs on it with zero new download. Qwen3-0.6B stays unrun.
- The essay's audit uses LLM/human interpretation of "jobs". No labelling provider or spend is authorized, so R1 uses an **operational token-level proxy**, calibrated on planted composites. It lower-bounds "explained"; it cannot reproduce the essay's 90/7/2-4 split, only measure what a token-level audit sees.

## Materials
Qwen/Qwen3-1.7B-Base (28 layers, MLP width 6144, tied embeddings), bf16 weights, activations cast to fp32. Neuron activation = input of `mlp.down_proj` (the SwiGLU product). WikiText-2 train lines (>= 200 chars): fit = indices 1600-2199, assess = 2200-2799 (disjoint from R4-R4e), truncated to 128 tokens.
Neurons: 20 uniformly random neurons (seed 0) in each of layers 4, 10, 16, 22, 27 = 100 neurons, both poles.

## Pole validity
Signed activation a over fit positions, z = (a - mean)/std. Positive pole valid iff max z >= 4; negative pole valid iff min z <= -4. Pole signal x = max(sign * a, 0). Invalid poles are counted as inactive and excluded.

## Pole classification (jobs derived on one text half, token predictor scored on the other)
1. Top positions: x in the top 1% of the deriving half (at least 200 positions).
2. Token types covering 80% of the summed x among top positions (cap 30 types).
3. Cluster those types: average-linkage on cosine distance of `embed_tokens` rows, cut at distance 1 - theta. theta = P-th percentile of pairwise cosine among the 2000 most frequent fit tokens; primary P = 95, sensitivity P = 90 and 98.
4. r_tok: token-lookup simulator (mean x per token type on the deriving half, count >= 3 else global mean) vs actual x on the other half, Pearson over all positions.
5. Class: r_tok < 0.3 -> `token_unexplained`; else clusters = 1 -> `one`, 2 -> `two`, >= 3 -> `many`. ("Unexplained at token level" includes context-dependent neurons the essay may call explained.)
Main analysis: derive on fit, score on assess.

## Instrument validity (calibration; required before any percentage is read as an estimate)
- **Sensitivity (planted composites):** pairs of `one` poles from the same layer whose token sets are genuinely distinct (max cross-set cosine < theta). Composite x_c = max(x_a, x_b) elementwise on both halves; run steps 1-5. Sensitivity = fraction classified `two` or `many`. Up to 100 pairs; fewer than 30 pairs = uninformative.
- **Test-retest:** classify every valid pole with halves swapped (derive on assess, score on fit); agreement = fraction with the same class.
- **Validated** iff sensitivity >= 0.80 and test-retest agreement >= 0.80.

## Reported and decision
Counts of valid poles; class fractions overall and per layer at P = 90/95/98; sensitivity; test-retest. The essay-comparison flag `essay_consistent` is true iff validated AND the primary `one` fraction is in [0.80, 0.97] AND `token_unexplained` <= 0.08. If not validated, the fractions are reported as unvalidated proxy values only. Limits: one small base model, one corpus, token-level proxy, embedding clustering, 100 sampled neurons. No threshold changes after data. The essay's 32B figures are neither confirmed nor refuted by this run.
