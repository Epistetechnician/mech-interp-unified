# R4 protocol: do neurons grow offshoots during training? (Pythia-160m checkpoints)

Frozen before any model or data is loaded. Slice `mech-interp-unified-r4`. Ceiling `LocalDevelopmentUnreviewed`. Local compute, spend 0. Interpretation caveat from R5: neuron "job" definitions via raw correlation are not trusted; R4 uses token-level activation sets, not job labels.

## Claim tested
Essay: neurons grow along adjacent inputs ("iy" -> "diy" -> "do it yourself"), rather than attaching unrelated inputs.

## Materials (pinned at download, hashes recorded in results)
- Model EleutherAI/pythia-160m, revisions step16000, step32000, step64000, step100000, step143000 (final). Layers 3, 6, 9; MLP activation = output of `mlp.act`.
- Text: Salesforce/wikitext `wikitext-2-raw-v1` train split, non-empty lines >= 200 chars. Fit = first 400 such lines, assess = next 400 (disjoint). Truncate each to 128 tokens.

## Procedure
1. Neuron selection (final checkpoint, fit text only): per layer the 100 neurons with the highest max z-score ((a-mean)/std over all fit tokens).
2. Token set T(n, c) on assess text, per neuron n and checkpoint c: token ids by input token at each position; keep types with >= 5 occurrences; mean activation per type; T = up to 15 types with mean activation >= 0.5 x that neuron's maximum type-mean.
3. Transitions: consecutive checkpoints among the five. For each neuron and transition with T_prev nonempty, N = T_next \ T_prev nonempty.
4. A new token t is *related* to T_prev iff (a) substring: decoded lower-case stripped strings, length >= 2, one contains the other for some s in T_prev; or (b) embedding: max cosine of final-checkpoint input-embedding rows to T_prev >= theta, theta = 95th percentile of pairwise cosine among the 2000 most frequent token types in fit.
5. Statistic R = mean over (neuron, transition, new token) events of related. Report R_sub (substring only) and R_emb (embedding only) separately.
6. Null: replace T_prev by T_prev of a different random neuron in the same layer and transition (1000 permutations, stratified by layer). p = (1+#null>=obs)/(1001).

## Decision (preregistered)
Offshoot growth **supported** iff, in at least 2 of 3 layers, R_obs - R_null_mean >= 0.10 and p <= 0.01, and total events >= 200 per layer. Fewer than 200 events in a layer: that layer is "uninformative". Otherwise "not supported". Limits stated up front: small model, one corpus, token-level not concept-level, embedding similarity is a weak proxy for relatedness, five checkpoints are coarse, substring track tests only the essay's literal example type. No threshold changes after data is seen.
