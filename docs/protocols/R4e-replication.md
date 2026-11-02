# R4e protocol: replicate R4d on fresh text and a second model

Frozen before the R4e run. Slice `mech-interp-unified-r4e`. Ceiling `LocalDevelopmentUnreviewed`, spend 0.

## Why
R4d's support for directed growth was weak and came from a post-hoc power increase on reused data. R4e reruns the R4d procedure unchanged on data and a model it has not seen.

## Arms (statistic, null and thresholds are exactly R4d's/R4c's; only the items below change)
- **Arm A (fresh text, same model):** Pythia-160m, the same 13 checkpoints, layers 3/6/9, 500 neurons/layer. Text: WikiText-2 train lines (>= 200 chars) with indices 800-1199 as fit and 1200-1599 as assess (R4/R4b/R4c/R4d used indices 0-799; disjoint). No new downloads.
- **Arm B (second model size, fresh text):** Pythia-70m (6 layers, MLP width 2048), layers 1/3/4 (nearest to 25/50/75% depth), 500 neurons/layer of 2048, same fresh text as arm A. Checkpoints: step2000, 4000, 8000, 16000, 24000, 32000, 48000, 64000, 80000, 100000, 120000, 143000 (12; step1000 omitted to stay inside the 10 GB download budget; pytorch_model.bin fp16 files, about 0.17 GB each, cast to fp32 for computation).
Embedding relatedness uses the arm's own final-checkpoint input embeddings; frequency terciles use the arm's fit text.

## Decision rules (unchanged from R4c/R4d)
Layer growth-pass: mean rho <= -0.05, p <= 0.01 (1000 frequency-stratified permutations), >= 30 qualifying neurons. Arm passes iff growth-pass in >= 2 of 3 layers.
- **Replicated** iff both arms pass.
- **Partial** iff exactly one arm passes (reported with which).
- **Not replicated** iff neither passes. Layers with < 30 qualifying neurons count as failing.
No threshold changes after data. Limits: same family and tokenizer; token-level; no concept-level test; Arm B's different checkpoint count changes the birth-index resolution.
