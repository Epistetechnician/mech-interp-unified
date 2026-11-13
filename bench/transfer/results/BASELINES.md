# Baseline results (frozen split)

Items 36 (train 26, holdout 10); majority label on train: `replicated`.
Accuracy with Wilson 95% interval; macro-F1 over labels present in truth or prediction. Holdout n is small: read the intervals.

| method | train acc | holdout acc | all acc | holdout tested-only (replicated vs failed) |
|---|---|---|---|---|
| majority | 12/26 = 0.462 [0.29, 0.65], macro-F1 0.211 | 5/10 = 0.500 [0.24, 0.76], macro-F1 0.222 | 17/36 = 0.472 [0.32, 0.63], macro-F1 0.214 | 5/7 = 0.714 [0.36, 0.92], macro-F1 0.417 |
| meta_a | 19/26 = 0.731 [0.54, 0.86], macro-F1 0.526 | 6/10 = 0.600 [0.31, 0.83], macro-F1 0.405 | 25/36 = 0.694 [0.53, 0.82], macro-F1 0.500 | 5/7 = 0.714 [0.36, 0.92], macro-F1 0.417 |
| meta_b | 16/26 = 0.615 [0.43, 0.78], macro-F1 0.481 | 5/10 = 0.500 [0.24, 0.76], macro-F1 0.372 | 21/36 = 0.583 [0.42, 0.73], macro-F1 0.459 | 4/7 = 0.571 [0.25, 0.84], macro-F1 0.364 |

Majority, leave-one-cluster-out over all items (cluster-grouped sensitivity): 17/36 = 0.472 [0.32, 0.63], macro-F1 0.214

Confusion matrices (rows truth, columns prediction) are in `baselines.json`.

LLM predictor: not run (needs a paid API call; see README).
