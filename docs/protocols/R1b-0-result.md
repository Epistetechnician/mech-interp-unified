# R1b-0 result: harness, tests, load and timing probe

Slice `mech-interp-unified-r1b`, ceiling `LocalDevelopmentUnreviewed`. No real poles were audited; no percentages exist. Zero spend, no model download (isolated `mlx_lm` 0.31.3 venv at `~/.local/share/mechinterp/mlx-venv`).

## Built
`experiments/r1b/lib.py` (evidence, job parsing, scoring, classification, C1-C4 metrics, rule poles, composites, derangement), `fakes.py`, `mlx_backend.py` (labeller and logprob simulator), `probe.py`. `tests/r1b/test_lib.py`: 17 hermetic tests pass (fake labeller passes C1/C2; a random labeller fails C1; assess data never reaches the labeller; composites and verdict logic). One test caught a real bug (a non-list `jobs` was split into characters); fixed.

## Probe (`results/r1b0_probe.json`, synthetic text)
- Qwen3.6-35B-A3B-MLX-4bit loads: 248 s.
- Explainer call: 10 s, 3 jobs parsed from valid JSON.
- Simulator: 0.39 s per 64-token context per job (~454 prompt tokens, ~1160 tok/s). Sanity: planted token scored 3.4 vs 1.4 for others, so it discriminates but is soft; whether it is good enough is exactly what C1 decides.

## Cost and a design amendment (made before freezing)
Design section 5 assumed 400 assess contexts per job. That costs ~470 s per pole, ~52 h for 400 poles: infeasible. Amendment proposed for the freeze: score each pole on 60 assess contexts (30 highest-activation by the pole's own assess-pool activations, 30 random), the same set for every job and control. Per pole ~80 s including the explainer; 200 poles (100 neurons, both poles) ~4.5 h; C1 (60 poles) ~1.3 h. Stratifying by activation inflates r relative to a random sample, so thresholds are calibrated on the same sampling by C1. Not yet frozen.

## Next
R1b-1: run C1 and C2 only, with a threshold of freeze committed first. If C1 fails, R1b closes as "labeller too weak".
