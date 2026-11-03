# R1b-1 protocol (frozen before the run)

Slice `mech-interp-unified-r1b`, ceiling `LocalDevelopmentUnreviewed`. Runs only C1 and C2 of `R1b-design.md`; no real neuron poles. Zero spend, no downloads.

- Labeller and simulator: Qwen3.6-35B-A3B-MLX-4bit (temperature 0, no thinking). Prompts, parsing, scoring, classification (delta 0.05, primary threshold 0.5) and bars are exactly those in `experiments/r1b/lib.py` at the commit of this file. No threshold changes after data are seen.
- Text: wikitext-2 train lines 2800-3400 (fit) and 3400-4000 (assess), 64 Qwen3 tokens per context. Disjoint from R1.
- 60 rule poles (seed 100): 20 token-set (6 tokens), 20 "token after P", 20 union of one set rule and one after rule (truth `two`). Assess per pole: 30 highest-activation + 30 random contexts (amendment from R1b-0).
- C1 bars: mean r3 over single-rule poles >= 0.7; `one` recall >= 0.8 on single-rule poles; `two` detection >= 0.8 on unions.
- C2 (run only if C1 passes): each pole scored with another pole's jobs (derangement, seed 7); mean r3 <= 0.1 and < 5% explained at 0.5.
- If C1 fails, R1b closes as "labeller too weak" and no real-pole percentages are produced. No retuning of prompts, bars or model within this identity.
- Limits: synthetic rules are simpler than real neurons; passing C1 does not validate the labeller on real poles.
