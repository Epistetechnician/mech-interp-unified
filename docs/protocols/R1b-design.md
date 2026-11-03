# R1b design: context-aware, calibrated neuron-pole audit

Status: **DESIGN, not frozen, nothing run.** Slice (proposed) `mech-interp-unified-r1b`. Ceiling `LocalDevelopmentUnreviewed`. Zero new downloads, zero spend as designed. Freezing requires the two owner decisions in section 8.

## 1. Why R1 failed and what R1b must fix
R1 (Qwen3-1.7B-Base, token-lookup proxy): 68% of poles not token-predictable, 3% single-cluster, and the planted-composite calibration could not run (6 one-job poles). R5/R5b: threshold-style audits miss diffuse superposed neurons and swing with one cutoff. So R1b needs (a) a labeller that sees context, (b) controls with known answers that run BEFORE any percentage is read, (c) results as curves with the audit's measured miss rate, never a single number.

## 2. Resources already on this machine (checked, no download needed)
- Subject model: Qwen3-1.7B-Base (HF cache, bf16, 3.2 GB). Optional second subject for scale: none cached beyond SmolLM2-135M; Qwen3-0.6B would need a ~1.2 GB download and exceeds the 10 GB budget (see AUTHORIZATION).
- Labeller/simulator candidates (LM Studio dir and HF cache): Qwen3.6-35B-A3B-MLX-4bit (primary), Qwen3.5-9B-GGUF, DeepSeek-R1-0528-Qwen3-8B-MLX-4bit, Qwen2.5-7B-Instruct-1M-GGUF. `mlx_lm` 0.31.3 exists in the `theseus-lab` venv (not in this repo's venv; would be an isolated install, a package download but no model download). `llama-cli`/`llama-server` exist in /opt/homebrew/bin.
- Hardware: 38.7 GB RAM, Apple Silicon (MPS). Stages are run sequentially so the subject model and the 4-bit 35B labeller are never resident together.
- Note: an earlier belief that Qwen2.5-1.5B/3B/7B were cached was wrong (empty directories).

## 3. Method (explain -> simulate -> score, with a logprob simulator)
Per pole (valid by the R1 rule z >= 4, signal x = max(sign*a, 0)), on a fit/assess split of wikitext (new disjoint lines) plus a second text domain if one is already cached (none is; otherwise single-domain, a disclosed limit):
1. **Evidence:** 12 top-activating contexts (64 tokens, activating token marked, activation shown), 6 mid-activation contexts, 6 random contexts, all from fit.
2. **Explainer (labeller):** writes up to three short, separate "jobs" (each a condition on token/context), in a fixed JSON format; temperature 0; one call per pole.
3. **Simulator:** for held-out assess contexts, scores each token 0-9 for each job description using a single forward pass and the expected value of the digit logprobs at each token (no sampling). Simulated signal for a job set = max over jobs of per-job scores.
4. **Scores:** Pearson r between simulated and true x over all assess tokens, for job 1 alone (r1), jobs 1-2 (r2), jobs 1-3 (r3).
5. **Classification (reported as curves over r-threshold in {0.3, 0.5, 0.7}; primary 0.5):** explained iff r3 >= threshold; `one` iff explained and r1 >= threshold and r3 - r1 < 0.05; `two` iff explained, r1 < threshold or r3 - r1 >= 0.05, and r2 >= threshold with r3 - r2 < 0.05; `many` otherwise explained; `unexplained` iff r3 < threshold.

## 4. Calibration (all must run before percentages are interpreted)
- **C1 positive control (rule neurons):** 60 synthetic poles on real wikitext whose activation is a known rule: token-set membership (20), token-after-specific-preceding-word (20), union of two distinct rules (20, ground truth `two`). Required: mean r3 >= 0.7 on single-rule poles; `one` recall >= 0.8 on single-rule poles; `two`-detection >= 0.8 on unions.
- **C2 known-bad (shuffled descriptions):** each real pole scored with another pole's descriptions. Required: mean r3 <= 0.1 and < 5% explained at threshold 0.5.
- **C3 labeller reliability:** rerun the explainer on a different random evidence draw; class agreement >= 0.8; and a second simulator (Qwen3.5-9B) must correlate >= 0.8 with the primary simulator's scores across poles.
- **C4 composite sensitivity on real poles:** composites max(x_a, x_b) of `one` poles from the same layer with distinct top-token sets, >= 30 pairs required, detection (`two` or `many`) >= 0.8.
- **Validated** iff C1-C4 all meet their bars. If C1 fails, R1b closes as "labeller too weak": no percentages are reported, only the failure.

## 5. Sampling and scale
200 neurons (40 per layer x layers 4, 10, 16, 22, 27 of Qwen3-1.7B-Base; seed 0, disjoint from R1's 100 where possible), both poles, so up to 400 poles. Held-out assess: 400 contexts x 64 tokens. Estimated cost with the 4-bit 35B-A3B on MPS: roughly 400 explainer calls (about 3 s each) plus roughly 400 poles x 3 job-sets x 20-30 forward passes of short prompts; a few hours, no spend. A stage-0 timing probe fixes the final N before freezing.

## 6. Improvement over the essay (stage R1c, separate protocol)
For each `one` pole with description d: ablate the neuron (zero down_proj input) and measure change in next-token loss at the description-matching positions versus 100 matched-random neuron ablations (matched on layer and activation magnitude). A "job" must leave a causal footprint beyond matched random. Reuses the Astral V53-V59 control logic; preregistered separately.

## 7. Build order
R1b-0: harness plus hermetic tests with a fake labeller/simulator (no model); timing probe; checks mlx_lm loads the 35B model. R1b-1: run C1 and C2 only (no real poles' percentages). R1b-2: freeze thresholds, run full audit, C3-C4. Each stage ends in a commit before the next; failures close the stage without retuning.

## 8. Owner decisions needed to freeze (everything else is decided here)
1. **Labeller:** approve the local Qwen3.6-35B-A3B-MLX-4bit (recommended; cross-check with Qwen3.5-9B) and an isolated `mlx_lm` install (package only). Alternative: Qwen3.5-9B-GGUF only (weaker, faster).
2. **No hosted/frontier labeller, no spend** (recommended). A frontier API labeller would be stronger but needs a provider, an allocation receipt and a USD cap inside the existing USD 10 ceiling; not planned.
## 9. Limits (declared now)
One small base model, one text domain, 200 sampled neurons, a quantized local labeller (weaker than the essay's agents), a logprob simulator whose validity is exactly what C1-C3 test, and no human audit. R1b can calibrate a proxy audit; it cannot confirm or refute the essay's 32B percentages.
