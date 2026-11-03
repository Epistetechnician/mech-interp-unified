# R1b result: closed, instrument too weak (negative on the instrument)

Verdict: `labeller_too_weak` (frozen R1b-1) and `simulator_too_weak` (dev diagnostic). No real-neuron audit was run.

## Evidence
- R1b-1 (frozen, 39 rule poles scored, stopped early): best possible mean r3 on single-job poles 0.173; set rules 0.253; after-token rules 0.045; one-job recall 0.0. C1 bar (r3 >= 0.7) failed by design; C2 not run per protocol.
- Dev diagnostic with the *true rule text* as the description (oracle), Qwen3.5-35B simulator, 12 dev poles:
  - digit_prev simulator: set 0.578, after 0.406, mean 0.492
  - binary_prev simulator: set 0.217, after 0.101, mean 0.159
- Pre-stated rule: freeze a variant only if oracle mean r >= 0.8. Neither did, so R1b closes without further tuning.

## Interpretation
Even a perfect description does not recover known rules, so the simulator (soft scores, weak on preceding-token context) caps r, not the labeller. A C1 pass was unreachable with this instrument.

## Does not show
Nothing about whether neurons are clean at the essay's 32B scale, nor about Qwen3-1.7B-Base neurons. It only shows this local simulator cannot grade descriptions.

## Next if revisited (new protocol identity, not a retune)
Stronger or fine-tuned simulator, or an intervention-based (causal) scorer instead of simulation correlation (Delphi / Transluce style).

Artifacts: results/r1b1_c1c2_stopped.json, r1b1_poles.jsonl, r1b2_dev_diag*.json/.log; code experiments/r1b/{data,diag_dev,diag_dev2,sim2}.py.
