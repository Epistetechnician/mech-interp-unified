# R5b result (run once, protocol unchanged)

Source `results/r5b_causal_audit.json`, runner `experiments/r5b_causal_audit.py`.

Verdict per frozen rule: **audit NOT validated for R1** (H1 false, H2 true, H3 false).

| Condition | audit multi (share>=0.2) | ground-truth multi | correlation audit multi |
|---|---|---|---|
| C1 n=d=10, S=0 | 0.37 | 0.49 | 0.77 |
| C2 n=d=10, S=0.9 | 0.88 | 0.75 | 0.89 |
| C3 n=20,d=10, S=0 | 0.25 | 0.68 | 0.99 |
| C4 S=0.5 | 0.70 | 0.94 | 0.98 |
| C5 S=0.9 | 0.01 | 0.28 | 0.98 |
| C6 S=0.95 | 0.03 | 0.26 | 0.99 |

- H1 power: C6 - C1 = 0.03 - 0.37 = -0.34 (needed +0.30). Failed, wrong direction.
- H2: balanced accuracy 0.776 >= 0.75 (specificity 0.94, sensitivity 0.61). Passed, only because the audit rarely calls a neuron multi-job when it is not.
- H3: audit multi at C1 = 0.37 > 0.20. Failed.

## Reading
- The causal share audit is far better than the R5 correlation audit (which calls ~99% of neurons multi-job everywhere) but is not a valid superposition detector. At high sparsity (C5, C6) neurons in superposition spread damage across many features, each below a 0.2 share, so the audit reports them as "unexplained" (zero jobs), not multi-job. The share rule systematically misses diffuse polysemantic neurons.
- The "no-superposition" control is itself not clean: even n=d dense (C1) has ground-truth multi fraction 0.49, because ReLU hidden units with dense inputs mix features. So H3's 0.20 bound assumed a cleaner control than the toy model provides. Both failures are recorded as results; thresholds were not changed.
- Share-cutoff sensitivity (0.1/0.2/0.3) swings multi fractions from ~1.0 to ~0 within the same models, i.e. conclusions depend on one arbitrary threshold. This supports the essay's own caveat that its percentages are threshold-dependent.

Consequence for R1: report neuron "jobs" only as threshold curves, and treat "unexplained" as a separate category that may hide superposition. Any "X% of neurons have one main job" statement needs the audit's miss rate on planted ground truth (here sensitivity 0.61 at share 0.2).
