# R5 result (run once, protocol unchanged)

Verdict per frozen rule: **`detects_planted_superposition = false`**. H1 (power) failed, H2 (monotone) held, H3 (random-arm specificity) held. Source: `results/r5_planted_superposition.json`, runner `experiments/r5_planted_superposition.py`.

Mean multi-job fraction M at tau=0.2 (10 seeds): S=0.0: 0.87, 0.5: 0.99, 0.9: 1.00, 0.95: 1.00. M(0.95)-M(0.0)=0.13 < 0.30 required. Random-weight arm: 0.95-0.97 at every S.

Reading, stated plainly:
- The correlation-threshold job audit cannot separate planted superposition from its control. It reports almost every neuron as multi-job in all conditions, including random weights.
- Cause (diagnosed after the run, not used to change the verdict): hidden width d=10 < n=20 features means even the dense (S=0) condition cannot give each feature its own neuron, and a pre-activation linear readout correlates with several features by construction. The control condition is therefore not a clean "no superposition" case, and the Pearson>=tau job definition is too permissive.
- Threshold sensitivity (tau 0.1-0.4) is non-monotone and collapses at 0.4, so no tau rescues it.
- Consequence: any "no superposition found" claim from a threshold-style neuron audit is uninformative until the audit passes a corrected positive control. R1 must not use this job definition without R5b.

Nothing was retuned. R5b (fresh identity, to be preregistered): n=d control (no superposition possible at S=0), ground-truth per-neuron participation ratio as the planted measure, and an audit that thresholds on causal/ablation contribution rather than raw correlation.
