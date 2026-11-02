# R5 protocol: can the neuron-audit pipeline see planted superposition?

Frozen before any code runs. Slice: `mech-interp-unified-r5`. Ceiling: `LocalDevelopmentUnreviewed`. No model, no network, spend 0.

## Why
The essay finds no real-world superposition circuit. That absence only means something if the audit pipeline would detect superposition when it is known to exist. R5 is that positive control.

## Setup
Toy model (Elhage-style): n=20 features, hidden d=10, input x in R^20 with each feature independently active with probability p = 1 - S (value U(0,1)). Model: y = ReLU(W^T W x + b) trained to reconstruct x (Adam, 6000 steps, batch 1024, lr 1e-3). Conditions: S in {0.0, 0.5, 0.9, 0.95}. 10 seeds each. Control arm "random": W random with same row norms, untrained.

## Pipeline under test (the neuron audit)
Hidden neuron h_j = (W x)_j pre-ReLU, evaluated on 20000 fresh samples. A feature i is a *job* of neuron j iff Pearson(h_j, x_i) >= tau where tau = 0.2. Neuron has "one main job" iff exactly one job; "multi-job" iff >= 2 jobs; "unexplained" iff 0. Report the fractions per condition. Primary statistic M = fraction of multi-job neurons.

## Hypotheses (preregistered)
- H1 (power): mean M at S=0.95 exceeds mean M at S=0.0 by >= 0.30.
- H2 (monotone): mean M is non-decreasing in S across the four conditions (allowing ties within 0.02).
- H3 (null): in the random arm, M does not differ from the S=0.0 trained value by more than 0.15 absolute (the pipeline does not just report multi-job for any weights).
Threshold sensitivity (reported, not decision-bearing): tau in {0.1, 0.2, 0.3, 0.4}.
## Decision
Pipeline "detects planted superposition" iff H1 and H2 both hold. If H3 fails, the pipeline is unspecific and R1 results must carry that caveat. Failure of H1 closes R5 as "pipeline blind to superposition at this tau" and every later "no superposition found" statement inherits it. No tuning after seeing results.
