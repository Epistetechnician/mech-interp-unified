# R5b protocol: does a causal neuron audit detect planted superposition? (fresh identity)

Frozen before any R5b code runs. Slice `mech-interp-unified-r5b`. Ceiling `LocalDevelopmentUnreviewed`. No model download, spend 0. Supersedes R5's design only; R5's negative result stands.

## Fixes to R5's flaws
1. Privileged basis: hidden layer is ReLU, so neurons are a meaningful basis: h = ReLU(W x), y = ReLU(W^T h + b). (R5's linear hidden layer had rotation symmetry.)
2. Clean no-superposition control: n = d = 10 features/neurons, dense (S = 0), plus a sparse n = d control (S = 0.9).
3. Audit is causal and share-based, not a raw correlation threshold.
4. Ground truth is read from the weights and is never seen by the audit.

## Conditions (10 seeds each, 6000 Adam steps, batch 1024, lr 1e-3, feature value U(0,1) when active)
C1: n=d=10, S=0.0 (no superposition possible). C2: n=d=10, S=0.9. C3-C6: n=20, d=10, S in {0.0, 0.5, 0.9, 0.95}.

## Audit (uses only the trained model and data)
On 20000 fresh samples, for each hidden neuron j: ablate it (set post-ReLU h_j = 0) and compute per-feature reconstruction-error increase D_i = mean((y_abl_i - x_i)^2) - mean((y_i - x_i)^2), clipped at 0. Share s_i = D_i / sum_k D_k (neuron with sum D_k < 1e-6 is "unused"). Job i iff s_i >= 0.2. one-job iff exactly 1; multi-job iff >= 2.

## Ground truth (from weights, held out from the audit)
Row w_j of W; squared shares q_i = w_ji^2 / sum_k w_jk^2. Neuron is GT-multi iff its second-largest q_i >= 0.15; GT-one otherwise; used neurons only.

## Hypotheses
- H1 (power): mean audit multi fraction at C6 (n=20,d=10,S=0.95) minus at C1 >= 0.30.
- H2 (accuracy): balanced accuracy of "audit multi" vs "GT multi", pooled over all used neurons in C1-C6, >= 0.75.
- H3 (null control): audit multi fraction at C1 <= 0.20.
Reported, not decision-bearing: the R5 correlation audit (Pearson >= 0.2 on pre-ReLU activations) on the same models; per-condition multi fractions; threshold sensitivity of the share cutoff in {0.1, 0.2, 0.3}.

## Decision
Audit **validated for R1** iff H1, H2 and H3 all hold. Otherwise it must not be used to claim absence of superposition. No change to thresholds after results.
