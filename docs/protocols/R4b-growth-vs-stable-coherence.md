# R4b protocol: directed growth vs stable coherence (fresh identity)

Frozen before R4b code runs. Slice `mech-interp-unified-r4b`. Same models, text, checkpoints, neuron selection, token-set definition, embedding theta (0.0941653847694397, from R4) as R4; recomputed deterministically. Ceiling `LocalDevelopmentUnreviewed`, spend 0. Addresses R4's confound: a neuron with a fixed concept looks "adjacent" at every checkpoint.

Checkpoints in order: c1=step16000, c2=step32000, c3=step64000, c4=step100000, c5=step143000. T(n,c) as in R4. Relatedness is the embedding relation only (max cosine of final input-embedding rows >= theta). Substring relation is dropped (R4 showed it is artifact-contaminated).

## Test A: persistence
Added token = in T(n,c_next) not in T(n,c_prev), for transitions c1->c2, c2->c3, c3->c4. Persistent = in T(n,c5); transient = not. Statistic: gap_A = mean(related to T(n,c_prev) | persistent) - mean(... | transient), events pooled per layer. Null: permute persistent/transient labels within each (neuron, transition) group, 1000 permutations; p = (1+#null>=obs)/1001. Pass iff gap_A >= 0.10, p <= 0.01, and >= 50 events in each class. Logic: genuine attachments should be semantically anchored and persist; noise tokens around a fixed concept should not differ.

## Test C: frontier vs core
Transitions c2->c3, c3->c4, c4->c5. For each added token t of neuron n: frontier = tokens added in the previous transition (T(n,c_prev)\T(n,c_prevprev)); core = T(n,c1). Require both nonempty. d(t) = [t related to frontier] - [t related to core]. Neuron statistic D_n = mean of d over its events; layer statistic = mean of D_n over neurons with >= 1 event, requiring >= 30 such neurons. Null: sign-flip of D_n across neurons, 10000 permutations, two-sided p.
- Growth-direction pass iff mean >= +0.05 and p <= 0.01 (new tokens sit closer to the recently added frontier than to the original core, as chained offshoots predict).
- Stable-coherence pass iff mean <= -0.05 and p <= 0.01 (new tokens sit closer to the original core).

## Verdict (per layers 3, 6, 9)
- **Directed growth supported** iff Test A passes and Test C growth-direction passes, in the same >= 2 layers.
- **Stable coherence favored** iff Test C stable-coherence passes in >= 2 layers.
- Otherwise **inconclusive**. Limits as R4 (small model, one corpus, 5 coarse checkpoints, token level, embedding proxy). No threshold changes after data.
