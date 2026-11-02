# GPT-2 residual-mid continuation

State slice: `interpretability-lab-sandbox-v1`.

The original GPT-2 MLP-out run is closed historical evidence. Its exact asset,
threshold, corpus, and report are not modified or rerun. This continuation has
a new packet and corpus identity and uses the model-matched residual-mid SAE at
the pre-`ln_2` input (`transformer.h[6].ln_2.input`). The upstream asset reports
0.9233 explained variance for this hook family; the continuation therefore
freezes a pooled, globally centered NMSE ceiling of 0.10 before execution.

The runner acquires the exact model and asset revisions outside Git, verifies an
owner-only custody root and internal authorization, then runs these gates in
order: native/instrumented parity, deterministic repeatability, exact no-op,
nonzero intervention reach, pooled residual-mid reconstruction, same-checkpoint
feature repeat stability, and held-out causal scrubbing.

The result is aggregate-only and capped at
`LOCAL_INTERNAL_REAL_MODEL_CONTINUATION_QUALIFICATION_ONLY`. Even if all gates
pass, this is not independent review, a causal assessment, continual-learning
evidence, an alignment result, or production authorization.
