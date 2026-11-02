# Internal real-model run

State slice: `interpretability-lab-sandbox-v1`.

`scripts/run_real_gpt2.py` is the first real adapter lane. It loads the
public, pinned `openai-community/gpt2` checkpoint and the model-matched
`jbloom/GPT2-Small-OAI-v5-32k-mlp-out-SAEs` affine asset from a fresh,
owner-only custody root. The canonical hook is
`transformer.h[6].mlp`; the adapter rejects any other layer. Model execution
is offline after acquisition, deterministic, and aggregate-only.

The lane runs qualification in this fixed order:

1. native versus hooked parity;
2. deterministic repeatability;
3. exact no-op patch;
4. nonzero intervention reach;
5. SAE reconstruction;
6. same-checkpoint feature repeat stability;
7. held-out causal-scrubbing reach.

The packet and local authorization are checked by the internal execution gate
before weights are loaded. Internal authorization is operator-controlled and
does not create independent review, proof, publication evidence, alignment
evidence, or continual-learning evidence. A passing result is bounded by
`LOCAL_INTERNAL_EXPLORATORY_ONLY`. Gemma remains a separate gated lane and is
not silently substituted by this public checkpoint.

The runner writes only an aggregate report and digest. Raw model files and
temporary prompt/activation values belong below the external owner-only
custody root and must be deleted within 72 hours of validation.
