# Held-out causal and continual-learning assessment

State slice: `interpretability-lab-sandbox-v1`.

The frozen packet is [`configs/gpt2-resid-mid-assessment-v1.json`](../configs/gpt2-resid-mid-assessment-v1.json). It binds the GPT-2 residual-mid hook, the model-matched layer-6 SAE identity, a fresh 144-family panel (`48/48/48` fit/tune/assessment), the paired final-token logit-margin estimand, exact-copy/activation-only/text-only/shuffled/constant/matched controls, prediction locking, and a USD 333 hard ceiling.

Its statistical contract is fixed at three repeats, family-level independent
units, ICC assumption `0.50`, minimum standardized effect `0.50`, two-sided
normal design-effect planning, alpha `0.05`, four Holm-controlled endpoints,
and target power `0.80`. The packet records that this is a planning target;
familywise power remains unvalidated until the fresh panel is independently
reviewed and executed.

The packet is currently `FROZEN_DESIGN_REQUIRES_EXTERNAL_CUSTODY`. It is a protocol freeze, not an execution authorization. The gate in [`scripts/check_assessment_gate.py`](../scripts/check_assessment_gate.py) remains closed because the fresh corpus digest, base and post-update checkpoint digests, owner-only custody root, and reviewer-controlled packet-bound Ed25519 `ACCEPT` are absent.

The packet also freezes Python `3.11.11`, Torch `2.6.0`, Transformers
`4.57.3`, safetensors `0.8.0`, CUDA `12.4`, an NVIDIA H100 device, float32
inference, deterministic Torch/CUDA settings, and separate SHA-256 fields for
the corpus, both checkpoint custody manifests, and the two SAE files. A
custody-ready packet must populate every one of those fields.

For each checkpoint, the packet `sha256` is the canonical digest emitted by
`file_manifest(checkpoint_root)`, so the model directory is verified as a set
of exact files rather than by an unbound label.

The external receipt must use the checked-in verifier contract: an
`interp-lab.external-review-receipt.v1` object with an
`interp-lab.reviewer-registry-entry.v1`, a canonical signed payload containing
the packet digest, reviewer identity, public key, and conflict statement, and a
strict base64 Ed25519 signature. `src/interp_lab/review.py` verifies the bytes;
it does not turn an operator-controlled key or a local test key into an
independent reviewer.

After external inputs are acquired, create a new packet inside the custody
root with `scripts/assemble_assessment_packet.py`. It verifies the design
packet, corpus roster, donor restrictions, prediction lock, checkpoint
manifests, and SAE file hashes, then emits a new digest-bound packet and an
intake record. It does not modify the Git design packet or produce an ACCEPT.

## Measurement order

1. Acquire the fresh corpus and both checkpoint artifacts into an owner-only root; hash them and bind the hashes to the packet.
2. Capture fit and tune observations offline. Select the feature and sign rule only from fit/tune data, then freeze the prediction lock before reading assessment effects.
3. Run the held-out intervention and scrubbing evaluator. Every assessment row must contain all fixed arms, with zero attrition and exact-copy no-op.
4. Run the independent aggregate validator against the custody-held trial inputs and report digest.
5. Run the continual-learning panel only after the causal gates pass. Every controller receives the same ordered stream, update count, learning-progress target, and total compute budget. Monitoring units are reported separately and cheaper policies are padded to the same fixed budget.
6. Delete raw prompts, activations, logits, states, and per-trial outcomes within 72 hours of independent validation. Retain aggregate and digest records only.

The corpus runner requires exact row fields `row_id`, `split`, `family`,
`document_id`, `text`, `target_token_id`, `contrast_token_id`, `donor_row_id`,
and `shuffled_donor_row_id`. Families and documents must be unique and
disjoint across all three splits; assessment donors must come from fit or tune.

The local synthetic command [`scripts/run_synthetic_assessment.py`](../scripts/run_synthetic_assessment.py) exercises the estimator, validator, and matched controller accounting on fresh synthetic identifiers. Its `LOCAL_SYNTHETIC_ONLY` output is a contract check and cannot be used as real-model evidence.

A passing internal qualification or local aggregate validator does not create independent review, causal proof, continual-learning evidence, alignment evidence, or production authorization.
