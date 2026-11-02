# Gemma 3 V10 independent-review handoff

State slice: `interpretability-lab-sandbox-v1`.

Review these exact bytes:

- protocol: [`configs/gemma3-27b-heldout-causal-retention-v10-design-only.json`](../../configs/gemma3-27b-heldout-causal-retention-v10-design-only.json)
- protocol description: [`docs/GEMMA3_V10_ASSESSMENT_PLAN.md`](../GEMMA3_V10_ASSESSMENT_PLAN.md)
- review request: [`docs/review/gemma3-v10-review-packet.json`](gemma3-v10-review-packet.json)

The review-request digest is
`866ac7b5470afcae6d34802ee070a8037094a56c10872b4d6cbaf3eba4dc586b`; the
protocol digest is
`1eb483339b3c938acf684ab15b28905ae2a3d971e91a6471b39264ecf4e6f8db`.

Review the model and official feature identity, affine-transcoder orientation,
layer and dimension binding, fresh-corpus and donor rules, fixed PEFT update
recipe, reconstruction and cross-checkpoint gates, held-out estimand and
controls, prediction lock, Holm statistics, compute matching, retention
policy, and claim ceiling.

Return either `ACCEPT` or `REJECT`. An `ACCEPT` must be an
`interp-lab.external-review-receipt.v1` object with:

- reviewer-controlled Ed25519 public-key registry entry;
- reviewer identity and conflict-of-interest statement;
- signed payload binding the exact review-request packet digest;
- strict base64 Ed25519 signature over the canonical signed payload.

The operator must not generate this receipt. A local key, synthetic signature,
or same-control-plane review is invalid. This request reviews the protocol
design; before model execution, fresh corpus and base/post-update checkpoint
manifests must be acquired, bound into a custody packet, and accepted against
that exact execution digest as well.
