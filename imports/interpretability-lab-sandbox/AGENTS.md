# Interpretability Lab Sandbox rules

All mutations in this repository belong to the named state slice
`interpretability-lab-sandbox-v1`.

This repository is a production-shaped sandbox, not a license to bypass
custody, provenance, or provider authorization. Synthetic execution is the
default and is the only mode enabled by the checked-in configuration.
Real model weights, feature assets, private corpus bytes, activations, traces,
and generated results stay outside Git under an owner-only custody root.

External projects are references or pinned submodules only; their source is not
copied into this repository. A real internal exploratory run requires an exact
packet, custody manifest, pinned runtime, local authorization record, and live
runtime attestation. Internal authorization is explicitly not independent
review, proof, publication evidence, or production authorization. An external
packet-bound Ed25519 `ACCEPT` remains required for any externally validated
lane. This sandbox never promotes synthetic or internal output to independent
scientific, alignment, continual-learning, or production claims.

The internal lane is intentionally available for bounded engineering
qualification without an external reviewer. It may exercise a custody-bound
real adapter, parity, repeatability, no-op, intervention reach, and
reconstruction checks under a local claim ceiling. It may not execute held-out
causal assessment, causal scrubbing assessment, or continual-learning
retention experiments under that lane, and it may not relabel internal output
as independent evidence. Those activities remain behind the exact external
packet-bound Ed25519 acceptance gate.

Measure twice, cut once. Keep the tree free of credentials, raw model data,
traces, temporary files, and generated result payloads. Every mutation must
name this state slice in its commit message or artifact metadata.
