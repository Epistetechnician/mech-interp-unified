# Internal exploratory run

State slice: `interpretability-lab-sandbox-v1`.

The sandbox can run a real adapter under an internal authorization record. The
record is an operator decision for local exploration. It is not independent
review, a proof, publication evidence, or production authorization.

Before an internal run, freeze a packet containing the exact model revision,
feature asset revision, runtime, corpus manifest, controls, estimand,
thresholds, split identities, prediction lock, and bounded claim ceiling. Add
`packet_digest` as the SHA-256 digest of the packet with that field omitted.
For Qwen3.8, bind the acquired model tree, `v2/sae_L32_final.pt`, and fresh
corpus with `scripts/freeze_internal_qwen38_packet.py`. After creating the
authorization and live runtime-attestation records, run
`scripts/write_internal_custody_manifest.py`. The runner verifies this
manifest and rejects null packet hashes, path substitutions, or an unfrozen
Transformers version before importing a model library.
The authorization object must contain exactly:

```json
{
  "schema": "interp-lab.internal-authorization.v1",
  "state_slice": "interpretability-lab-sandbox-v1",
  "source": "internal",
  "decision": "AUTHORIZE",
  "purpose": "exploration",
  "packet_digest": "<packet digest>",
  "operator_id": "<operator identity>",
  "claim_ceiling": "LOCAL_INTERNAL_EXPLORATORY_ONLY"
}
```

The runtime attestation must contain exactly `schema`, `state_slice`, `source`,
`runtime`, `device`, and `determinism`. Use schema
`interp-lab.runtime-attestation.v1`, source `internal`, a concrete runtime and
device identity, and `determinism.enabled: true`.

Keep the packet, authorization, model, asset, corpus, and raw traces in an
owner-only custody root. Set the root to mode `0700`; retain raw traces for no
longer than the frozen policy. Validate the gate without running a model:

```sh
python3 scripts/check_internal_gate.py \
  --config /path/to/internal-config.json \
  --packet /path/to/packet.json \
  --authorization /path/to/authorization.json \
  --runtime-attestation /path/to/runtime-attestation.json \
  --custody-root /path/to/custody
```

The checked-in configuration is synthetic and keeps real execution disabled.
Changing it to `mode: "internal-real"` is a new, reviewable local state slice.
Every resulting report must retain `LOCAL_INTERNAL_EXPLORATORY_ONLY` and the
nonclaims that distinguish internal exploration from independent validation.
