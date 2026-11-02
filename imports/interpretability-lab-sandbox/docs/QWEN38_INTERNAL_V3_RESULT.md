# Qwen3.8-27B internal qualification result

State slice: `interpretability-lab-sandbox-v1`.

The internal exploratory qualification ran once on the pinned local Qwen3.8-27B
checkpoint and then closed with a failed fixed scientific gate. The run
was offline after acquisition, used the canonical
`model.language_model.layers[32]` hook, and emitted one aggregate-only result
under the owner-only custody root. No raw traces were retained in the
repository.

## Frozen identities

- Packet: `qwen38-27b-internal-qualification-20260915-v3-custody`
- Packet digest: `5bf2ca32ab2d34cc6c77a356db9f95a15de57f10f0655bd2a4555e8ea4c9c65e`
- Model: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Model custody tree digest: `aa2558f48f1aaa84d68b8d423841d95acae550071f3fcd9d716d065416921e45`
- SAE: `LeonInverse/qwen3.8-27b-scope-code`, revision `0f965454ffedb93e826546a7dc11b51a4622188e`, `v2/sae_L32_final.pt`
- SAE SHA-256: `1ef1cabe0ac77b298be6793fac3da1abaed130d88564670a58dd7a1c513c89bc`
- Corpus: `fresh-qwen38-27b-code-internal-20260915-v3`, 18 rows (`6/6/6`)
- Corpus SHA-256: `5c737a2314de40ae2be04b92a7402630d5dd8cf24d0088c4af0847f5414251b4`
- Input custody manifest digest: `dd6d8f958805419e0a4f4c81df1265563c1928f0487bc17fe0e74309b7ac0f3d`
- Runtime: Python `3.11.11`, Torch `2.6.0` (`cu124`), CUDA `12.4`, safetensors `0.8.0`, Transformers `5.18.0.dev0`
- Transformers source revision: `a4aa98904e41952cc515fef42c285eecacd18a65`
- Device: NVIDIA H100 80GB HBM3, BF16 enabled, UUID `GPU-536a3c65-776e-1287-fef1-7dad043b45b4`

## Qualification gates

| Gate | Observed | Result |
| --- | ---: | --- |
| Native/instrumented parity (maximum logit delta) | `0.0` | pass |
| Deterministic repeatability (maximum activation delta) | `0.0` | pass |
| Exact no-op (maximum logit delta) | `0.0` | pass |
| Nonzero intervention reach (absolute logit effect) | `0.625` | pass |
| Feature reconstruction (maximum NMSE) | `0.2530649293983919` | **fail; threshold `<=0.05`** |
| Feature repeat stability (minimum cosine) | `0.9999999999999999` | pass |
| Causal-scrubbing reach (maximum absolute logit effect) | `14.953125` | pass |

The result report digest is
`f524546f8fead333ce65f720d7e0572c61285ad62893e197fe011ad8aca198f1`. Its
`all_gates_pass` value is `false`, with exactly one failed gate:
`feature_reconstruction`.

The reconstruction failure closes this packet as
`QualificationFailedFeatureReconstruction`. The threshold, normalization,
layer, asset, corpus, and controls are not retuned under this identity. The
passing intervention and scrubbing-reach values establish that the hook can
reach logits; they do not establish causal validity, held-out effects, or a
continual-learning retention benefit.

The claim ceiling is `LOCAL_INTERNAL_EXPLORATORY_ONLY`. This record is not
independent review, proof, held-out scientific assessment, a benchmark, a
continual-learning result, an alignment result, or production evidence.

The H100 node was stopped immediately after validation. The provider usage
record reported 47.67 H100 minutes and `$3.1746` for this node, below the
authorized `$333` ceiling. No Weco run or downstream scientific experiment
was authorized after the failed gate.
