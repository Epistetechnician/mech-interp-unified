# Qwen3.8-27B stage1 internal qualification result

State slice: `interpretability-lab-sandbox-v1`.

This fresh internal exploratory slice tested the predeclared stage1 SAE on a
new 18-row deterministic corpus. It ran offline on the pinned Qwen3.8-27B
checkpoint and closed without retuning when the fixed reconstruction gate
failed. Only an aggregate report remains in the owner-only custody root.

Frozen identities:

- Packet digest: `7143a6675e9ccaa94108f106d5f900967675d9191bd42af81a53bd8195553c2c`
- Model revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Model tree digest: `aa2558f48f1aaa84d68b8d423841d95acae550071f3fcd9d716d065416921e45`
- SAE: `stage1/sae_L32_final.pt` at revision `0f965454ffedb93e826546a7dc11b51a4622188e`
- SAE SHA-256: `6a106f453461d19a5063604f1993007f64618980a51e32c9e7d2695139d9b489`
- Corpus identity: `fresh-qwen38-27b-code-internal-20260915-v4`
- Corpus SHA-256: `d5b887d87703eba3e9209b30b72dddb79910d3579ad1274c100583145b755893`
- Input custody manifest: `2ffb39e965708b26e9f10460ad3fcc3e6acd0ad0509e33e4d65e7f67831eee2f`
- Runtime: Python `3.11.11`, Torch `2.6.0` (`cu124`), CUDA `12.4`, Transformers `5.18.0.dev0`
- Device: NVIDIA H100 80GB HBM3, BF16 enabled, UUID `GPU-536a3c65-776e-1287-fef1-7dad043b45b4`

Gate results:

| Gate | Observed | Result |
| --- | ---: | --- |
| Native/instrumented parity | `0.0` | pass |
| Deterministic repeatability | `0.0` | pass |
| Exact no-op | `0.0` | pass |
| Nonzero intervention reach | `0.89111328125` | pass |
| Feature reconstruction NMSE | `0.2636105839973192` | **fail; threshold `<=0.05`** |
| Feature repeat stability | `0.9999999999999999` | pass |
| Causal-scrubbing reach | `1.015625` | pass |

The report digest is
`96a2df11e8c07cd46b0405e883190231900aa06f525854c7ac774fb155664b53`.
The report and packet hashes were recomputed after execution, and the result
directory contains only `qualification.json` with mode `0600`.

This slice is closed as `QualificationFailedFeatureReconstruction`. The
stage1 swap did not improve reconstruction over v3, so another asset swap or
threshold adjustment would be post hoc tuning. The intervention and
scrubbing values demonstrate reach to logits only; they do not establish
causal validity, held-out effects, continual-learning retention, alignment, or
production readiness. Claim ceiling: `LOCAL_INTERNAL_EXPLORATORY_ONLY`.

The H100 was stopped after validation. Provider usage recorded 54.67 minutes
and `$3.6408`, below the authorized `$333` ceiling. No Weco or downstream
scientific experiment ran.
