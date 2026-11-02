# Qwen3.8 feature compatibility audit and V9 decision

State slice: `interpretability-lab-sandbox-v1`.

Audit mode: design-only. No model, feature weights, corpus, activations, or
H100 node was used for this audit. The V8 packet and its failed qualification
remain immutable historical evidence.

## Decision

The Qwen3.8 candidate asset is dimensionally and schematically compatible, but
it is not a qualified feature asset for a new causal experiment. V8 tested the
fresh all-token path against the locked fit reconstruction gates and neither
predeclared layer-33 hook passed. The asset was trained on a coding-domain
corpus, while the proposed scientific corpus is broader and held out. Its
published FVU range is therefore a description of the upstream training
evaluation, not evidence for this model/corpus pair.

The current official Qwen model listing contains Qwen3.5 SAE releases, but no
official Qwen3.8-27B SAE release. A Qwen3.5 asset cannot be substituted for the
Qwen3.8 checkpoint. The Qwen3.8 path is closed for V9. The next V9 packet uses
the official Gemma 3 27B model together with the official Gemma Scope 2 27B
asset whose metadata names that exact model.

## Qwen3.8 audit matrix

| Property | Frozen model/asset fact | Audit result |
| --- | --- | --- |
| Model architecture | `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; Transformers reports the `qwen3_5` multimodal wrapper | Identity recorded; no substitution |
| Model dimensions | 64 decoder layers, hidden width 5120 | Matches the V8 packet |
| Canonical hook | `model.language_model.layers[33]`; `decoder_in` and `decoder_out` were the only declared candidates | Path is explicit and validated by the adapter |
| SAE orientation | `W_enc` is `[5120, 40960]`; `W_dec` is `[40960, 5120]` | Matches the loader and checkpoint schema |
| Biases and sparsity | `b_pre` has 5120 entries; `b_enc` has 40960 entries; Top-K 50; width 40960 | Matches the frozen V8 contract |
| Scale | Stored scalar; encoding applies `(x * scale - b_pre) @ W_enc + b_enc`; decoding applies `(features @ W_dec + b_pre) / scale` | Matches the implementation; no post-hoc rescaling allowed |
| Normalization | Upstream describes `sqrt(d) / mean(norm(x))` input scaling | The scale is asset-owned; runtime normalization must not be added |
| Layer provenance | `v2/sae_L33_final.pt` is the layer-33 checkpoint in the upstream repository | Provenance is explicit; hook validity still requires reconstruction |
| Training domain | 110M coding tokens for v2, including on-policy Qwen3.8 code generations and code datasets | Domain shift relative to the proposed broad corpus; scientific generalization is unproven |
| Empirical reconstruction | V8: neither predeclared hook passed pooled FVU `<=0.20` and maximum row NMSE `<=0.40` | **Fail**; the asset is not eligible for causal/continual experiments |

The matrix establishes a schema and dimensional match. It does not establish
that the learned representation reconstructs the actual model activations with
the required fidelity or that feature interventions are causal.

The Gemma Scope 2 asset audit also verified the affine-transcoder tensor
roster: `w_enc` `[5376, 65536]`, `w_dec` `[65536, 5376]`, `b_enc` and
`threshold` `[65536]`, `b_dec` `[5376]`, and `affine_skip_connection`
`[5376, 5376]`. Its JumpReLU path is `x @ w_enc + b_enc`, followed by the
official hard mask `(pre > threshold) * relu(pre)`, and its reconstruction is
`features @ w_dec + b_dec + x @ affine_skip_connection`. The runtime therefore binds the output hook to
`model.language_model.layers[31].post_feedforward_layernorm`, the exact output
named by the official transcoder config, rather than the enclosing decoder
block.

## Official asset search

The upstream Qwen organization lists SAE releases for Qwen3, Qwen3.5, and
other model families. The listed 27B release is
`Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50` (and its L0_100 variant), not a
Qwen3.8-27B release. This is an architecture-family mismatch and is excluded.

The official Gemma Scope 2 27B repository states that its SAEs and transcoders
are trained for Gemma 3 27B PT. Its `transcoder/layer_31_width_65k_l0_small_affine`
metadata names `google/gemma-3-27b-pt`, layer-31 input/output hook points,
width 65536, and an affine connection. This is the first candidate with an
official model/feature identity pair suitable for a fresh qualification.

## V9 stop conditions

1. Do not acquire or run Qwen3.8 again under this V9 identity.
2. Do not import Qwen V3–V8 activations, corpora, effects, or results into the
   Gemma packet.
3. Do not use the upstream examples file as a reconstruction target; fresh
   model-derived activation rows are required.
4. Do not run Weco, causal intervention, causal scrubbing, or continual
   learning until Gemma reconstruction passes its frozen gate.
5. Do not manufacture an independent review or Ed25519 receipt. The review
   packet can be prepared, but the externally validated lane remains closed
   until an independent reviewer returns a packet-bound acceptance.

## Sources and evidence

- Upstream Qwen SAE schema, scaling, training corpus, and reported FVU:
  `https://huggingface.co/LeonInverse/qwen3.8-27b-scope-code/blob/main/README.md`
- Official Qwen model and SAE listing:
  `https://huggingface.co/Qwen/models`
- Official Gemma 3 27B model card:
  `https://huggingface.co/google/gemma-3-27b-pt`
- Official Gemma Scope 2 27B model card and release layout:
  `https://huggingface.co/google/gemma-scope-2-27b-pt`
- Model-matched Gemma Scope 2 layer-31 asset metadata:
  `https://huggingface.co/google/gemma-scope-2-27b-pt/blob/main/transcoder/layer_31_width_65k_l0_small_affine/config.json`

Claim ceiling: `LOCAL_DESIGN_AUDIT_ONLY`.
