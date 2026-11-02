# Qwen3.8-27B exploratory continuation

State slice: `interpretability-lab-sandbox-v1`.

The candidate packet is [`configs/qwen3.8-27b-assessment-v1.json`](../configs/qwen3.8-27b-assessment-v1.json), digest `bcef5171b7f5a3fad8f1872e9da8fbc3a9591e5a7fb34ad457052e60359f923a`.

The internal-only qualification packet is
[`configs/qwen3.8-27b-internal-qualification-v1.json`](../configs/qwen3.8-27b-internal-qualification-v1.json), digest
`cea1aabdd75055456d7a68b25077f542e3f1719441c93a3749eabdb90de4d013`.  It
does not require external review, and its claim ceiling remains local internal
exploration.

That design is retained as historical input-discovery evidence. The corrected
internal packet is
[`configs/qwen3.8-27b-internal-qualification-v2.json`](../configs/qwen3.8-27b-internal-qualification-v2.json); its asset revision is
`0f965454ffedb93e826546a7dc11b51a4622188e`, the commit that contains the
declared `v2/sae_L32_final.pt` file. The v2 packet is retained as the
4.57.3 compatibility probe; the v3 packet is the only packet eligible for a
real internal run under its pinned runtime.

The v2 model-load probe showed that Transformers 4.57.3 does not register the
`qwen3_5` model type. That probe is closed as a runtime incompatibility. The
corrected v3 internal design
[`configs/qwen3.8-27b-internal-qualification-v3-corrected.json`](../configs/qwen3.8-27b-internal-qualification-v3-corrected.json)
binds the deterministic internal corpus identity. Its custody-bound packet
pins Transformers source revision `a4aa98904e41952cc515fef42c285eecacd18a65`
(`5.18.0.dev0`), which registers the required architecture. The resulting
qualification is recorded in
[`docs/QWEN38_INTERNAL_V3_RESULT.md`](QWEN38_INTERNAL_V3_RESULT.md).

The model identity is `Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.  It is a 27B, 64-layer
multimodal checkpoint.  The adapter freezes the text-only decoder path
`model.language_model.layers[32]`; it does not infer a layer from a mutable
module name.

The feature candidate is `LeonInverse/qwen3.8-27b-scope-code` at revision
`0f965454ffedb93e826546a7dc11b51a4622188e`, path `v2/sae_L32_final.pt`, with
`d_in=5120`, `d_sae=40960`, and TopK `k=50`.  The asset is self-trained on a
code-domain corpus.  Its published reconstruction figures are not accepted as
our evidence; fresh custody, schema checks, pooled reconstruction, controls,
and causal intervention checks are required.

The external packet remains a design freeze and still requires independent
review before held-out effects. The internal v3 packet has a fresh owner-only
custody root and live H100 attestation, but its fixed reconstruction gate
failed. No weights or asset files are stored in Git. The GPT-2 packet remains
immutable.

For internal exploration, execution order is fixed: acquire and hash external
inputs, freeze the runtime supporting `Qwen3_5ForConditionalGeneration`, bind
the packet with `scripts/freeze_internal_qwen38_packet.py`, create a local
authorization and live H100 attestation, write the packet-bound custody
manifest, run native/instrumented parity and reconstruction, and retain only
aggregate qualification output. The runner rejects null packet hashes,
missing manifests, path substitutions, and unfrozen Transformers versions.
The v3 run passed parity, repeatability, no-op, intervention reach, feature
stability, and scrubbing reach, then failed the fixed reconstruction gate at
NMSE `0.2530649293983919` against `<=0.05`; it is closed without retuning.
Held-out causal effects and continual-learning results remain closed in this
lane. The external packet follows the same sequence but also requires an
independent packet-bound Ed25519 acceptance before any held-out effects. The
USD 333 ceiling remains the maximum spend.

The next isolated continuation is the stage1 asset design
[`configs/qwen3.8-27b-internal-qualification-v4-stage1.json`](../configs/qwen3.8-27b-internal-qualification-v4-stage1.json),
with local binding
[`configs/internal-real-qwen38-v4-stage1.json`](../configs/internal-real-qwen38-v4-stage1.json).
It changes only the predeclared SAE path to `stage1/sae_L32_final.pt`, binds a
new corpus identity, and carries a new design digest. The upstream model card
describes stage1 as a separate 50M-token reference checkpoint; that description
does not substitute for a fresh reconstruction measurement. The v4 design has
no custody-bound hashes yet and therefore authorizes no model execution.

The v4 stage1 continuation is now closed as a second reconstruction failure.
Its aggregate-only record is
[`docs/QWEN38_INTERNAL_V4_RESULT.md`](QWEN38_INTERNAL_V4_RESULT.md). The
predeclared stage1 asset produced NMSE `0.2636105839973192`, so no Weco,
causal assessment, or continual-learning experiment may start from v4.
