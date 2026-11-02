# Qwen3.8 V8 qualification closure

State slice: `interpretability-lab-sandbox-v1`.

V8 was the fresh internal continuation for the exact Qwen3.8-27B revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, the layer-33
`LeonInverse/qwen3.8-27b-scope-code` SAE at revision
`0f965454ffedb93e826546a7dc11b51a4622188e`, and the pinned Transformers main
source revision `dbea10932a0e18f40d8103a3bae6fa7927f8d36e` (`5.18.0.dev0`).
The H100 runtime also pinned Torch `2.6.0`, Torchvision `0.21.0`, Pillow
`11.3.0`, Safetensors `0.8.0`, CUDA `12.4`, BF16, and deterministic CUDA
settings. Model execution was offline after setup, with the fresh V8 custody
root `/hfcache/interp-lab-reconstruction-v8-20260916`.

The adapter correction added the missing all-token capture method and was
verified by the local 46-test suite and focused Ruff checks. The remote job
`job-dst3e` loaded all 1,184 model shards and exercised both predeclared
layer-33 hook candidates. It then exited with code `1` because neither hook
passed the unchanged fit reconstruction gate (`pooled FVU <= 0.20` and
maximum row NMSE `<= 0.40`). The runner emitted no result artifact or
scientific aggregate before that fail-closed exit.

The attempt billed 363 seconds (`$0.2992`) under the workspace ceiling. V8 is
closed without threshold, normalization, hook, asset, corpus, or model
retuning. No Weco optimization, held-out causal intervention, causal
scrubbing assessment, continual-learning comparison, publication claim, or
Evidence Ledger mutation is authorized from this packet. The historical V3,
V4, V5, V6, and V7 records remain immutable and are not V8 inputs.
