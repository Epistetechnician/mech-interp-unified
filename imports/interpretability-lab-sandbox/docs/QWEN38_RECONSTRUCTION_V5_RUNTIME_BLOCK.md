# Qwen3.8 V5 runtime qualification record

State slice: `interpretability-lab-sandbox-v1`.

The layer-33 V5 design was not executed.  Its fixed runtime canary loaded the
exact cached checkpoint configuration with Python 3.11.11, Torch 2.6.0, and
Transformers 4.57.3, then stopped before model construction because the
checkpoint reports `model_type: qwen3_5`, which that Transformers release does
not register.  The canary was Gman job `job-8aepc`; its exit code was `1` and
no model or activation was loaded.

The correction is a new V6 design and packet identity.  The pinned
Transformers source tarball for commit
`dbea10932a0e18f40d8103a3bae6fa7927f8d36e` installed as `5.18.0.dev0` in
Gman job `job-ktr3s`.  Configuration loading returned
`Qwen3_5ForConditionalGeneration`, and the class-import canary `job-5wnmr`
loaded `AutoProcessor` and `AutoModelForMultimodalLM` at the same revision.
These are runtime canaries only; they are not reconstruction or causal
results.

The first V6 qualification submission (`job-hiwf9`) also stopped before model
loading when the runner passed the packet's public hook IDs directly to the
adapter's internal aliases.  The failure was deterministic and isolated to
the adapter boundary; it consumed 274 seconds of the bounded H100 attempt.
The runner now uses an explicit `decoder_input -> decoder_in` and
`decoder_output -> decoder_out` registry, with a test that rejects unregistered
IDs.  The packet identity and all scientific inputs remain unchanged.

The V6 design keeps the model revision, layer-33 asset, corpus split, hook
order, reconstruction estimand, thresholds, offline execution rule, and
owner-only custody requirements unchanged.  It records the source/runtime
correction in
[`configs/qwen3.8-27b-reconstruction-v6-transformers-main-design-only.json`](../configs/qwen3.8-27b-reconstruction-v6-transformers-main-design-only.json).
The V5 packet, if later materialized, remains closed at the runtime gate and
cannot consume V6 output.

The V6 rerun then stopped at processor construction because the multimodal
processor requires Pillow and Torchvision.  Gman job `job-5sjug` exited with
that missing-dependency error after 275 seconds; no model weights or
activations were loaded.  The new V7 design pins the PyTorch-supported pair
Torchvision `0.21.0` with Torch `2.6.0`, plus Pillow `11.3.0`, and records the
runtime check in the packet.  V7 is a separate design identity and has no
scientific result; its V8 continuation is recorded below.

The V8 adapter correction added all-token capture and passed the local
contract suite. Gman job `job-dst3e` loaded the exact model on an H100 and
exercised both frozen layer-33 hook candidates, but neither passed the fixed
fit reconstruction gate. It exited with code `1` after 363 billed seconds
(`$0.2992`). No reconstruction result, causal effect, Weco run, or
continual-learning result was emitted. V8 is closed without retuning; see
[`QWEN38_INTERNAL_V8_RESULT.md`](QWEN38_INTERNAL_V8_RESULT.md).
