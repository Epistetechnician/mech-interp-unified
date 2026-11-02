# Interpretability Lab Sandbox

State slice: `interpretability-lab-sandbox-v1`.

This repository is a deterministic, production-shaped harness for testing the
interfaces needed by the interpretability and continual-learning program:

- native and instrumented model parity;
- feature encoding, reconstruction, and intervention reach;
- checkpoint feature stability and causal scrubbing;
- explicit circuit lineage and transition hash chains;
- continual-learning controllers with matched-data accounting;
- custody, packet binding, and fail-closed execution gates.

The default suite is synthetic and runs without model libraries, weights,
external data, provider calls, or network access:

```sh
python3 scripts/run_synthetic.py
PYTHONPATH=src python3 -m unittest discover -s tests
```

The synthetic report is labelled `LOCAL_SYNTHETIC_ONLY`. It is useful for
engineering and falsifier development. It is not evidence that an
interpretability method works on a real model, improves retention, establishes
causality, or is production-ready.

The first clean-room real adapter is documented in
[`docs/REAL_RUN.md`](docs/REAL_RUN.md). It targets the pinned public GPT-2
checkpoint and model-matched layer-6 SAE behind the internal authorization
gate. A real run is aggregate-only and closes on the first fixed-gate failure;
its custody and result bytes stay outside Git. Internal execution does not
create independent review, proof, or publication authority.

`third_party/UPSTREAMS.lock.json` records the exact upstream revisions and
licenses that real adapters may consume later. The source is not vendored in
this sandbox. The checked-in configuration keeps real execution disabled. A
future internal exploratory run may use a local authorization record bound to
the exact packet and owner-only custody; that record is not an independent
review, proof, or publication authorization. Externally validated execution
still requires a separate packet-bound Ed25519 receipt.

The closed MLP-out qualification is preserved as historical negative evidence.
The additive residual-mid continuation is documented in
[`docs/REAL_CONTINUATION.md`](docs/REAL_CONTINUATION.md) and uses a fresh packet,
corpus, and model-matched asset identity. It does not reopen the failed slice.

The next held-out causal and matched continual-learning contract is frozen in
[`docs/ASSESSMENT_PLAN.md`](docs/ASSESSMENT_PLAN.md), with its packet at
[`configs/gpt2-resid-mid-assessment-v1.json`](configs/gpt2-resid-mid-assessment-v1.json).
The local synthetic harness passes the estimator and independent aggregate
recomputation; the real assessment gate remains closed until fresh custody,
checkpoint hashes, a prediction-lock digest, and an external packet-bound
Ed25519 acceptance exist.

The bounded Weco engineering lane is recorded in
[`.weco/metrics-throughput-v2/report.md`](.weco/metrics-throughput-v2/report.md).
It found a semantics-checked aggregate-metric throughput candidate in an
isolated copy; the candidate has not been applied to the source and does not
change any scientific claim.

The larger-model continuation is design-only in
[`docs/QWEN38_PLAN.md`](docs/QWEN38_PLAN.md), with its packet at
[`configs/qwen3.8-27b-assessment-v1.json`](configs/qwen3.8-27b-assessment-v1.json).
It freezes the Qwen3.8-27B decoder path and a separately identified code-domain
SAE candidate.  External custody, runtime support, reconstruction gates, and
independent review remain prerequisites; this packet does not authorize model
execution or alter the GPT-2 packet.

The internal qualification counterpart is
[`configs/qwen3.8-27b-internal-qualification-v1.json`](configs/qwen3.8-27b-internal-qualification-v1.json).
It uses the same model and asset identities with an explicitly local claim
ceiling, local authorization, and live runtime attestation.

The corrected internal custody design is
[`configs/qwen3.8-27b-internal-qualification-v2.json`](configs/qwen3.8-27b-internal-qualification-v2.json), with its real-run binding at
[`configs/internal-real-qwen38-v2.json`](configs/internal-real-qwen38-v2.json).
It pins the asset revision that actually contains `v2/sae_L32_final.pt` and
the required Transformers 4.57.3 runtime.

The v2 runtime probe found that 4.57.3 lacks the `qwen3_5` architecture. The
corrected internal design
[`configs/qwen3.8-27b-internal-qualification-v3-corrected.json`](configs/qwen3.8-27b-internal-qualification-v3-corrected.json)
and binding
[`configs/internal-real-qwen38-v3.json`](configs/internal-real-qwen38-v3.json)
pin the Transformers source commit that provides it. The one bounded H100
qualification is recorded in
[`docs/QWEN38_INTERNAL_V3_RESULT.md`](docs/QWEN38_INTERNAL_V3_RESULT.md): all
instrumentation and intervention-reach checks passed, but the fixed feature
reconstruction gate failed at NMSE `0.2530649293983919` versus `<=0.05`.
The packet is closed without retuning.

The internal runner requires a custody-bound packet and manifest before it
loads a real model. Use `scripts/freeze_internal_qwen38_packet.py` after input
acquisition, then `scripts/write_internal_custody_manifest.py`; no real input
hash may remain null at execution time.

The v4 stage1 continuation is closed as a second fixed reconstruction failure;
see [`docs/QWEN38_INTERNAL_V4_RESULT.md`](docs/QWEN38_INTERNAL_V4_RESULT.md).
The aggregate-only NMSE was `0.2636105839973192` against `<=0.05`. No asset
shopping, threshold retuning, Weco search, causal assessment, or
continual-learning run follows this packet.

A fresh design-only reconstruction continuation is recorded in
[`docs/QWEN38_RECONSTRUCTION_V5.md`](docs/QWEN38_RECONSTRUCTION_V5.md) with
its machine-readable boundary at
[`configs/qwen3.8-27b-reconstruction-v5-design-only.json`](configs/qwen3.8-27b-reconstruction-v5-design-only.json).
It predeclares a model-matched Top-K asset, fixed layer-32 input/output hook
selection on fit data, pooled globally-centered FVU, and row-wise coverage
checks.  It has null custody and asset digests by design and cannot execute.
The packet compiler at
[`scripts/freeze_qwen38_reconstruction_v5.py`](scripts/freeze_qwen38_reconstruction_v5.py)
materializes that design only after fresh owner-only inputs and runtime
revisions are supplied; compilation hashes inputs but does not execute or
authorize a run.
The corresponding offline runner is
[`scripts/run_real_qwen38_reconstruction_v5.py`](scripts/run_real_qwen38_reconstruction_v5.py);
it batches all-token reconstruction on CUDA, locks hook selection from fit
data, and emits aggregate-only output after custody and runtime gates pass.

The fresh layer-33 continuation is separately frozen in
[`configs/qwen3.8-27b-reconstruction-v5-layer33-design-only.json`](configs/qwen3.8-27b-reconstruction-v5-layer33-design-only.json).
It binds the unused `v2/sae_L33_final.pt` model-matched asset revision and
cannot consume the closed layer-32 V3/V4 assets.
The Gman orchestration entrypoint
[`scripts/prepare_and_run_qwen38_reconstruction_v5.py`](scripts/prepare_and_run_qwen38_reconstruction_v5.py)
creates the fresh custody packet, attestation, authorization, and manifest,
then enables offline qualification.

The V5 layer-33 runtime canary is closed because Transformers 4.57.3 does not
recognize the checkpoint's `qwen3_5` architecture.  The fresh V6 continuation
pins Transformers `5.18.0.dev0` at source revision
`dbea10932a0e18f40d8103a3bae6fa7927f8d36e`, which passed the H100 architecture
and class-import canaries.  Its design-only packet is
[`configs/qwen3.8-27b-reconstruction-v6-transformers-main-design-only.json`](configs/qwen3.8-27b-reconstruction-v6-transformers-main-design-only.json).
V6 keeps the layer-33 asset and all reconstruction gates unchanged; it is a
new packet identity and does not reuse V5 scientific artifacts.

The V6 processor attempt stopped before model construction because its
multimodal image processor required Pillow and Torchvision.  V7 is a fresh
runtime continuation that pins Torchvision `0.21.0` with Torch `2.6.0` and
Pillow `11.3.0`; its design-only boundary is
[`configs/qwen3.8-27b-reconstruction-v7-vision-runtime-design-only.json`](configs/qwen3.8-27b-reconstruction-v7-vision-runtime-design-only.json).
The V8 adapter correction added all-token capture and loaded the model on an
H100, but both frozen layer-33 hook candidates failed the unchanged fit
reconstruction gate.  V8 is closed without retuning and produced no
scientific result; see [`docs/QWEN38_INTERNAL_V8_RESULT.md`](docs/QWEN38_INTERNAL_V8_RESULT.md).

The design-only Qwen3.8 compatibility audit is recorded in
[`docs/QWEN38_V9_COMPATIBILITY_AUDIT.md`](docs/QWEN38_V9_COMPATIBILITY_AUDIT.md).
It closes the Qwen3.8 path for V9 because the only identified Qwen3.8 SAE is a
third-party coding-domain asset and failed the V8 reconstruction gate. The
fresh fallback design binds the official Gemma 3 27B checkpoint to the
model-named Gemma Scope 2 layer-31 asset in
[`configs/gemma3-27b-reconstruction-v9-design-only.json`](configs/gemma3-27b-reconstruction-v9-design-only.json).
The config is design-only: its custody digests are intentionally null, real
execution is disabled, and Weco/causal/continual lanes remain closed until a
fresh custody-bound reconstruction qualification passes.

The V9 review packet is frozen at
[`docs/review/gemma3-v9-review-packet.json`](docs/review/gemma3-v9-review-packet.json)
with packet digest
`9956c1bc697aa8730046fb99a6d0975bd940b56da4068e100e159e85ccc608a0`. It is a
request for review, not an acceptance receipt. An independent reviewer must
return a packet-bound Ed25519 `ACCEPT` before any externally validated causal
assessment; a local signature cannot satisfy that requirement.

The design was then bound to a fresh owner-only custody root for one internal
real-model qualification. The corrected adapter uses the official affine
JumpReLU equations and captures the paired layer-31 input/output required by
the transcoder. The H100 run passed parity, repeatability, no-op, intervention
reach, reconstruction, feature stability, event replay, and custody gates;
the aggregate record is [`docs/GEMMA3_INTERNAL_V9_RESULT.md`](docs/GEMMA3_INTERNAL_V9_RESULT.md).
The pooled FVU was `0.014107974246144295` and the maximum row NMSE was
`0.01468808576464653`, both below the fixed `0.05` limits. The prompt roster
was deterministic and internal, so this remains qualification evidence rather
than a held-out scientific assessment.

After reconstruction passed, Weco ran only on the isolated synthetic metric
benchmark. Its lineage and independently checked `2.479196x` throughput ratio
are recorded in [`docs/GEMMA3_V9_WECO_RESULT.md`](docs/GEMMA3_V9_WECO_RESULT.md).
The optimized snapshot was not applied to production code. Causal
intervention, scrubbing, and continual-learning experiments remain closed
until an independent reviewer accepts a packet-bound assessment configuration
and its fresh held-out corpus.

The next scientific packet is the Gemma 3 V10 held-out causal and retention
design in
[`configs/gemma3-27b-heldout-causal-retention-v10-design-only.json`](configs/gemma3-27b-heldout-causal-retention-v10-design-only.json),
with its protocol described in
[`docs/GEMMA3_V10_ASSESSMENT_PLAN.md`](docs/GEMMA3_V10_ASSESSMENT_PLAN.md).
The independent-review request is frozen at
[`docs/review/gemma3-v10-review-packet.json`](docs/review/gemma3-v10-review-packet.json)
with packet digest
`866ac7b5470afcae6d34802ee070a8037094a56c10872b4d6cbaf3eba4dc586b`.
V10 requires a new external corpus, exact base and post-update checkpoint
manifests, a prediction lock, and a reviewer-controlled Ed25519 `ACCEPT`.
The reviewer handoff is [`docs/review/GEMMA3_V10_REVIEW_HANDOFF.md`](docs/review/GEMMA3_V10_REVIEW_HANDOFF.md).
After custody arrives, [`scripts/assemble_gemma3_v10_packet.py`](scripts/assemble_gemma3_v10_packet.py)
will bind the fresh inputs and prediction lock; [`scripts/check_gemma3_v10_gate.py`](scripts/check_gemma3_v10_gate.py)
will fail closed until the exact packet has an external Ed25519 `ACCEPT`.
The local frozen controller smoke test is recorded in
[`docs/GEMMA3_V10_SYNTHETIC_READINESS.md`](docs/GEMMA3_V10_SYNTHETIC_READINESS.md);
it correctly returned `NO_CANDIDATE` because the toy causal monitor did not
beat EWC on prior retention. No V10 model execution or assessment effect has
run.
