# Gemma 3 V9 Weco engineering result

State slice: `interpretability-lab-sandbox-v1`.

Weco ran only after the Gemma V9 internal reconstruction gate passed. It
optimized the isolated synthetic metric module under
`.weco/metrics-throughput-v2/`; it did not modify production source, the
model, the feature asset, the corpus, the protocol, thresholds, or any model
result.

| Field | Value |
| --- | --- |
| Run ID | `8746fb52-8083-4acf-bd68-8cfe76b26472` |
| Steps | baseline plus one candidate step |
| Fixed objective | `throughput_ops_per_second`, maximize |
| Weco baseline | `5382.167533` |
| Weco best step | `14368.619166` |
| Independent evaluator baseline | `5331.074569` |
| Independent evaluator candidate | `13216.781294` |
| Independent evaluator ratio | `2.479196` |
| Best snapshot SHA-256 | `e34ef95342d09435f0673d2bf4ab2cca0e5360f1a8d353cfc1a216af30b10eb1` |

The hostile oracle and fixed output checks passed, including exceptions,
NaN/Infinity behavior, signed zero, hostile `Sequence` inputs, and function
signatures. The resulting snapshot is retained at
`.weco/gemma3-v9-weco/best.py` for lineage. This is engineering evidence
only; the speedup is not evidence that interpretability improves causal
validity or continual-learning retention.
