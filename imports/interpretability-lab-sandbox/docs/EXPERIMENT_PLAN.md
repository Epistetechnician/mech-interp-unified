# End-to-end experiment plan

State slice: `interpretability-lab-sandbox-v1`.

The real program is staged so a failed gate closes the slice without retuning:

1. Native/instrumented parity and deterministic repeatability.
2. Exact no-op and nonzero intervention reach.
3. Feature reconstruction and cross-checkpoint stability.
4. Held-out causal scrubbing with exact controls.
5. Continual-learning comparison against SGD, replay, EWC, behavioral-only,
   and causal-monitor controllers with equal data and measured overhead.
6. Independent aggregate reproduction and raw-trace deletion within 72 hours.

The sandbox currently executes only a synthetic analogue of these stages. An
internal exploratory run may execute a real adapter after a local
authorization record freezes the model/runtime/asset/corpus, controls,
estimand, thresholds, splits, prediction lock, and bounded claim ceiling. That
record is an operator decision, not independent review or proof. External
validation still requires a separate packet-bound Ed25519 receipt. No
synthetic result can satisfy either real-run requirement.
