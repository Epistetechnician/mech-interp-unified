# Architecture

State slice: `interpretability-lab-sandbox-v1`.

The sandbox has four boundaries. `contracts.py` defines immutable identities
for observations, checkpoints, transitions, and reports. `adapters.py`
implements deterministic in-memory ports and leaves external model libraries
behind optional runtime adapters. `qualification.py` executes fixed gates in a
fixed order. `execution.py` has two explicit modes. Internal exploratory mode
requires a local authorization record, exact packet binding, and owner-only
custody. External mode additionally requires a genuinely independent
packet-bound Ed25519 receipt.

Every report includes a canonical digest and an explicit claim ceiling. The
synthetic path emits aggregate values only; it does not retain activations or
other raw payloads. A future real adapter must satisfy the same ports and must
provide its own model, feature asset, runtime, corpus, custody, and authorization
identities before it can be enabled. Internal results retain an explicit local
claim ceiling and cannot be relabelled as independent evidence.
