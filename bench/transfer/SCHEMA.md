# F1 mechanism-transfer bench: item schema (Phase 1)

Ceiling for everything in `bench/transfer/`: `LocalDevelopmentUnreviewed`
(items from V49/V53/V54 inherit `OwnerSelfAttestedLocalDevelopment` only as a
property of their *sources*; this bench is not independently reviewed).
Machine-readable schema: [`item.schema.json`](item.schema.json) (JSON Schema
2020-12). Labels are produced by the rules in [`LABEL_RULES.md`](LABEL_RULES.md),
which were committed before any label existed.

## What one item is

One item = **one small-scale causal claim C**, established in a **base run B**
under its own frozen rule, paired with **one transfer arm T** that re-ran the
same frozen procedure with at least one of these axes changed:

| axis | meaning |
|---|---|
| `scale` | parameter count of the subject model (same family) |
| `device` | hardware/runtime (H100 CUDA, Apple MPS, CPU, A10G, Cloud Run CPU) |
| `dtype` | numeric precision |
| `checkpoint` | different weights revision or training checkpoint |
| `model_family` | different architecture family |
| `corpus` | different text, templates or adaptation data |
| `task` | different task construction using the same claimed carrier |

`scale`, `device`/`dtype`, `checkpoint` are the primary axes; `model_family`,
`corpus`, `task` are recorded because the sources vary them and usually change
several axes at once. Every axis that changed is listed in `axes_changed`;
anything else that also moved and is not an axis (for example adaptation prompt
count) is listed in `confounds`. Items with several changed axes are labelled
exactly once; the bench does not attribute a failure to one axis.

An item where **no transfer arm exists on disk** is still an item (label
`not_tested`, reason `no_arm_run`): it records that a supported claim has never
been checked under a change of scale, device or checkpoint.

## Fields

| field | type | meaning |
|---|---|---|
| `id` | string | stable slug, `^[A-Za-z0-9][A-Za-z0-9._-]*$`. Split assignment hashes this string. |
| `cluster_id` | string | experimental thread; items in a cluster share base runs, sources or a code path and are **not independent**. |
| `claim.text` | string | the claim, minimal restatement of the source's own wording; carries no outcome. |
| `claim.family` | enum | `neuron_ablation_effect`, `class_swap_sufficiency`, `donor_patch_sufficiency`, `unit_selection_stability`, `affine_replacement_prediction`, `task_transfer`, `offshoot_growth`, `route_bypass`, `instrument_gate`. |
| `model`, `size`, `device`, `dtype`, `checkpoint`, `corpus`, `instrument_gate_status` | string or `"unrecorded"` | **base-run** conditions as recorded in the source. `"unrecorded"` means the source does not state it; nothing is inferred. |
| `size_b` | number or null | base parameter count in billions when the source names a single model size, else null. |
| `base` | object | `run_id`, `verdict` (source's own status word), `verdict_class` (`supported` / `not_supported` / `inconclusive`). |
| `transfer` | object | `run_id` (null if none), the same condition fields for the target arm, `size_b`, `axes_changed[]`, `confounds[]`, `arm_status` (`completed` / `completed_instrument_invalid` / `closed_before_measurement` / `inconclusive` / `not_run`), `verdict` (source status word), `instrument_gate_status`. |
| `label` | enum | `replicated`, `failed`, `not_tested`. |
| `label_rule` | string | the rule id from `LABEL_RULES.md` that fired (`L0` ... `L5`, with the source-rule id for `L2`). |
| `label_reason` | enum | `rule_met`, `rule_not_met`, `no_arm_run`, `arm_closed_before_measurement`, `arm_inconclusive_by_source_rule`, `instrument_invalid_no_rule`, `base_not_established`, `no_preregistered_rule`. |
| `label_inputs` | object | the extracted numbers/flags the rule was applied to, copied from sources. Re-derivation recomputes the label from these. |
| `evidence` | array | one entry per extracted value: `name` (the key the rule reads), `role` (`base`, `transfer`, `rule`, `absence`), `root` (`repo`, `jspace`, `zkbench`), `file`, `sha256` (of the file at build time), `kind` (`json_pointer`, `regex`, `absence_assertion`), `pointer`, `value`. |
| `ceiling` | string | `LocalDevelopmentUnreviewed` or `OwnerSelfAttestedLocalDevelopment` (source-side attestation, V49/V53/V54 only). |
| `notes` | string | caveats copied from the source (post-hoc design, confounds, caveats). Never a re-interpretation of a result into a label. |

### Evidence kinds

* `json_pointer`: RFC 6901 pointer into a JSON file; `value` is what the pointer
  returns. Re-derivable by `scripts/verify.py`.
* `regex`: a pattern that must match exactly once in a text file; `value` is the
  first capture group (or `true` when there is no group). Used where the source
  stores a result only in prose (Markdown) and no JSON exists on disk.
* `absence_assertion`: a reviewer statement that a transfer arm does not exist,
  with the directories searched listed in `pointer`. It cannot be machine-verified;
  verify only checks that the listed paths exist. It is the only evidence kind
  whose truth rests on the bench author's search.

### External roots

`root: jspace` is `/Users/shaanp/Documents/GitHub/patelshaan/jspace-pilot`;
`root: zkbench` is `/Users/shaanp/Documents/GitHub/composed-zk-benchmark-os`.
Both are read-only. Override with `F1_ROOT_JSPACE` / `F1_ROOT_ZKBENCH`. If a root
is missing, `verify` reports those items as `external_unavailable` and checks
only the stored digests; it does not fail.

## Non-independence and leakage

Items inside a cluster are correlated (shared base run, shared code path, same
author). The frozen holdout split hashes `id`, as specified, so cluster members
can land on both sides. `scripts/baselines.py` therefore also reports a
cluster-grouped sensitivity split; see `README.md`.

## Not in an item

No raw prompts, text, weights or traces (AGENTS rule 5). Only digests and
aggregates that already exist in the source repositories.
