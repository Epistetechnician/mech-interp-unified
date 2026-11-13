# F1 mechanism-transfer bench, Phase 1

Small-scale causal claims from this repo's runs and two read-only sibling repositories, each labelled
`replicated` / `failed` / `not_tested` by the **source protocol's own frozen rule** after a change of
scale, device, checkpoint, model family, corpus or task. Ceiling `LocalDevelopmentUnreviewed`. Owner-run,
unreviewed, not independent. No model was run and nothing was spent for this bench.

* `SCHEMA.md`, `item.schema.json`: item format.
* `LABEL_RULES.md`: preregistration (committed before any label). Section 0 discloses that the author had read most outcomes beforehand.
* `items.jsonl`: 36 items, each with file + pointer evidence and digests.
* `EXCLUDED.md`: what was surveyed and not used.
* `splits/split.json`: frozen holdout (hash of item id, `f1/split.py`; salt `f1-transfer-bench-v1-split`, 1 in 4).
* `results/BASELINES.md`: majority, metadata rules; LLM predictor stub.
* `EXTENSION_PROTOCOL.md`: preregistered plan for runs that would add items. Not run.

```bash
python3 bench/transfer/scripts/verify.py          # re-derives labels and split from sources
python3 -m pytest bench/transfer/tests -q
```

`verify.py` needs `jsonschema`; jspace items need `/Users/shaanp/Documents/GitHub/patelshaan/jspace-pilot` (override `F1_ROOT_JSPACE`).

## Counts (36 items)

See `results/BASELINES.md` and the commit message for the numbers; the generator is `scripts/build_items.py`.

## Read this before using the numbers

* **Small and clustered.** 36 items in 7 clusters (Astral ablation, Astral swap, V49, R6, R4, theseus, jspace). Items in a cluster share base runs and code. The hash-of-id split puts clusters on both sides, as specified; the leave-one-cluster-out majority number is the sensitivity check.
* **Mostly Pythia/Qwen agreement and one affine-projection method.** No second interpretability method family beyond those, and the jspace items are all `not_tested`.
* **Labels are rule outputs, not truth about mechanism.** A `replicated` item means the source's frozen rule was met in one run, with the confounds listed. Several arms change more than one axis. Disclosed limits (post-hoc designs, V49 gate edited after r3, 100-draw p floors) are in each item's `notes`.
* **Metadata baselines partly recover label construction.** `no transfer arm` and `target gate invalid` are inputs to rules L0/L4; the tested-only column removes those.
* **Absence evidence** (`no_arm_run` items) rests on the author's search of listed directories.

## LLM predictor (not run)

`scripts/llm_predictor.py dry` writes label-free prompts; `score` reads predictions made elsewhere. A live call
needs a paid provider call, which `AGENTS.md` rule 4 forbids without an owner-stated USD ceiling in chat,
a recorded provider identity and an allocation receipt. None is granted, so no model was queried.
