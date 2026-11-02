# mech-interp-unified

One home for the mechanistic-interpretability research previously spread across
`composed-zk-benchmark-os` (Astral neuron lane V49-V62, neuron-gate Lean proofs),
`theseus-lab` (mechanism replacement/splitting) and `interpretability-lab-sandbox`
(SAE/transcoder harness). Goal: reproduce, then improve on, the claims in
Goodfire's essay "Neural networks are already clean" (https://already-clean.goodfire.com/).

## Layout

| Path | Content |
|---|---|
| `imports/` | Byte-exact copies of the source research, with `imports/MANIFEST.json` (SHA-256, source repo, source path, source HEAD, dirty files). Never edited. |
| `docs/INDEX.md` | Synthesis: every result, its status, its ceiling, what it does not show. |
| `docs/REPRODUCTION_PLAN.md` | Claim-by-claim reproduction and improvement protocol, gated. |
| `scripts/verify_imports.py` | Re-hashes every imported file against the manifest. |
| `AGENTS.md` | Governance: claim ceilings, preregistration, spend gates. |

## Verify

```sh
python3 -I scripts/verify_imports.py
```

## Status

Consolidation done. Phase 2 (local, USD 0) started: R5 negative (audit blind to planted superposition), R4 passes its frozen rule but is confounded; R5b audit not validated; R4b/R4c unresolved, R4d weak support for temporal clustering of token births (post-hoc, one model); see `docs/protocols/` results. Authorization: `docs/AUTHORIZATION.md` (paid ceiling USD 10.00, none used). Nothing here is independently reviewed.
