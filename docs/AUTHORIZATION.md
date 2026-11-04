# Owner authorization record

Source: owner chat reply "Approve, set ceiling" to the question "Approve model downloads and a USD ceiling for Phase 2?" (after the consolidation report).

The owner gave no dollar figure and delegated it. Interpretation, recorded here and binding on agents:

- **Downloads approved (Phase 2 only):** public Pythia-160m checkpoints (EleutherAI/pythia-160m, revisions step1000, step4000, step16000, step32000, step64000, step100000, step143000); WikiText-2 raw (Salesforce/wikitext, wikitext-2-raw-v1); Qwen3-0.6B/1.7B for R1 only after the R1 protocol is committed. Python packages: torch, transformers, numpy, scipy, pyarrow into an isolated venv outside the repo. Every download is recorded with file SHA-256 before use. Total budget for downloads: 10 GB (disk is 95% full).
- **Hard USD ceiling for paid compute: USD 10.00, total for this repo, USD 0.00 used.** Phase 2 uses local compute only (spend 0). No paid action before a provider/node identity and an allocation receipt are recorded. Any ceiling change needs a new owner statement.
- Ceiling on claims: `LocalDevelopmentUnreviewed`. No independent review exists or is waived by this record; nothing here raises a claim.

## Addendum 2 (2026-11-03): V54 corpus rebuild for R6
Source: owner chat reply "Approve rebuild" to the question "Approve fetching the ~38 Gutenberg books to rebuild the V54 corpus?".
- **Approved:** fetch only the frozen book list in `imports/astral-neuron/tools/astral-neuron-level-causal-lane-v54/books_v54.py` from gutenberg.org (the script's own limits: at most 60 documents, 300 MB, 8 MB per file) and run the unchanged `build_corpus_v54.py`. Install `spacy==3.8.16` and `en_core_web_sm==3.8.0` (the exact parser recorded in the V54 manifest) into an isolated venv outside this repo. No other download. (The venv was created at `~/.mech-interp-private/r6/venv` just before this record was written, within this approval.)
- **Verification before use:** each fetched book's SHA-256 is compared with `runs/v54-run/corpus-public.json`; the rebuilt manifest digest is compared with `59c514f0c5a89a1cf046f94a64e4abbe1254158bf3e992f37fbcf5e754a33221`. A mismatch is reported as such and the corpus is NOT treated as the V54 corpus (R6 would then need a protocol note before any run).
- **Storage:** raw text and the rebuilt `corpus.json` live outside the repo in an owner-only (0700) directory; only digests and public aggregates are committed.
- No spend; no change to the USD 10.00 ceiling or to the claim ceiling.

## Addendum 3 (2026-11-03): R6 on Modal (remote GPU), spend allocation
Source: owner chat instruction "do so using modal cli to launch a remote run" (in reply to the question whether to start the multi-hour local R6 run).
- **Provider/node identity:** Modal (modal.com), CLI 1.6.3, profile `epistetechnician`, GPU `A10G` (24 GB), one container, region chosen by Modal.
- **Allocation:** at most USD 7.00 of the USD 10.00 ceiling (A10G list price about USD 1.10/h x 6 h hard function timeout = about USD 6.60). Modal's own billing is the receipt; the actual figure is recorded in the R6 result note. No prior use of this ceiling; the ceiling itself is unchanged. A timeout is reported `NotCompleted` (budget), not retried or extended without a new owner statement.
- **Data sent to Modal:** the Qwen3-1.7B-Base files from the local HF cache (SHA-256 recorded locally first, verified again in the container), the rebuilt V54 `corpus.json` (public-domain Gutenberg-derived items; manifest digest verified in the container), and the committed code. Stored on a private Modal Volume `r6-data`. No other download; the container runs with `HF_HUB_OFFLINE=1`. Image packages: torch==2.12.0, transformers==5.8.1, numpy (same torch/transformers versions as the local environment).
- **Protocol deviation, recorded before the run:** device is CUDA float32 (TF32 disabled) instead of Apple MPS float32. Frozen config, layers, draws, gate and decision rule are unchanged (digest guard enforced). Results are reported with `device: cuda`; no cross-device selection stability is claimed.
- Ceiling on claims: `LocalDevelopmentUnreviewed`. Nothing here raises a claim.

## Addendum 4 (2026-11-03): R6 rerun on fixed code, spend allocation
Source: owner chat reply "okay yes go ahead with 1" to the question "Approve up to USD 6.60 for one rerun on the fixed code (2394bfc), exceeding the USD 7.00 Addendum 3 allocation but within the USD 10.00 ceiling?".
- **Provider/node identity:** unchanged from Addendum 3 (Modal, CLI 1.6.3, profile `epistetechnician`, A10G, one container).
- **Allocation:** at most USD 6.60 for exactly one rerun (6 h hard function timeout), on top of roughly USD 4 already spent on the crashed run and the duplicate run (both on pre-fix code). Cumulative R6 spend may therefore reach about USD 10.60 on list-price estimates; Modal's actual billing is the receipt and the sum is checked before launch. If actual prior spend plus 6.60 would exceed the USD 10.00 total ceiling, the run is NOT launched without a new owner statement. A timeout or crash is reported `NotCompleted`, not retried.
- Everything else (data sent, device deviation, frozen config, claim ceiling `LocalDevelopmentUnreviewed`) is as in Addendum 3.

## Addendum 5 (2026-11-03): total ceiling raised to USD 12.00, R6 rerun launch
Source: owner chat reply "go with 1 end to end" to the option "Raise the total ceiling to USD 12.00 and launch the one run."
- **Ceiling:** total paid-compute ceiling for this repo is now **USD 12.00** (was 10.00). Estimated prior spend about USD 5.10 (crashed run 1 about 2.10, duplicate run 2 about 3.00, list price; the Modal billing report was empty at the time). The one rerun (6 h hard timeout, at most about USD 6.60) therefore reaches at most about USD 11.70. Nothing further may be launched under this ceiling without a new owner statement.
- **Launch:** exactly one run, code commit `5e6f1f21438a984147179073bda0dc29cb463872`, same Modal provider/node identity as Addendum 3, per-stage partial checkpoints enabled. A timeout or crash is reported `NotCompleted` and not retried.
- Claim ceiling unchanged: `LocalDevelopmentUnreviewed`.
- **Launch record (2026-11-03):** a first launch (`ap-4CdMJabSsareMjR9FavYQh`, 22:29-22:31 EST, about 2 min, about USD 0.04) used a mistyped `--code-commit` and was stopped by the agent; it read no result. The single authorized run is `ap-LNyGF2MCdy8dSLqxFPZPW3`, profile `epistetechnician`, commit `5e6f1f21438a984147179073bda0dc29cb463872`, started about 22:32 EST. Note: the active Modal CLI profile had switched to `episte2001`; the run was launched with `MODAL_PROFILE=epistetechnician` as authorized in Addendum 3, no other account was used.

## Addendum 6 (2026-11-04): R6 outcome record (no new authorization)
- Run `ap-LNyGF2MCdy8dSLqxFPZPW3` ended about 22:37 EST on 2026-11-03, about 5 min after start, when Modal returned `ConflictError: workspace ac-hUFoWZVZfuVYtLXbEi4eDM is disabled`. Only candidate selection completed. No result or `.partial` exists on volume `r6-data`; no containers remain. Extra spend about USD 0.10 (estimate; billing report not yet available).
- Outcome: R6 is `NotCompleted`. No relaunch, retry, or other account/provider is authorized by this addendum. Any further run needs a new owner statement (and the workspace issue resolved).

## Addendum 7 (2026-11-04): R6 relaunch on the `episte2001` Modal workspace
Source: owner chat instruction "switch to episte2001 workspace that we have here in our cli because we have credit there" (in reply to the options after the `epistetechnician` workspace was disabled).
- **Provider/node identity:** Modal, CLI 1.6.3, profile `episte2001` (explicitly authorized by this statement; supersedes the `epistetechnician`-only wording of Addendum 3), GPU `A10G`, one container, new private Volume `r6-data` in that workspace.
- **Interpretation (agent's, recorded):** the owner gave no new dollar figure, so the USD 12.00 total ceiling of Addendum 5 stands. Estimated prior spend about USD 5.20 (about 5.10 plus about 0.10 from the aborted run), so one run (at most about USD 6.60) reaches at most about USD 11.80. Exactly one run, code commit `5e6f1f21438a984147179073bda0dc29cb463872`, per-stage checkpoints on. Timeout or crash is `NotCompleted`; no retry without a new owner statement.
- **Data sent:** same staged files as Addendum 3 (model files, hashes.json, corpus.json), now also uploaded to the `episte2001` workspace volume. Claim ceiling `LocalDevelopmentUnreviewed`.

## Addendum 8 (2026-11-04): R6 launch record (no new authorization)
The single run authorized by Addendum 7 started 2026-11-04 17:21 EST: app `ap-SSFI4E2C2e3oeMQDfSIPcY`, profile `episte2001`, A10G, `--code-commit 5e6f1f21438a984147179073bda0dc29cb463872` (no code differences from HEAD in `experiments/r6`, `imports`, `tests/r6`). Model hashes, corpus digest and candidate selection (96) passed in the container. Results land in volume `r6-data` at `/out/r6_result.json` (partials per stage). No relaunch authorized.
