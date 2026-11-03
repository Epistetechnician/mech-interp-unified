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
