# Agent rules (mech-interp-unified)

Inherited discipline from the source repositories, restated for this repo.

1. `imports/` is immutable. Changes go in new files; `scripts/verify_imports.py` must pass.
2. Every experiment: protocol document committed before any data is read or model is run
   (hypothesis, estimand, matched-random control, thresholds, multiplicity, abort rules,
   spend cap). Thresholds never change after data is seen; a failed gate closes the run
   without retuning.
3. Claim ceilings: `LocalDevelopmentUnreviewed` by default. Owner self-attestation may
   reach `OwnerSelfAttestedLocalDevelopment`, always labelled "owner self-attested, NOT
   independent". No independent, general, mechanism, circuit, proof, SOTA, benchmark or
   production claim without genuine independent review.
4. Spend/network: no model or dataset download, no paid compute, no provider call
   without an owner-stated USD ceiling in chat, a recorded provider/node identity and an
   allocation receipt before any paid action. Past ceilings and spend are recorded in
   `docs/AUTHORIZATION.md` (latest: Addendum 10, USD 35.00 total for the R6 run only,
   about 25.2 spent). No ceiling is granted for new work.
5. Raw traces stay outside the repo in an owner-only (`0700`) root; commit digests and
   aggregates only. Delete raw traces within 72 hours.
6. Instrument validity: every effect claim passes a null-calibrated known-bad control and
   a matched-random control (>=1000 draws) before effects are read.
7. Report negative and failed results with the same prominence as positive ones.
