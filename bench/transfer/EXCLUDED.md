# Sources surveyed and not turned into items

Written with the items from the same survey; each exclusion is a statement about what is on disk, not a judgement of the science.

| source | why no item |
|---|---|
| Astral V51, V52 | V51 stopped on budget with no verdict; V52 is `NoCandidate` on the same units as V53 (it differs only in the constant control), so it is not a transfer arm of anything. |
| Astral V56 | Path analysis on the V53 selection; same model, same data. No axis changed. |
| Astral V60-local / V61 cloud as bases | V61 cloud appears as the no-arm item `astral-v61-minimal-pair`; no second arm exists. |
| theseus V9-V21, V30, V31 | Single-setting FAIL/INCONCLUSIVE protocols; their own bases are not `supported`, or they share no frozen rule with a transfer arm on disk. |
| theseus V22 | `NO_ELIGIBLE_FRESH_TARGET`: stopped before any measurement, and it would be an arm of V8 only through V28, already an item. |
| theseus Band A r2 arms S1/S4/S16 | They vary intervention magnitude, not scale/device/checkpoint, and are all `FAIL`. Not added, so the bench has no entry counting them. The r2 C4 arm is included because it varies adaptation size and is an inconclusive resource stop (`theseus-v8-r2-c4`). |
| theseus r4 ridge ladder | Varies a ridge hyperparameter, all `FAIL`; not a transfer axis. |
| theseus r5 T256 | Concurrent template control inside the r5 packet; same adaptation type as r3. |
| jspace gateS/T/P/Q/R, sweeps, gateG | Efficiency tuning and a graded-clamp diagnostic; the source says they never read T_J/T_B in the pre-registered way. Graded-clamp and the damage-free bound are post hoc. |
| composed-zk-benchmark-os (V42-V48, Gemma V4, V41R27-R31) | Single-setting qualification or NoCandidate verdicts without a claim re-run under a changed scale/device/checkpoint. V41R30 re-runs one failing training cell on H100 twice (independent non-replication of a failure), but that cell is a continual-learning LoRA acquisition cell, not an interpretability causal claim, and its base is a `pass:false` run. |
| interpretability-lab-sandbox | SAE/transcoder adapters; outcomes are qualification gates, not causal claims with transfer arms. Not read in depth. |
