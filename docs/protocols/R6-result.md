# R6 result: Replicates (owner-unreviewed, single run)

Qwen3-1.7B-Base, layers 8/15/22, k=32, swap size 8, 1000 matched draws, instrument gate v3. Claim ceiling `LocalDevelopmentUnreviewed`. Raw result: `R6-result.json` (frozen6 digest `42ec6cc0...`, candidates digest `b4e00aa1...`).

- Gate: InstrumentValid (computed on A10G in the Modal partial, reused; signal/noise 16.97).
- Selection repeat Jaccard 1.0.
- All three tests supported, one-sided p = 0.000999 (the floor for 1000 draws):
  - T0 all-position swap: selected 1.437 vs control max 0.044.
  - L last-position swap: selected 1.158 vs control max 0.038.
  - P donor patch: selected 1.550 vs control max 0.044 (same-class donor diagnostic 0.100).
- Outcome per the frozen rule: **Replicates**.

Caveats: single run, single second model, devices differ between gate (A10G) and later stages (H100, CUDA fp32, TF32 off); no cross-device stability claimed; this is not independent verification. Spend and deviations: AUTHORIZATION.md Addendum 10 outcome.
