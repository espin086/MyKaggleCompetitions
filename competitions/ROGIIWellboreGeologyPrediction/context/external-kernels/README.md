# External public kernels. reference material

Public Kaggle Notebooks pulled via `kaggle kernels pull` for study. Per the competition's
public code-sharing rules, publicly shared code is licensed under an OSI-approved license
permitting commercial use. kept here as read-only reference, not modified in place.

## rogii-dual-pipeline-self-verifying

- Author: `lightningv08` (kernel id `lightningv08/rogii-dual-pipeline-self-verifying`)
- Pulled: 2026-07-14
- Approach: dual independent physics+ML pipelines (particle filter + beam search trackers,
  LightGBM/CatBoost stack, Ridge meta-ensemble, drift-aware post-processing), blended, then
  a **guarded physical override**: exact TVT reconstruction from formation-contact columns
  (`tvt_from_contacts`) for any test well whose ID also appears in train, applied only after
  self-verifying the reconstruction against that well's own known prefix at runtime.

Ideas extracted for our own pipeline (see `context/05-plan-of-attack.md` Stage 7):

1. **Guarded physical override**. cheap, safe (self-verifying, never applied blind), and
   potentially high-value if the real hidden test set reuses any train well IDs.
2. **Cross-well spatial priors** (`FormationPlaneKNN`). impute a well's formation-contact
   depths from nearby wells' known contacts via inverse-distance-weighted local plane fit in
   (X, Y). Something our own pipeline has never used. every stage so far treats each well
   in complete isolation.
3. **Multi-scale GR normalized cross-correlation**. a more robust relative of our Stage 3
   windowed shape-match, worth a look if Stage 7's simpler additions don't move the needle.

Not reused: the full particle-filter/beam-search tracker stack (JIT-compiled, hundreds of
lines, 128-seed ensembles). out of scope for the time available; the guarded override and
spatial-prior ideas are the tractable, high-leverage extractions.
