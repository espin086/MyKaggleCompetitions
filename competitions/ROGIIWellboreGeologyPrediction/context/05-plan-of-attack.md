# Plan of attack. ROGII Wellbore Geology Prediction

Staged plan. Each stage ends with a verifiable check (per the verify-with-real-data rule).
Ship the pipeline before the model. a submitting baseline beats a perfect model that can't
submit. Deadline 2026-08-05; ~5 submissions/day.

## Stage 0. Enter + load (blocked on Join)
1. Accept rules / Join on the website (manual, one click). Download + unzip data (README).
2. Write a per-well loader in `src/`: given a well hash, return the horizontal_well df + the
   typewell df. Enumerate `data/train/*__horizontal_well.csv` → ~700 well hashes; `data/test/`
   → the visible examples (real test swapped in on rerun).
   - **Verify:** counts of train/test wells; a well's df has expected columns; the eval-zone
     mask `TVT_input.isna()` selects a contiguous span, not scattered rows.

## Stage 1. EDA (answers that shape the model)
- Distribution of `tvt`; per-well range; how big is the eval zone vs the known zone.
- Is `tvt` smooth in `MD`? Where are the jumps (faults)? Plot a few wells + their `.png`.
- Correlation of `tvt` with `Z`, `MD`, `GR`. Fit per-well `tvt ~ MD + Z`. how good is the
  linear prior alone (R², residual std)?
- Typewell: does lateral `GR` visibly match typewell `GR` when shifted by `tvt`? This validates
  the whole log-matching premise.
  - **Verify:** a notebook/figure per question saved to `context/eda/` or `src/`.

## Stage 2. Submitting baseline (lock the pipeline)
- Per-well linear prior: fit `tvt = a·MD + b·Z + c` on the known-zone rows, predict the eval
  zone. Assemble `submission.csv` with `id = {hash}_{row_index}`.
- Build the **Kaggle notebook** that reproduces this offline (internet off) and writes
  `submission.csv`. This is the real submission artifact. get it green early.
  - **Verify:** submission row count matches sample_submission; first Kaggle submission returns
    a public score; log it in `SUBMISSIONS.md`. This proves the end-to-end path works.
- **Local CV that doesn't lie:** GroupKFold by well hash; within held-out wells, mask a
  contiguous span to imitate the eval zone. Track local RMSE vs public LB gap.

## Stage 3. Signal matching (typewell alignment). TESTED, underperformed; feed as feature not gate

Two heuristic variants were tested against the real per-well eval zones (same 773-well
harness as Stage 2), both **worse than the Stage 2 linear prior**:

| Approach | Local RMSE (50-well sample) | vs Stage 2 (70.23 same sample) |
|---|---|---|
| Pointwise nearest-GR match within a ±100ft window of the linear prior (`src/stage3_gr_match.py`) | 74.72 | worse |
| Windowed shape-match (±5-row window, SSE) + per-well confidence gate (`src/stage3_windowed_match.py`) | 75.06 | worse |

**Why:** diagnosed on well `000d7d20`. its true eval-zone span is only 15 ft, but the
typewell is noisy (GR std 26, range 28-158) over ~650 ft at 0.5 ft resolution, and the
linear prior was *already* tight there (RMSE 10). A wide/noisy search window gives the
matcher enough freedom to latch onto spurious GR coincidences instead of real structure.
A follow-up diagnostic (median |lateral GR − typewell GR at known TVT| across 50 wells)
showed typewell-fit quality is fairly **uniform** (~7.7 GR units, not bimodal). so a
simple per-well trust/no-trust gate has no signal to key off either. Two informed attempts
without a reinforcing trend; per `verify-with-real-data.md` this stopped the heuristic-gate
line rather than burn a third blind parameter guess.

**Verdict, revised:** don't hand-tune a GR-match heuristic further. The signal is still
probably useful, but only as an *input feature* to a learned model (Stage 4) that can weigh
it per-region rather than a fixed threshold. see Stage 4's `windowed_match` /
`match_minus_prior` features, which feed the same matcher's output into the boosted model
instead of gating it by hand. Real DTW alignment (proper dynamic time warping over the full
known-zone sequence, not a fixed-radius local window) remains untried and is the more
faithful reading of the hengck23 DTW thread. worth a real attempt if Stage 4/5 plateau.
Watch for shared/duplicate typewells across wells (RMorrison thread) if revisiting this.

## Stage 4a. Global gradient-boosted model. TESTED, real improvement (67.09 → 52.90)

Before the 1D-CNN lift, tried the simpler "let a learned model combine the existing signals"
step: one `HistGradientBoostingRegressor` trained across **all** wells' real eval zones
pooled together (not per-well), with `linear_prior`, `windowed_match`,
`match_minus_prior`, geometry (`MD/X/Y/Z`), `GR`, `dist_from_known_boundary`,
`eval_zone_frac`, and `known_zone_rows` as features (`src/stage4_global_model.py`).
Validated with 5-fold **GroupKFold by well** (never train and score on the same well). directly comparable to Stage 2's 67.09 since both use the full 773-well real-eval-zone set.

| Signal | RMSE (773 wells, full set) |
|---|---|
| `linear_prior` alone (Stage 2) | 67.09 |
| `windowed_match` alone (Stage 3, confirms the 50-well finding at full scale) | 71.98 |
| **Stage 4a: HistGradientBoostingRegressor combining both + features (OOF)** | **52.90** |

**Why it worked where Stage 3's hand-tuned gate didn't:** the model learns *when* to trust
the GR-match signal (and by how much) per region of feature space, rather than a single
fixed threshold. Confirms the Stage 3 verdict above. the GR-match signal had real
information, it just needed a learner instead of a heuristic to extract it.

**Caveat:** per-fold RMSE ranged 46.7. 63.8. real well-to-well difficulty variance
(structural complexity), not fold noise; the closest-to-plausible test wells still likely
score somewhere in that range, still far from the ~4.86 LB leader. Runtime: ~11 min for the
full 773-well feature build + 5-fold train (feature build dominates. the windowed-match
step is O(rows) with per-row Python overhead; vectorizing it would speed later iterations).

## Stage 4b. Sequence model (the needle-mover, still ahead)
- 1D CNN (and/or temporal model) over `[GR, MD, Z, X, Y, TVT_input, dip features]` predicting
  `tvt`. or the **residual** off the Stage-2/4a prediction (usually more stable). This is the
  log-inversion approach the leaders use (hengck23 MTP; nvidia-kaggle teardown) and is what
  closes the remaining ~48-point gap to the LB leader. Stage 4a's gradient boosting is still
  a flat/tabular model over per-row features, not a true sequence model.
- If using pretrained weights, upload them as a Kaggle Dataset (internet-off at submit).
  - **Verify:** GroupKFold RMSE beats Stage 4a's 52.90; a submission confirms the LB moves the
    same direction (guard against CV/LB divergence).

## Stage 5. Ensemble + finalize
- Blend physics prior + DTW match + CNN. Weight by held-out RMSE.
- Uncertainty estimation (per-prediction confidence). cheap wins and rewarded by the Working
  Note criteria even if the writeup award has passed; it also flags where to trust the blend.
- Pick **2 diverse final submissions** (e.g. best-CV CNN blend + a robust physics-heavy one),
  not two near-identical ones, to hedge public↔private LB shift.

## Tooling note. reuse the repo's kaggle-ml-loop where it fits
The `kaggle-ml-loop` skill automates the tabular loop (EDA → recipes → MLflow → champion).
It fits Stages 1. 2 (EDA + the linear/tabular baseline) but **not** the sequence/DTW stages,
which are custom. Use it to bootstrap the baseline fast, then hand-build Stages 3. 5.

## Risks / watchlist
- **CV that leaks:** random-row CV will look great and fail on LB. Group by well; mask spans.
- **Visible-test overfit:** the visible `test/` is copied from train. don't tune to it.
- **Private rescore:** a prior rescore happened; trust GroupKFold over early public LB.
- **Notebook runtime:** ≤ 9h, internet off. Keep inference lean; preload weights as a Dataset.
