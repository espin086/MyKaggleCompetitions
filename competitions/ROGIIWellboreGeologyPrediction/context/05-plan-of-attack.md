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

## Stage 3. Signal matching (typewell alignment)
- DTW / cross-correlation of lateral `GR` against typewell `GR` to estimate geologic position;
  turn the alignment offset into a `tvt` estimate. (hengck23 DWT thread.)
- Add as a feature and/or a second base model. Watch for shared/duplicate typewells across
  wells (RMorrison thread). respect that in CV grouping.
  - **Verify:** does the DTW estimate reduce held-out RMSE vs Stage-2 prior? Keep only if yes.

## Stage 4. Sequence model (the needle-mover)
- 1D CNN (and/or temporal model) over `[GR, MD, Z, X, Y, TVT_input, dip features]` predicting
  `tvt`. or the **residual** off the Stage-2 linear prior (usually more stable). This is the
  log-inversion approach the leaders use (hengck23 MTP; nvidia-kaggle teardown).
- If using pretrained weights, upload them as a Kaggle Dataset (internet-off at submit).
  - **Verify:** GroupKFold RMSE beats Stages 2. 3; a submission confirms the LB moves the same
    direction (guard against CV/LB divergence).

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
