# ROGII - Wellbore Geology Prediction - Submissions log

Real scores from `kaggle competitions submissions -c rogii-wellbore-geology-prediction`.
Lower RMSE is better. Public LB leader for reference: ~4.86 (as of 2026-07-13).

| Date | Kaggle ref | Approach | Local CV RMSE | Public LB | Notes |
|---|---|---|---|---|---|
| 2026-07-13 | 54662772 | Stage 2: per-well linear `tvt ~ MD + Z` | 67.09 (median 33.07) | **80.534** | Pipeline-proving baseline. Extrapolates a straight line into the eval zone; breaks on faults. |
| 2026-07-13 | 54664284 | Stage 4a: global `HistGradientBoostingRegressor` (linear prior + windowed GR/typewell match + geometry as features, GroupKFold-by-well OOF) | 52.90 (773-well OOF) | **45.196** | 44% improvement over Stage 2. Public LB gain (80.53→45.2) tracked the local CV gain (67.09→52.90) in the same direction. no CV/LB divergence. Still a flat/tabular model, not a true sequence model; Stage 4b (1D CNN / real DTW) is next. |

## How each was submitted

Both went through the real Code Competition path: pushed as a Kaggle Notebook (`kaggle
kernels push`), executed on Kaggle's own infrastructure against
`/kaggle/input/competitions/rogii-wellbore-geology-prediction/`, then submitted via
`kaggle competitions submit -f submission.csv -k jjespinoza/<slug> -v <version>`. Score
read back with `kaggle competitions submissions -c rogii-wellbore-geology-prediction`. not
guessed or estimated.

- Stage 2: `notebooks/submission.ipynb`
- Stage 4a: `notebooks/submission_stage4a.ipynb`, kernel
  `jjespinoza/rogii-stage-4a-global-gradient-boosted-model`
