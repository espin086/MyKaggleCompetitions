"""Stage 5 follow-up: is the NNLS-vs-Stage4a-alone gap real, or noise?

Stage 5's held-out estimate (single 5-fold split) showed NNLS blending
beating Stage 4a alone by ~0.9 RMSE (52.90 -> 52.03), but the real Kaggle
submission came back slightly WORSE (45.997 vs 45.196) - a ~0.8 point
regression in the other direction. Two questions this script answers with
real repeated-CV data instead of guessing:

1. Is a ~1 RMSE point difference from a SINGLE held-out-by-well split even
   distinguishable from noise, given only 773 wells split 5 ways (~155/fold)?
   Repeat the held-out split 10 times with different RANDOM well-to-fold
   assignments and look at the spread.
2. Does replacing NNLS with a regularized (ridge, on STANDARDIZED features)
   blend produce a more stable held-out estimate, or does it just shrink
   toward equal weighting without helping?

Bug fixed from the first version: sklearn's GroupKFold sorts groups by size
internally before assigning folds, so it is fully DETERMINISTIC regardless
of input order - shuffling the well list before calling it does nothing (all
10 "repeats" came back bit-for-bit identical, which is how this was caught).
Fixed here with a genuinely randomized manual fold assignment. Also fixed:
ridge alphas 0.1-100 had zero effect because the three base signals are
unstandardized (~11,000 magnitude) - the penalty term was negligible next to
the Gram matrix's scale. Features are standardized before ridge here so alpha
is meaningful.

Reuses the already-computed Stage 4a OOF predictions from stage5_ensemble.py
(expensive to build - 5 GB fits, ~2 min) and only varies how the META-blend
train/val split is drawn, which is cheap.
"""

import time

import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import Ridge

import config
import stage5_ensemble as s5

BASE_SIGNAL_COLS = s5.BASE_SIGNAL_COLS
N_REPEATS = 10
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0]  # on STANDARDIZED features, so this range is meaningful
N_SPLITS = 5


def random_group_folds(all_wells, n_splits, rng):
    """Genuinely randomized well-to-fold assignment (NOT sklearn's GroupKFold,
    which is deterministic regardless of input order - see module docstring).
    Returns a dict well -> fold_id."""
    shuffled = rng.permutation(all_wells)
    fold_of_well = {}
    for i, w in enumerate(shuffled):
        fold_of_well[w] = i % n_splits
    return fold_of_well


def held_out_rmse_for_split(X, y, fold_of_well, wells_shuffled_order, well_to_rows,
                             n_splits=N_SPLITS, method="nnls", alpha=None):
    groups = np.array([w for w in wells_shuffled_order for _ in well_to_rows[w]])
    row_order = np.concatenate([well_to_rows[w] for w in wells_shuffled_order])
    Xo, yo = X[row_order], y[row_order]
    fold_ids = np.array([fold_of_well[w] for w in groups])

    blend_preds = np.full(len(yo), np.nan)
    stage4a_preds = np.full(len(yo), np.nan)

    for fold in range(n_splits):
        val_mask = fold_ids == fold
        train_mask = ~val_mask
        stage4a_preds[val_mask] = Xo[val_mask][:, BASE_SIGNAL_COLS.index("stage4a_oof_pred")]

        if method == "nnls":
            weights, _ = nnls(Xo[train_mask], yo[train_mask])
            blend_preds[val_mask] = Xo[val_mask] @ weights
        elif method == "ridge":
            mean, std = Xo[train_mask].mean(axis=0), Xo[train_mask].std(axis=0)
            std[std < 1e-6] = 1.0
            Xtr = (Xo[train_mask] - mean) / std
            Xval = (Xo[val_mask] - mean) / std
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(Xtr, yo[train_mask])
            blend_preds[val_mask] = model.predict(Xval)

    blend_rmse = float(np.sqrt(np.mean((blend_preds - yo) ** 2)))
    stage4a_rmse = float(np.sqrt(np.mean((stage4a_preds - yo) ** 2)))
    return blend_rmse, stage4a_rmse


if __name__ == "__main__":
    t0 = time.time()
    wells = s5.list_wells(config.TRAIN_DIR)
    print(f"Building features + Stage 4a OOF predictions for {len(wells)} wells...")
    dataset_df = s5.s4a.build_dataset(config.TRAIN_DIR, wells)
    dataset_df = dataset_df.dropna(subset=["target"])
    dataset_df = s5.add_stage4a_oof_predictions(dataset_df)
    print(f"Dataset ready in {time.time()-t0:.1f}s")

    X = dataset_df[BASE_SIGNAL_COLS].to_numpy()
    y = dataset_df["target"].to_numpy()
    well_to_rows = dict(dataset_df.groupby("well").indices)
    all_wells = list(well_to_rows.keys())

    print(f"\n{N_REPEATS} repeats x {{nnls, ridge x {len(RIDGE_ALPHAS)} alphas}}, "
          f"each a genuinely random {N_SPLITS}-way well split (blend weights only - "
          f"Stage 4a OOF fixed)\n")

    results = {"nnls": []}
    for alpha in RIDGE_ALPHAS:
        results[f"ridge_a{alpha}"] = []
    stage4a_baseline = []

    rng = np.random.default_rng(config.RANDOM_STATE)
    for rep in range(N_REPEATS):
        order = rng.permutation(all_wells).tolist()
        fold_of_well = random_group_folds(all_wells, N_SPLITS, rng)

        blend_rmse, stage4a_rmse = held_out_rmse_for_split(X, y, fold_of_well, order, well_to_rows, method="nnls")
        results["nnls"].append(blend_rmse)
        stage4a_baseline.append(stage4a_rmse)

        for alpha in RIDGE_ALPHAS:
            blend_rmse, _ = held_out_rmse_for_split(X, y, fold_of_well, order, well_to_rows, method="ridge", alpha=alpha)
            results[f"ridge_a{alpha}"].append(blend_rmse)

        print(f"  repeat {rep}: stage4a={stage4a_rmse:.3f}  nnls={results['nnls'][-1]:.3f}  "
              + "  ".join(f"ridge(a={a})={results[f'ridge_a{a}'][-1]:.3f}" for a in RIDGE_ALPHAS))

    print("\n=== Summary (mean +/- std over 10 repeats) ===")
    s4a_arr = np.array(stage4a_baseline)
    print(f"Stage 4a alone:  {s4a_arr.mean():.4f} +/- {s4a_arr.std():.4f}")
    for name, vals in results.items():
        arr = np.array(vals)
        diff = arr - s4a_arr  # paired: same held-out split each repeat
        se = diff.std() / np.sqrt(N_REPEATS)
        print(f"{name:18s}: {arr.mean():.4f} +/- {arr.std():.4f}   "
              f"paired diff vs stage4a: {diff.mean():+.4f} +/- {diff.std():.4f}  "
              f"(t-stat ~ {diff.mean()/se if se > 0 else float('nan'):.2f})")

    print(f"\nSingle-split result reported in the PR: stage4a=52.90, nnls=52.03 (diff -0.87)")
    print(f"Real Kaggle result: stage4a=45.196, nnls-blend=45.997 (diff +0.80, wrong direction)")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
