"""Stage 5: ensemble of the three real, working signals.

Original plan called for "physics prior + DTW match + CNN" - revised given
Stage 4b's real result (public LB 72.734, worse than Stage 4a's 45.196): the
CNN drags an ensemble down rather than helping, so it's excluded by default
here. This ensembles the three signals that actually work:

1. `linear_prior` (Stage 2) - per-well straight-line extrapolation.
2. `windowed_match` (Stage 3) - typewell/GR shape-match, worse alone but
   informative as a feature (confirmed in Stage 4a).
3. `stage4a_oof_pred` - Stage 4a's HistGradientBoostingRegressor, using
   properly out-of-fold predictions (never trained on the row it predicts) so
   stacking a meta-model on top doesn't leak.

A simple non-negative least-squares blend (not another full model) is the
meta-learner - with only 3 base signals there's little to gain from anything
fancier, and it stays interpretable (the weights ARE the "how much to trust
each signal" answer Stage 3/4a's diagnostics were chasing).
"""

import glob
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import config
import stage4_global_model as s4a

RANDOM_STATE = config.RANDOM_STATE
BASE_SIGNAL_COLS = ["linear_prior", "windowed_match", "stage4a_oof_pred"]


def list_wells(split_dir):
    files = glob.glob(os.path.join(split_dir, "*__horizontal_well.csv"))
    return sorted(os.path.basename(f).split("__")[0] for f in files)


def add_stage4a_oof_predictions(dataset_df, n_splits=5):
    """Leak-free Stage 4a predictions for every row: each row's prediction
    comes from a model trained on the OTHER folds' wells only."""
    X = dataset_df[s4a.FEATURE_COLS]
    y = dataset_df["target"].to_numpy()
    groups = dataset_df["well"].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = np.full(len(dataset_df), np.nan)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
        model.fit(X.iloc[train_idx], y[train_idx])
        oof_pred[val_idx] = model.predict(X.iloc[val_idx])
        fold_rmse = np.sqrt(np.mean((y[val_idx] - oof_pred[val_idx]) ** 2))
        print(f"  stage4a fold {fold}: {len(val_idx)} rows, RMSE {fold_rmse:.4f}")

    dataset_df = dataset_df.copy()
    dataset_df["stage4a_oof_pred"] = oof_pred
    return dataset_df


def fit_blend_weights(dataset_df):
    """Non-negative least squares: target ~ w1*linear_prior + w2*windowed_match
    + w3*stage4a_oof_pred, weights constrained >= 0 so a signal can be zeroed
    out but never flipped in sign (keeps the blend physically interpretable).
    """
    X = dataset_df[BASE_SIGNAL_COLS].to_numpy()
    y = dataset_df["target"].to_numpy()
    weights, residual = nnls(X, y)
    return weights


def held_out_blend_rmse(dataset_df, n_splits=5):
    """Honest estimate: fit blend weights on 4/5 of wells' OOF predictions,
    evaluate on the held-out 1/5 - separate from the OOF predictions'
    leak-free-ness, this checks the BLEND WEIGHTS themselves don't overfit."""
    groups = dataset_df["well"].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    X = dataset_df[BASE_SIGNAL_COLS].to_numpy()
    y = dataset_df["target"].to_numpy()

    all_preds = np.full(len(dataset_df), np.nan)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        weights, _ = nnls(X[train_idx], y[train_idx])
        all_preds[val_idx] = X[val_idx] @ weights

    rmse = float(np.sqrt(np.mean((all_preds - y) ** 2)))
    return rmse


if __name__ == "__main__":
    t0 = time.time()
    wells = list_wells(config.TRAIN_DIR)
    print(f"Building features for {len(wells)} wells...")
    dataset_df = s4a.build_dataset(config.TRAIN_DIR, wells)
    dataset_df = dataset_df.dropna(subset=["target"])
    print(f"Dataset: {dataset_df.shape}, built in {time.time()-t0:.1f}s")

    print("\nComputing leak-free Stage 4a OOF predictions (5-fold GroupKFold)...")
    dataset_df = add_stage4a_oof_predictions(dataset_df)

    for col in BASE_SIGNAL_COLS:
        rmse = np.sqrt(np.mean((dataset_df[col] - dataset_df["target"]) ** 2))
        print(f"  {col} alone: RMSE {rmse:.4f}")

    print("\nFitting blend weights (NNLS) on full OOF set...")
    weights = fit_blend_weights(dataset_df)
    for col, w in zip(BASE_SIGNAL_COLS, weights):
        print(f"  weight[{col}] = {w:.4f}")
    in_sample_preds = dataset_df[BASE_SIGNAL_COLS].to_numpy() @ weights
    in_sample_rmse = np.sqrt(np.mean((in_sample_preds - dataset_df["target"]) ** 2))
    print(f"In-sample blend RMSE (weights fit and evaluated on the same OOF set): {in_sample_rmse:.4f}")

    print("\nHeld-out blend RMSE (weights fit on 4/5 wells, evaluated on 1/5 - the honest number)...")
    honest_rmse = held_out_blend_rmse(dataset_df)
    print(f"Held-out blend RMSE: {honest_rmse:.4f}")

    print("\nReference:")
    print("Stage 2 (linear alone): 67.09 local, 80.534 public LB")
    print("Stage 4a (GB alone): 52.90 local, 45.196 public LB")
    print("Stage 4b (CNN alone): 70.18 local, 72.734 public LB")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
