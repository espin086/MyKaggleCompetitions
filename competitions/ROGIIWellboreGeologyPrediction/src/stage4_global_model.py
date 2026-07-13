"""Stage 4: global gradient-boosted model across all wells.

Stage 2 (per-well linear tvt~MD+Z) and Stage 3 (per-well typewell/GR matching,
two variants) are all PER-WELL heuristics with no learning across wells. Stage
3 in particular underperformed the plain linear baseline on a 50-well sample
(74.72 and 75.06 vs 70.23 RMSE) - a hand-tuned heuristic gate couldn't tell
good GR matches from noise, and a diagnostic showed typewell-fit quality is
fairly uniform across wells (median residual ~7.7 GR units, not bimodal), so a
simple per-well trust/no-trust gate has no signal to key off either.

This stage trains ONE model across every well's real evaluation zone (each
well already carries a real, labeled eval zone - TVT_input is NaN there but
the true TVT is known - so no synthetic masking is needed, exactly like the
earlier stages). Features include the Stage 2 linear prior and both Stage 3
match candidates as INPUTS, not hard gates - a gradient-boosted model can learn
per-feature, per-region how much to trust each signal, which a hand-written
heuristic threshold could not.

Validated with GroupKFold by well (never train and score on the same well) so
the RMSE is a fair, leak-free estimate directly comparable to Stage 2's 67.09.
"""

import glob
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

import config
import stage3_windowed_match as s3w

RANDOM_STATE = config.RANDOM_STATE


def list_wells(split_dir):
    files = glob.glob(os.path.join(split_dir, "*__horizontal_well.csv"))
    return sorted(os.path.basename(f).split("__")[0] for f in files)


def build_well_features(well, split_dir):
    """One row per eval-zone position for this well, with every signal Stages
    2-3 produce as a feature, plus geometry. Target is the true TVT (train only
    - this function assumes train, where TVT is present for scoring)."""
    hz = pd.read_csv(os.path.join(split_dir, f"{well}__horizontal_well.csv")).reset_index(drop=True)
    tw_path = os.path.join(split_dir, f"{well}__typewell.csv")
    tw = pd.read_csv(tw_path).dropna(subset=["TVT", "GR"]).sort_values("TVT")

    known = hz[hz["TVT_input"].notna()]
    eval_rows = hz[hz["TVT_input"].isna()]
    if len(eval_rows) == 0:
        return None

    linear_prior = s3w.linear_prior_predict(known, eval_rows)

    if len(tw) >= s3w.HALF_WINDOW * 2 + 1:
        tw_tvt = tw["TVT"].to_numpy()
        tw_gr = tw["GR"].to_numpy()
        eval_gr = eval_rows["GR"].to_numpy()
        windowed_match = s3w.windowed_shape_match(linear_prior, eval_gr, tw_tvt, tw_gr)
    else:
        windowed_match = linear_prior.copy()

    known_md_max = known["MD"].max() if len(known) else eval_rows["MD"].min()
    n_eval = len(eval_rows)

    df = pd.DataFrame({
        "well": well,
        "MD": eval_rows["MD"].to_numpy(),
        "X": eval_rows["X"].to_numpy(),
        "Y": eval_rows["Y"].to_numpy(),
        "Z": eval_rows["Z"].to_numpy(),
        "GR": eval_rows["GR"].to_numpy(),
        "linear_prior": linear_prior,
        "windowed_match": windowed_match,
        "match_minus_prior": windowed_match - linear_prior,
        "dist_from_known_boundary": eval_rows["MD"].to_numpy() - known_md_max,
        "eval_zone_frac": (np.arange(n_eval) + 1) / n_eval,
        "known_zone_rows": len(known),
    })

    if "TVT" in hz.columns:
        df["target"] = eval_rows["TVT"].to_numpy()

    return df


def build_dataset(split_dir, wells):
    frames = []
    for well in wells:
        f = build_well_features(well, split_dir)
        if f is not None:
            frames.append(f)
    return pd.concat(frames, ignore_index=True)


FEATURE_COLS = [
    "MD", "X", "Y", "Z", "GR", "linear_prior", "windowed_match",
    "match_minus_prior", "dist_from_known_boundary", "eval_zone_frac",
    "known_zone_rows",
]


def cross_validated_rmse(data, n_splits=5):
    X = data[FEATURE_COLS]
    y = data["target"].to_numpy()
    groups = data["well"].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = np.full(len(data), np.nan)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
        model.fit(X.iloc[train_idx], y[train_idx])
        oof_pred[val_idx] = model.predict(X.iloc[val_idx])
        fold_rmse = np.sqrt(np.mean((y[val_idx] - oof_pred[val_idx]) ** 2))
        print(f"  fold {fold}: {len(val_idx)} rows, RMSE {fold_rmse:.4f}")

    overall_rmse = float(np.sqrt(np.mean((y - oof_pred) ** 2)))
    return overall_rmse, oof_pred


if __name__ == "__main__":
    t0 = time.time()
    wells = list_wells(config.TRAIN_DIR)
    print(f"Building features for {len(wells)} wells...")
    data = build_dataset(config.TRAIN_DIR, wells)
    print(f"Dataset: {data.shape}, built in {time.time()-t0:.1f}s")

    print("\nBaseline features alone (no learning), for reference:")
    print(f"  linear_prior RMSE:   {np.sqrt(np.mean((data.target - data.linear_prior)**2)):.4f}")
    print(f"  windowed_match RMSE: {np.sqrt(np.mean((data.target - data.windowed_match)**2)):.4f}")

    print("\nStage 4: GroupKFold (5-fold, by well) HistGradientBoostingRegressor")
    rmse, oof = cross_validated_rmse(data)
    print(f"\nStage 4 overall out-of-fold RMSE: {rmse:.4f}")
    print("Stage 2 baseline for reference: 67.09 (median 33.07)")
    print("Public LB leader for reference: ~4.86")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
