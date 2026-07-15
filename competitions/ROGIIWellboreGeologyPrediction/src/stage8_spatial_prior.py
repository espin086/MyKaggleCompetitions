"""Stage 8: cross-well spatial prior feature (FormationPlaneKNN), adapted
from lightningv08/rogii-dual-pipeline-self-verifying.

Every stage so far treats each well in complete isolation - the model only
ever sees a single well's own MD/Z/GR/log data. But wells drilled near each
other in (X, Y) tend to hit the same geological formations at similar depths
(structural continuity). This stage builds a cross-well spatial prior: for
any (X, Y) location, fit a locally-weighted plane through the K nearest
OTHER wells' known formation-contact depths (train-only columns), predict
what the formation depth "should be" at that location, then derive a
candidate TVT from it - genuinely new information no other stage has used.

Leak-free by construction: the plane fit for any well's rows always excludes
that well's own entry from the neighbor pool (`self_wid` exclusion, same as
the source kernel), so a well's own true formation depths never leak into
its own predicted feature - this holds automatically for any GroupKFold
split, no per-fold index rebuild needed (this is feature engineering from
OTHER wells' data, not a model fit on the target).
"""

import glob
import os
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import config
import stage4_global_model as s4a

RANDOM_STATE = config.RANDOM_STATE
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
PLANE_K = 10


class FormationPlaneKNN:
    """Locally-weighted plane fit of each formation's depth as a function of
    (X, Y), using the K nearest OTHER wells (inverse-distance weighted)."""

    def __init__(self, well_ids, data_dir):
        rows = []
        for wid in well_ids:
            path = os.path.join(data_dir, f"{wid}__horizontal_well.csv")
            try:
                df = pd.read_csv(path, usecols=["X", "Y"] + FORMATIONS).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            row = {"wid": wid, "x": float(df.X.median()), "y": float(df.Y.median())}
            for c in FORMATIONS:
                row[f"{c}_m"] = float(df[c].median())
            rows.append(row)

        self.df = pd.DataFrame(rows)
        self.wmap = {w: i for i, w in enumerate(self.df.wid)}
        xy = self.df[["x", "y"]].to_numpy()
        self.scale = np.where(xy.std(0) < 1e-3, 1.0, xy.std(0))
        self.tree = cKDTree(xy / self.scale)
        self.xa = self.df.x.to_numpy()
        self.ya = self.df.y.to_numpy()
        self.fa = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64)

    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        """Returns (n_query, n_formations) imputed depths at each query point,
        excluding self_wid's own entry from the neighbor pool if present."""
        q = xy_q / self.scale
        nf = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid in self.wmap:
            dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        ordr = np.argpartition(dist, min(k - 1, nf - 1), axis=1)[:, :k]
        dk = np.take_along_axis(dist, ordr, 1)
        ik = np.take_along_axis(idx, ordr, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0).astype(np.float64)

        xn, yn, fn = self.xa[ik], self.ya[ik], self.fa[ik]
        wx, wy = w * xn, w * yn
        A = np.zeros((len(q), 3, 3))
        A[:, 0, 0] = (wx * xn).sum(1); A[:, 0, 1] = (wx * yn).sum(1); A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1];       A[:, 1, 1] = (wy * yn).sum(1); A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2];       A[:, 2, 1] = A[:, 1, 2];       A[:, 2, 2] = w.sum(1)
        A[:, 0, 0] += 1e-6; A[:, 1, 1] += 1e-6; A[:, 2, 2] += 1e-6

        rhs = np.stack([
            (wx[:, :, None] * fn).sum(1),
            (wy[:, :, None] * fn).sum(1),
            (w[:, :, None] * fn).sum(1),
        ], axis=1)

        coef = np.zeros((len(q), 3, len(FORMATIONS)))
        try:
            coef = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            for r in range(len(q)):
                try:
                    coef[r] = np.linalg.solve(A[r], rhs[r])
                except np.linalg.LinAlgError:
                    coef[r] = 0.0

        pred = coef[:, 0, :] * q[:, [0]] + coef[:, 1, :] * q[:, [1]] + coef[:, 2, :]
        return pred  # (n_query, n_formations), still in original (unscaled) target units


def list_wells(split_dir):
    files = glob.glob(os.path.join(split_dir, "*__horizontal_well.csv"))
    return sorted(os.path.basename(f).split("__")[0] for f in files)


def add_spatial_prior_feature(dataset_df, hw_cache, knn_index, guard_ft=200.0):
    """For each well's eval-zone rows, impute the 6 formation depths at their
    (X, Y) via the KNN plane (self-well excluded), calibrate an offset from
    that SAME well's own known-zone rows (also spatially imputed, not exact -
    so the feature is computed identically for train and test wells), derive
    a candidate TVT per formation, and take the median across formations as
    one aggregate `spatial_tvt_prior` feature.

    Guarded (diagnosed after the first attempt): the local weighted-plane fit
    is ill-conditioned when its k neighbor wells are near-collinear in (X, Y),
    which produces wild extrapolated coefficients - confirmed on well
    `09ec2ca9`, whose raw spatial estimate was -51,563 ft against a true
    value of ~11,051 ft. Guard: fall back to `linear_prior` (the already-
    trusted Stage 2 signal) for any row whose candidate spatial estimate
    disagrees with it by more than `guard_ft` - mirrors the Stage 7 pattern
    of never trusting an unstable projection blindly."""
    out = np.full(len(dataset_df), np.nan)

    for well, group in dataset_df.groupby("well", sort=False):
        hw = hw_cache[well]
        known = hw[hw["TVT_input"].notna()]
        if len(known) < 5:
            continue

        known_xy = known[["X", "Y"]].to_numpy(dtype=float)
        known_form = knn_index.impute(known_xy, self_wid=well)  # (n_known, 6)
        known_z = known["Z"].to_numpy(dtype=float)
        known_tvt = known["TVT_input"].to_numpy(dtype=float)
        # offset[f] = median(known_TVT - (-Z + imputed_formation_f))
        offsets = np.median(known_tvt[:, None] - (-known_z[:, None] + known_form), axis=0)

        eval_xy = group[["X", "Y"]].to_numpy(dtype=float)
        eval_form = knn_index.impute(eval_xy, self_wid=well)  # (n_eval, 6)
        eval_z = group["Z"].to_numpy(dtype=float)
        candidate_tvt = -eval_z[:, None] + eval_form + offsets[None, :]
        spatial_prior = np.median(candidate_tvt, axis=1)

        linear_prior = group["linear_prior"].to_numpy(dtype=float)
        unstable = np.abs(spatial_prior - linear_prior) > guard_ft
        spatial_prior = np.where(unstable, linear_prior, spatial_prior)

        out[group.index.to_numpy()] = spatial_prior

    return out


if __name__ == "__main__":
    t0 = time.time()
    wells = list_wells(config.TRAIN_DIR)
    print(f"Building features for {len(wells)} wells...")
    dataset_df = s4a.build_dataset(config.TRAIN_DIR, wells)
    dataset_df = dataset_df.dropna(subset=["target"])
    dataset_df = dataset_df.reset_index(drop=True)
    print(f"Dataset: {dataset_df.shape}, built in {time.time()-t0:.1f}s")

    print("\nBuilding FormationPlaneKNN index from all 773 train wells...")
    knn_index = FormationPlaneKNN(wells, config.TRAIN_DIR)
    print(f"Index built on {len(knn_index.df)} wells with valid formation columns")

    print("\nCaching per-well horizontal_well.csv for spatial feature building...")
    hw_cache = {}
    for w in wells:
        hw_cache[w] = pd.read_csv(os.path.join(config.TRAIN_DIR, f"{w}__horizontal_well.csv")).reset_index(drop=True)

    print("\nComputing spatial_tvt_prior feature (leak-free: self-well excluded per query)...")
    t1 = time.time()
    dataset_df["spatial_tvt_prior"] = add_spatial_prior_feature(dataset_df, hw_cache, knn_index)
    print(f"Spatial feature built in {time.time()-t1:.1f}s")
    print(f"NaN spatial_tvt_prior: {dataset_df['spatial_tvt_prior'].isna().sum()} / {len(dataset_df)}")

    raw_rmse = np.sqrt(np.mean((dataset_df["spatial_tvt_prior"].fillna(dataset_df["target"].mean())
                                 - dataset_df["target"]) ** 2))
    print(f"spatial_tvt_prior ALONE (no model, just the raw estimate): RMSE {raw_rmse:.4f}")

    print("\n=== Baseline: Stage 4a without spatial feature ===")
    baseline_rmse = s4a.cross_validated_rmse(dataset_df)[0]
    print(f"Stage 4a baseline OOF RMSE (no spatial prior): {baseline_rmse:.4f}")

    print("\n=== Stage 4a + spatial_tvt_prior feature (5-fold GroupKFold OOF) ===")
    aug_feature_cols = s4a.FEATURE_COLS + ["spatial_tvt_prior"]
    X = dataset_df[aug_feature_cols].copy()
    X["spatial_tvt_prior"] = X["spatial_tvt_prior"].fillna(dataset_df["target"].mean())
    y = dataset_df["target"].to_numpy()
    groups = dataset_df["well"].to_numpy()

    gkf = GroupKFold(n_splits=5)
    oof_pred = np.full(len(dataset_df), np.nan)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
        model.fit(X.iloc[train_idx], y[train_idx])
        oof_pred[val_idx] = model.predict(X.iloc[val_idx])
        fold_rmse = np.sqrt(np.mean((y[val_idx] - oof_pred[val_idx]) ** 2))
        print(f"  fold {fold}: {len(val_idx)} rows, RMSE {fold_rmse:.4f}")

    aug_rmse = float(np.sqrt(np.mean((oof_pred - y) ** 2)))
    print(f"\nStage 4a + spatial_tvt_prior OOF RMSE: {aug_rmse:.4f}")
    print(f"Stage 4a baseline (no spatial):        {baseline_rmse:.4f}")
    print(f"Delta: {aug_rmse - baseline_rmse:+.4f}")

    print("\nReference:")
    print("Stage 4a (no spatial): 52.90 local, 45.196 public LB")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
