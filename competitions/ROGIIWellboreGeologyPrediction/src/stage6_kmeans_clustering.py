"""Stage 6: K-means cluster features + Optuna tuning of k.

Hypothesis: clustering rows on the Stage 4a feature set (geometry + GR +
linear_prior + windowed_match) captures geological "regimes" (e.g. flat vs
faulted terrain) that let the downstream gradient-boosted model specialize
its splits, on top of what it already learns from the raw features alone.

Pipeline:
1. Elbow analysis on the full standardized feature set (descriptive only -
   not a leak risk since no supervised model is trained here).
2. Search range = a few k values below/above the elbow.
3. Optuna tunes k, objective = leak-free GroupKFold-by-well OOF RMSE with
   KMeans fit WITHIN each training fold only (never on validation rows) and
   the cluster label passed to HistGradientBoostingRegressor as a native
   categorical feature.
4. Final honest 5-fold confirmation of the best k found.
"""

import glob
import os
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import config
import stage4_global_model as s4a

RANDOM_STATE = config.RANDOM_STATE
CLUSTER_FEATURE_COLS = s4a.FEATURE_COLS  # same 11 features Stage 4a already uses
AUGMENTED_COLS = CLUSTER_FEATURE_COLS + ["cluster_label", "cluster_dist"]


def list_wells(split_dir):
    files = glob.glob(os.path.join(split_dir, "*__horizontal_well.csv"))
    return sorted(os.path.basename(f).split("__")[0] for f in files)


def standardize(X_train, X_other=None):
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    std[std < 1e-6] = 1.0
    out_train = (X_train - mean) / std
    if X_other is None:
        return out_train, mean, std
    return out_train, (X_other - mean) / std, mean, std


def elbow_analysis(X, k_range=range(2, 16)):
    """Fit KMeans for each k on the full (standardized) dataset - descriptive
    only, not used to build the trained/validated model."""
    Xs, _, _ = standardize(X)
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5)
        km.fit(Xs)
        inertias.append(km.inertia_)
        print(f"  k={k:2d}  inertia={km.inertia_:,.0f}")
    return list(k_range), inertias


def find_elbow(ks, inertias):
    """Kneedle-style: point with max perpendicular distance from the line
    connecting the first and last (k, inertia) points."""
    ks = np.array(ks, dtype=float)
    inertias = np.array(inertias, dtype=float)
    # Normalize both axes to [0, 1] so distance isn't dominated by inertia's scale.
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (inertias - inertias.min()) / (inertias.max() - inertias.min())
    p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    distances = []
    for xi, yi in zip(x, y):
        p = np.array([xi, yi])
        dist = np.abs(np.cross(line_vec, p - p1)) / line_len
        distances.append(dist)
    elbow_idx = int(np.argmax(distances))
    return int(ks[elbow_idx])


def add_cluster_features(X_train, X_val, k):
    """Fit StandardScaler + KMeans on TRAIN rows only (leak-free), transform
    both train and val. Returns augmented feature matrices."""
    Xs_train, Xs_val, _, _ = standardize(X_train, X_val)
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5)
    train_labels = km.fit_predict(Xs_train)
    val_labels = km.predict(Xs_val)
    train_dist = np.linalg.norm(Xs_train - km.cluster_centers_[train_labels], axis=1)
    val_dist = np.linalg.norm(Xs_val - km.cluster_centers_[val_labels], axis=1)

    train_aug = np.column_stack([X_train, train_labels, train_dist])
    val_aug = np.column_stack([X_val, val_labels, val_dist])
    return train_aug, val_aug


def groupkfold_rmse_with_k(X, y, groups, k, n_splits=5):
    """Leak-free OOF RMSE with cluster features for a given k."""
    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = np.full(len(y), np.nan)
    cluster_label_idx = AUGMENTED_COLS.index("cluster_label")

    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train_aug, X_val_aug = add_cluster_features(X[train_idx], X[val_idx], k)
        model = HistGradientBoostingRegressor(
            random_state=RANDOM_STATE,
            categorical_features=[cluster_label_idx],
        )
        model.fit(X_train_aug, y[train_idx])
        oof_pred[val_idx] = model.predict(X_val_aug)

    return float(np.sqrt(np.mean((oof_pred - y) ** 2)))


def make_objective(X, y, groups, n_splits=3):
    """Optuna objective - uses a faster 3-fold split during search; the best
    k found gets a full honest 5-fold confirmation afterward."""
    def objective(trial):
        k = trial.suggest_int("k", K_SEARCH_MIN, K_SEARCH_MAX)
        rmse = groupkfold_rmse_with_k(X, y, groups, k, n_splits=n_splits)
        print(f"  trial k={k}: RMSE {rmse:.4f}")
        return rmse
    return objective


K_SEARCH_MIN = None  # set after elbow analysis
K_SEARCH_MAX = None


if __name__ == "__main__":
    t0 = time.time()
    wells = list_wells(config.TRAIN_DIR)
    print(f"Building features for {len(wells)} wells...")
    dataset_df = s4a.build_dataset(config.TRAIN_DIR, wells)
    dataset_df = dataset_df.dropna(subset=["target"])
    print(f"Dataset: {dataset_df.shape}, built in {time.time()-t0:.1f}s")

    X = dataset_df[CLUSTER_FEATURE_COLS].fillna(0).to_numpy()
    y = dataset_df["target"].to_numpy()
    groups = dataset_df["well"].to_numpy()

    print("\n=== Elbow analysis (k=2..15, full dataset, descriptive) ===")
    ks, inertias = elbow_analysis(X)
    elbow_k = find_elbow(ks, inertias)
    print(f"\nElbow point: k={elbow_k}")

    K_SEARCH_MIN = max(2, elbow_k - 3)
    K_SEARCH_MAX = elbow_k + 5
    print(f"Optuna search range: k in [{K_SEARCH_MIN}, {K_SEARCH_MAX}]")

    print("\n=== Baseline: Stage 4a without cluster features ===")
    baseline_rmse = s4a.cross_validated_rmse(dataset_df)[0]
    print(f"Stage 4a baseline OOF RMSE (no clusters): {baseline_rmse:.4f}")

    print(f"\n=== Optuna tuning k (3-fold objective, search range above) ===")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(make_objective(X, y, groups, n_splits=3), n_trials=10)

    best_k = study.best_params["k"]
    print(f"\nBest k from Optuna (3-fold): {best_k}, RMSE {study.best_value:.4f}")

    print(f"\n=== Final honest 5-fold confirmation of k={best_k} ===")
    final_rmse = groupkfold_rmse_with_k(X, y, groups, best_k, n_splits=5)
    print(f"Final 5-fold OOF RMSE WITH k={best_k} clusters: {final_rmse:.4f}")
    print(f"Stage 4a baseline (no clusters): {baseline_rmse:.4f}")
    print(f"Delta: {final_rmse - baseline_rmse:+.4f}")

    print("\nReference:")
    print("Stage 4a (no clusters): 52.90 local, 45.196 public LB")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
