"""Stage 4b: 1D CNN sequence model over the Stage 4a feature set.

Stage 4a (HistGradientBoostingRegressor) treats every eval-zone row as an
independent sample - it has no notion that row i and row i+1 are neighbors
along the wellbore. This stage keeps the exact same per-row features (so any
gain is attributable to the sequence structure, not new information) but
convolves over a local window of consecutive rows, letting the model learn
local shape patterns (the thing Stage 3's hand-tuned GR-match was trying and
failing to do with a fixed heuristic) end-to-end.

Windows are built lazily per-sample from a per-well feature matrix kept in
memory (not materialized for every window - 3.8M windows x 41 rows x 11
channels would be tens of GB). Validated with GroupKFold by well, same as
Stage 4a, for a direct, comparable RMSE.
"""

import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

import config
import stage4_global_model as s4a

RANDOM_STATE = config.RANDOM_STATE
WINDOW = 41
FEATURE_COLS = s4a.FEATURE_COLS  # same 11 features as Stage 4a, for a fair comparison

# Attempt 1 (all 11 features): converged to near-zero train loss within 1
# epoch, held-out RMSE 81 - well-identity memorization via X/Y, not sequence
# learning. Attempt 2 (dropped X/Y, added dropout+weight_decay): STILL
# near-zero train loss by epoch 1, held-out RMSE 84.67 (regressed slightly) -
# MD (absolute), dist_from_known_boundary, and known_zone_rows are apparently
# almost as well-identifying per-well as X/Y were. Per verify-with-real-data,
# two non-reinforcing attempts is the signal to stop tuning blindly - but this
# is a third, differently-informed idea, not a guess: strip every scalar that
# could serve as a well-identity key and keep only GR (the actual sequence
# signal) plus the three model-output signals (already normalized estimates
# of the target, not identity leaks) plus eval_zone_frac (position WITHIN the
# eval zone, 0-1, same range for every well - doesn't identify which well).
CNN_FEATURE_COLS = ["GR", "linear_prior", "windowed_match", "match_minus_prior", "eval_zone_frac"]


def list_wells(split_dir):
    files = glob.glob(os.path.join(split_dir, "*__horizontal_well.csv"))
    return sorted(os.path.basename(f).split("__")[0] for f in files)


def preprocess_features(dataset_df):
    """Impute GR NaNs (~39% missing) and standardize every feature column -
    raw scale (e.g. X ~ 2.9M, MD ~ 15000) blows up gradients in a fresh CNN,
    and a single NaN anywhere in a window propagates NaN through the whole
    batch. Fit on the full dataset passed in (same minor global-stats leakage
    Stage 4a doesn't have, since HistGradientBoostingRegressor handles NaN
    and raw scale natively - acceptable for comparing architectures here,
    called out in the writeup)."""
    dataset_df = dataset_df.copy()
    dataset_df["GR"] = dataset_df["GR"].fillna(dataset_df["GR"].median())

    stats = {}
    for col in FEATURE_COLS:
        mean, std = dataset_df[col].mean(), dataset_df[col].std()
        std = std if std > 1e-6 else 1.0
        dataset_df[col] = (dataset_df[col] - mean) / std
        stats[col] = (mean, std)

    # Target is ~11,000-12,000 (absolute depth). Regressing that raw with MSE
    # loss from a near-zero random init converges far slower than a scaled
    # target - the output bias alone needs many steps to reach that magnitude.
    if "target" in dataset_df.columns:
        t_mean, t_std = dataset_df["target"].mean(), dataset_df["target"].std()
        dataset_df["target_scaled"] = (dataset_df["target"] - t_mean) / t_std
        stats["target"] = (t_mean, t_std)

    assert not dataset_df[FEATURE_COLS].isna().any().any(), "NaNs survived preprocessing"
    return dataset_df, stats


def build_per_well_arrays(dataset_df):
    """dataset_df: the long-format output of s4a.build_dataset (one row per
    eval-zone position, already sorted in MD order within each well since
    build_well_features preserves row order). Returns dicts keyed by well.
    """
    wells_features = {}
    wells_targets = {}
    target_col = "target_scaled" if "target_scaled" in dataset_df.columns else "target"
    has_target = target_col in dataset_df.columns
    for well, group in dataset_df.groupby("well", sort=False):
        wells_features[well] = group[CNN_FEATURE_COLS].to_numpy(dtype=np.float32)
        if has_target:
            wells_targets[well] = group[target_col].to_numpy(dtype=np.float32)
    return wells_features, wells_targets


class WellWindowDataset(Dataset):
    def __init__(self, wells_features, wells_targets, index_list, window=WINDOW):
        self.wells_features = wells_features
        self.wells_targets = wells_targets
        self.index_list = index_list
        self.window = window
        self.half = window // 2

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, i):
        well, row_idx = self.index_list[i]
        feats = self.wells_features[well]
        n = feats.shape[0]
        lo, hi = row_idx - self.half, row_idx + self.half + 1
        idx = np.clip(np.arange(lo, hi), 0, n - 1)
        window = feats[idx].T  # (channels, window_len)
        y = self.wells_targets[well][row_idx]
        return torch.from_numpy(window.copy()), torch.tensor(y, dtype=torch.float32)


class CNN1D(nn.Module):
    def __init__(self, in_channels=len(CNN_FEATURE_COLS), dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        h = self.net(x).squeeze(-1)
        return self.fc(h).squeeze(-1)


def train_one_fold(train_wells_feat, train_wells_tgt, train_index,
                    val_wells_feat, val_wells_tgt, val_index,
                    epochs=2, batch_size=512, lr=1e-3, weight_decay=1e-3, device="cpu"):
    train_ds = WellWindowDataset(train_wells_feat, train_wells_tgt, train_index)
    val_ds = WellWindowDataset(val_wells_feat, val_wells_tgt, val_index)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CNN1D().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss, n_seen = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(yb)
            n_seen += len(yb)
        print(f"    epoch {epoch}: train MSE {total_loss/n_seen:.2f} (RMSE {np.sqrt(total_loss/n_seen):.2f})")

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            trues.append(yb.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return preds, trues


def cross_validated_rmse(dataset_df, target_stats, n_splits=5, epochs=2, max_wells=None):
    t_mean, t_std = target_stats["target"]
    wells_features, wells_targets = build_per_well_arrays(dataset_df)
    all_wells = list(wells_features.keys())
    if max_wells:
        all_wells = all_wells[:max_wells]
        wells_features = {w: wells_features[w] for w in all_wells}
        wells_targets = {w: wells_targets[w] for w in all_wells}

    # Build (well, row_idx) index for every row across the wells we're using.
    full_index = [(w, i) for w in all_wells for i in range(len(wells_targets[w]))]
    well_of_index = np.array([w for w, _ in full_index])

    gkf = GroupKFold(n_splits=n_splits)
    all_preds = np.full(len(full_index), np.nan)
    all_trues = np.full(len(full_index), np.nan)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, wells: {len(all_wells)}, total rows: {len(full_index)}")

    for fold, (train_i, val_i) in enumerate(gkf.split(full_index, groups=well_of_index)):
        train_index = [full_index[i] for i in train_i]
        val_index = [full_index[i] for i in val_i]
        t0 = time.time()
        preds, trues = train_one_fold(
            wells_features, wells_targets, train_index,
            wells_features, wells_targets, val_index,
            epochs=epochs, device=device,
        )
        # Predictions/targets come back in scaled space - invert to raw TVT
        # units (ft) so RMSE is directly comparable to Stage 2/4a's numbers.
        preds = preds * t_std + t_mean
        trues = trues * t_std + t_mean
        fold_rmse = np.sqrt(np.mean((preds - trues) ** 2))
        print(f"  fold {fold}: {len(val_index)} rows, RMSE {fold_rmse:.4f} ({time.time()-t0:.1f}s)")
        all_preds[val_i] = preds
        all_trues[val_i] = trues

    overall_rmse = float(np.sqrt(np.mean((all_preds - all_trues) ** 2)))
    return overall_rmse


if __name__ == "__main__":
    max_wells = int(sys.argv[1]) if len(sys.argv) > 1 else None
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    t0 = time.time()
    wells = list_wells(config.TRAIN_DIR)
    if max_wells:
        wells = wells[:max_wells]
    print(f"Building features for {len(wells)} wells (reusing Stage 4a's feature pipeline)...")
    dataset_df = s4a.build_dataset(config.TRAIN_DIR, wells)
    dataset_df = dataset_df.dropna(subset=["target"])
    dataset_df, target_stats = preprocess_features(dataset_df)
    print(f"Dataset: {dataset_df.shape}, built in {time.time()-t0:.1f}s")

    print("\nStage 4b: GroupKFold (5-fold, by well) 1D CNN")
    rmse = cross_validated_rmse(dataset_df, target_stats, epochs=epochs)
    print(f"\nStage 4b overall out-of-fold RMSE: {rmse:.4f}")
    print("Stage 4a for reference: 52.90 (773-well OOF), public LB 45.196")
    print("Stage 2 for reference: 67.09 local, public LB 80.534")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
