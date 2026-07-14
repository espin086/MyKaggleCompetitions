"""Stage 2 baseline: per-well linear prior tvt ~ MD + Z.

Each training well already carries its own real evaluation zone (TVT_input is
NaN there, but the true TVT is still present for scoring). So local CV needs no
synthetic masking: fit on each well's known rows, predict its real eval-zone
rows, score against the true TVT. This mirrors the actual test-time procedure.
"""

import glob
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import config


def list_wells(split_dir):
    files = glob.glob(os.path.join(split_dir, "*__horizontal_well.csv"))
    return sorted(os.path.basename(f).split("__")[0] for f in files)


def fit_predict_well(df):
    """Fit tvt ~ MD + Z on known rows, predict eval-zone rows. Returns (eval_idx, preds)."""
    known = df[df["TVT_input"].notna()]
    eval_rows = df[df["TVT_input"].isna()]
    if len(eval_rows) == 0 or len(known) < 2:
        return eval_rows.index, np.full(len(eval_rows), known["TVT_input"].mean() if len(known) else np.nan)

    model = LinearRegression()
    model.fit(known[["MD", "Z"]], known["TVT"])
    preds = model.predict(eval_rows[["MD", "Z"]])
    return eval_rows.index, preds


def local_cv():
    """Score the per-well linear prior against each train well's real eval zone."""
    wells = list_wells(config.TRAIN_DIR)
    all_true, all_pred = [], []
    per_well_rmse = {}

    for well in wells:
        df = pd.read_csv(os.path.join(config.TRAIN_DIR, f"{well}__horizontal_well.csv"))
        eval_idx, preds = fit_predict_well(df)
        if len(eval_idx) == 0:
            continue
        true = df.loc[eval_idx, "TVT"].to_numpy()
        all_true.extend(true)
        all_pred.extend(preds)
        per_well_rmse[well] = float(np.sqrt(np.mean((true - preds) ** 2)))

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    overall_rmse = float(np.sqrt(np.mean((all_true - all_pred) ** 2)))
    return overall_rmse, per_well_rmse, len(wells)


if __name__ == "__main__":
    rmse, per_well, n_wells = local_cv()
    sorted_wells = sorted(per_well.items(), key=lambda kv: kv[1])
    print(f"Wells scored: {len(per_well)} / {n_wells}")
    print(f"Overall local RMSE (per-well linear tvt ~ MD + Z): {rmse:.4f}")
    print(f"Public LB leader (for reference): ~4.86")
    print()
    print("Best 5 wells:", sorted_wells[:5])
    print("Worst 5 wells:", sorted_wells[-5:])
    print(f"Median per-well RMSE: {np.median(list(per_well.values())):.4f}")
