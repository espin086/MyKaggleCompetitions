"""Stage 3 (attempt 2): windowed GR shape-matching, gated by match confidence.

Attempt 1 (stage3_gr_match.py) matched each eval row to the typewell TVT with
the single closest GR value, within a window around the Stage 2 linear prior.
Tested WORSE than the plain linear prior (74.72 vs 70.23 RMSE on the same 50
wells) - a single GR reading is noisy and the nearest-value match latches onto
noise instead of real structure.

This version matches a small LOCAL WINDOW of consecutive GR readings (shape,
not a single value) against equal-length windows of the typewell, which
averages out point noise. It also only overrides the linear prior when the
match's residual error is confidently low (gated by a per-well threshold) -
a bad/ambiguous match falls back to the linear prior instead of injecting
noise, which is what likely hurt attempt 1.
"""

import glob
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import config

HALF_WINDOW = 5          # +/- rows around each eval row for the shape window
SEARCH_EXTRA_FT = 100.0  # +/- ft around the linear prior to search the typewell
ACCEPT_SLACK = 1.5       # keep matches with SSE <= this * per-well median SSE


def list_wells(split_dir):
    files = glob.glob(os.path.join(split_dir, "*__horizontal_well.csv"))
    return sorted(os.path.basename(f).split("__")[0] for f in files)


def linear_prior_predict(known, eval_rows):
    if len(known) < 2:
        fallback = known["TVT_input"].mean() if len(known) else np.nan
        return np.full(len(eval_rows), fallback)
    model = LinearRegression()
    model.fit(known[["MD", "Z"]], known["TVT_input"])
    return model.predict(eval_rows[["MD", "Z"]])


def windowed_shape_match(prior_preds, eval_gr, tw_tvt, tw_gr,
                          half_win=HALF_WINDOW, search_extra=SEARCH_EXTRA_FT,
                          accept_slack=ACCEPT_SLACK):
    n = len(prior_preds)
    if n == 0:
        return prior_preds

    gr_filled = pd.Series(eval_gr).interpolate(limit_direction="both").to_numpy()
    if np.isnan(gr_filled).all():
        return prior_preds  # no GR signal at all in this eval zone

    refined = prior_preds.copy()
    match_err = np.full(n, np.nan)

    for i in range(n):
        lo_row, hi_row = max(0, i - half_win), min(n, i + half_win + 1)
        local_gr = gr_filled[lo_row:hi_row]
        L = len(local_gr)
        if np.isnan(local_gr).any() or L < 3:
            continue

        center_prior = prior_preds[i]
        lo_idx = np.searchsorted(tw_tvt, center_prior - search_extra)
        hi_idx = np.searchsorted(tw_tvt, center_prior + search_extra)
        if hi_idx - lo_idx < L:
            continue

        seg_gr = tw_gr[lo_idx:hi_idx]
        seg_tvt = tw_tvt[lo_idx:hi_idx]
        windows = np.lib.stride_tricks.sliding_window_view(seg_gr, L)
        sse = np.sum((windows - local_gr[None, :]) ** 2, axis=1)
        best = int(np.argmin(sse))

        center_offset = i - lo_row
        refined[i] = seg_tvt[best + center_offset]
        match_err[i] = sse[best] / L

    valid = ~np.isnan(match_err)
    if valid.sum() == 0:
        return prior_preds

    thresh = np.nanmedian(match_err[valid]) * accept_slack
    keep = valid & (match_err <= thresh)
    return np.where(keep, refined, prior_preds)


def fit_predict_well(hz_df, tw_df):
    hz_df = hz_df.reset_index(drop=True)
    known = hz_df[hz_df["TVT_input"].notna()]
    eval_rows = hz_df[hz_df["TVT_input"].isna()]
    if len(eval_rows) == 0:
        return eval_rows.index, np.array([])

    prior_preds = linear_prior_predict(known, eval_rows)

    tw = tw_df.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    if len(tw) < HALF_WINDOW * 2 + 1:
        return eval_rows.index, prior_preds

    tw_tvt = tw["TVT"].to_numpy()
    tw_gr = tw["GR"].to_numpy()
    eval_gr = eval_rows["GR"].to_numpy()

    refined = windowed_shape_match(prior_preds, eval_gr, tw_tvt, tw_gr)
    return eval_rows.index, refined


def local_cv(max_wells=None):
    wells = list_wells(config.TRAIN_DIR)
    if max_wells:
        wells = wells[:max_wells]

    all_true, all_pred = [], []
    per_well_rmse = {}
    t0 = time.time()

    for well in wells:
        hz = pd.read_csv(os.path.join(config.TRAIN_DIR, f"{well}__horizontal_well.csv"))
        tw = pd.read_csv(os.path.join(config.TRAIN_DIR, f"{well}__typewell.csv"))
        eval_idx, preds = fit_predict_well(hz, tw)
        if len(eval_idx) == 0:
            continue
        true = hz.loc[eval_idx, "TVT"].to_numpy()
        all_true.extend(true)
        all_pred.extend(preds)
        per_well_rmse[well] = float(np.sqrt(np.mean((true - preds) ** 2)))

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    overall_rmse = float(np.sqrt(np.mean((all_true - all_pred) ** 2)))
    elapsed = time.time() - t0
    return overall_rmse, per_well_rmse, len(wells), elapsed


if __name__ == "__main__":
    rmse, per_well, n_wells, elapsed = local_cv()
    sorted_wells = sorted(per_well.items(), key=lambda kv: kv[1])
    print(f"Wells scored: {len(per_well)} / {n_wells}  ({elapsed:.1f}s)")
    print(f"Stage 3 (windowed shape match, gated) overall local RMSE: {rmse:.4f}")
    print("Stage 2 baseline for reference: 67.09 (median 33.07)")
    print("Public LB leader for reference: ~4.86")
    print()
    print("Best 5 wells:", sorted_wells[:5])
    print("Worst 5 wells:", sorted_wells[-5:])
    print(f"Median per-well RMSE: {np.median(list(per_well.values())):.4f}")
