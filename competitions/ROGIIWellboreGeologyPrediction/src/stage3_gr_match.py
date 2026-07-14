"""Stage 3: typewell/GR alignment on top of the Stage 2 linear prior.

Stage 2 (tvt ~ MD + Z) scored 67.09 RMSE - it extrapolates a straight line into
the eval zone and breaks wherever the well crosses a fault. The typewell is a
vertical reference log: GR as a function of geologic position (TVT). If the
lateral's GR reading at an eval-zone row matches a spot on the typewell's GR
curve, that spot's TVT is a much better estimate of true geologic position
than a straight-line extrapolation - it uses the actual rock signature.

Approach: for each eval row, search the typewell for the TVT whose GR most
closely matches the lateral's observed GR, restricted to a window around the
Stage 2 linear-prior estimate (GR values repeat across depths, so the linear
prior picks the right "branch" of an otherwise ambiguous match). Falls back to
the linear prior when GR is missing or no typewell coverage exists in-window.
"""

import glob
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import config

SEARCH_WINDOW_FT = 100.0  # +/- ft around the linear prior to search the typewell


def list_wells(split_dir):
    files = glob.glob(os.path.join(split_dir, "*__horizontal_well.csv"))
    return sorted(os.path.basename(f).split("__")[0] for f in files)


def linear_prior_predict(known, eval_rows):
    """Stage 2: fit tvt ~ MD + Z on known rows, predict eval rows."""
    if len(known) < 2:
        fallback = known["TVT_input"].mean() if len(known) else np.nan
        return np.full(len(eval_rows), fallback)
    model = LinearRegression()
    model.fit(known[["MD", "Z"]], known["TVT_input"])
    return model.predict(eval_rows[["MD", "Z"]])


def gr_match_refine(prior_preds, eval_gr, tw_tvt, tw_gr, window=SEARCH_WINDOW_FT):
    """Refine each prior prediction by matching GR against the typewell curve
    within +/- window ft. tw_tvt must be sorted ascending. Falls back to the
    prior estimate when GR is NaN or no typewell points fall in the window.
    """
    preds = prior_preds.copy()
    lo_idx = np.searchsorted(tw_tvt, prior_preds - window, side="left")
    hi_idx = np.searchsorted(tw_tvt, prior_preds + window, side="right")

    for i in range(len(preds)):
        gr = eval_gr[i]
        if np.isnan(gr):
            continue
        lo, hi = lo_idx[i], hi_idx[i]
        if hi <= lo:
            continue
        cand_tvt = tw_tvt[lo:hi]
        cand_gr = tw_gr[lo:hi]
        best = np.argmin(np.abs(cand_gr - gr))
        preds[i] = cand_tvt[best]

    return preds


def fit_predict_well(hz_df, tw_df):
    hz_df = hz_df.reset_index(drop=True)
    known = hz_df[hz_df["TVT_input"].notna()]
    eval_rows = hz_df[hz_df["TVT_input"].isna()]
    if len(eval_rows) == 0:
        return eval_rows.index, np.array([])

    prior_preds = linear_prior_predict(known, eval_rows)

    tw = tw_df.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    if len(tw) < 2:
        return eval_rows.index, prior_preds

    tw_tvt = tw["TVT"].to_numpy()
    tw_gr = tw["GR"].to_numpy()
    eval_gr = eval_rows["GR"].to_numpy()

    refined = gr_match_refine(prior_preds, eval_gr, tw_tvt, tw_gr)
    return eval_rows.index, refined


def local_cv(window=SEARCH_WINDOW_FT, max_wells=None):
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
    print(f"Stage 3 overall local RMSE (GR/typewell match, window={SEARCH_WINDOW_FT}ft): {rmse:.4f}")
    print("Stage 2 baseline for reference: 67.09 (median 33.07)")
    print("Public LB leader for reference: ~4.86")
    print()
    print("Best 5 wells:", sorted_wells[:5])
    print("Worst 5 wells:", sorted_wells[-5:])
    print(f"Median per-well RMSE: {np.median(list(per_well.values())):.4f}")
