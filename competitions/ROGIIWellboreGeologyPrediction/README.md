# ROGII - Wellbore Geology Prediction

Kaggle competition workspace. Slug: `rogii-wellbore-geology-prediction`
https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction

**Type:** Featured **Code Competition** (notebook submission, internet disabled) ·
**Prize:** $50,000 · **Final deadline:** 2026-08-05 · **Entry deadline:** 2026-07-29

## Goal in one line

Predict `tvt` (True Vertical Thickness, ft) along the hidden "evaluation zone" of ~200
horizontal oil-and-gas wells, given each well's trajectory + gamma-ray log and a vertical
reference log (typewell). Lower RMSE wins.

## Status

- [x] **Rules accepted / competition joined**
- [x] Data downloaded to `data/` (773 train wells, 3 visible test wells; real ~200-well test
      set is swapped in when the notebook reruns)
- [ ] EDA on well geometry + logs
- [x] Baseline model + local CV (Stage 2 linear prior. see result below, not competitive yet)
- [x] Submission notebook built + executed locally (`notebooks/submission.ipynb`)
- [ ] First notebook submission actually pushed to Kaggle for a public score

## Baseline result (Stage 2. per-well linear prior `tvt ~ MD + Z`)

`src/baseline.py` fits `tvt = a*MD + b*Z + c` per well on that well's own known rows
(`TVT_input` not null), then predicts that same well's real evaluation zone (`TVT_input`
null). no synthetic masking; every train well already carries its own real eval zone, so
this is the actual test-time procedure. Run: `PYTHONPATH=src python3 src/baseline.py`.

**Overall local RMSE: 67.09** (median per-well: 33.07) vs. the public LB leader's ~4.86.
Not competitive. confirms the forum consensus (`context/04-discussion-intel.md`): a flat
linear/tabular fit plateaus hard. Root cause, confirmed by inspecting the worst well
(`4a335117`, RMSE 410): the eval zone is a long extrapolation. known zone spans ~1,700 ft of
MD, the eval zone spans ~5,000 ft beyond it. so a straight-line fit blows up wherever the
well crosses a fault or dip. Best wells (RMSE ~1.5-2.5) are ones where the geology stays
flat clear through the eval zone; worst wells (RMSE 280-410) hit structural breaks.

**Verdict:** pipeline is proven end-to-end (real per-well fit -> real eval-zone scoring), but
the model itself needs the typewell/GR alignment (Stage 3) before it's submission-worthy.
a linear-only submission would score far off the leaderboard.

## Submission notebook

`notebooks/submission.ipynb` is the actual submission artifact for this **Code Competition**
(rules require submitting a Notebook, not a CLI file-upload). It's self-contained: same
Stage 2 per-well linear fit as `src/baseline.py`, no internet access needed, auto-detects
Kaggle's mounted input path (`/kaggle/input/rogii-wellbore-geology-prediction/`) vs. this
repo's local `../data/` for local testing, and writes `submission.csv` after asserting the
id set exactly matches `sample_submission.csv` with no NaNs.

**Verified locally** (executed end-to-end with `jupyter nbconvert --execute`, output saved
at `submissions/submission_stage2_linear.csv`): 14,151 rows, ids match sample_submission
exactly, `tvt` range ~11,589-12,216 (sane, no NaNs). This is the same non-competitive Stage 2
baseline (local RMSE 67.09) - it locks the submit pipeline, not a leaderboard attempt.

### To actually submit on Kaggle

1. Upload `notebooks/submission.ipynb` as a new Kaggle Notebook, attach the competition
   dataset (`rogii-wellbore-geology-prediction`) as input.
2. **Turn internet off** in notebook settings (required by the rules).
3. Run all cells (`Save & Run All`) - it writes `submission.csv` in the Kaggle output.
4. Click **Submit to Competition** from the notebook's Output tab.
5. Read the score back with `kaggle competitions submissions -c rogii-wellbore-geology-prediction`
   and log it in the Experiment log below - don't report a number the CLI hasn't returned.

You get 5 submissions/day - this pipeline-proving baseline is a fine use of one of today's,
but save most of them for after Stage 3 (typewell/GR alignment) actually moves the needle.

## Join (one-time, manual. unblocks everything else)

The CLI returns `403` on download/submit until the rules are accepted in the browser.
Open https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/rules → click
**Join Competition** → accept. Then:

```
cd competitions/ROGIIWellboreGeologyPrediction
kaggle competitions download -c rogii-wellbore-geology-prediction -p data/
cd data && unzip -q -o '*.zip' && rm -f *.zip && cd ..
```

> Requires kaggle CLI ≥ 2.x (the `KGAT_` access token in `~/.kaggle/access_token` needs it).
> Installed globally via `uv tool install kaggle` at `~/.local/bin/kaggle`.

## The data (per-well, not flat CSVs)

`data/train/` (~700 wells) and `data/test/` (~200 wells). Each well = an 8-char hash:

| File | What |
|---|---|
| `{hash}__horizontal_well.csv` | The lateral: `MD`, `X`, `Y`, `Z`, `GR`, `TVT_input`, and (train only) `TVT` + formation-depth columns `ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA`. |
| `{hash}__typewell.csv` | Vertical reference log: `TVT` (depth index), `GR`, `Geology` (formation label). Used to correlate the lateral's GR against a known column. |
| `{hash}.png` | (train only) cross-section visualization of the well path + geology. |

**Target:** `tvt` for rows in the evaluation zone, where `TVT_input` is `NaN`. Everywhere
else `TVT_input` is a copy of the true `TVT` and can be used as a feature.

**Submission:** `id,tvt` where `id = {hash}_{row_index}`. See `data/sample_submission.csv`.
The visible `test/` folder is a stand-in copied from train; the real hidden test set is
swapped in when the notebook is rerun.

Full field-by-field dictionary and strategy notes are in `context/`. read those before
modeling. Agents working this competition should start at `context/00-agent-brief.md`.

## Layout

```
data/           per-well train/ + test/ CSVs, sample_submission.csv (gitignored. re-download)
src/            config.py, model + submission code
submissions/    submission.csv (generated)
context/        agent-facing brief, data dictionary, rules, discussion intel, plan of attack
SUBMISSIONS.md  leaderboard-score log (create on first submit)
```

## Experiment log

| Date | Approach | Local CV RMSE | Public LB | Notes |
|---|---|---|---|---|
| 2026-07-13 | Per-well linear `tvt ~ MD + Z` (Stage 2 baseline) | 67.09 (median 33.07) | not submitted | Pipeline proven end-to-end; not competitive. Long extrapolation into eval zone breaks on faults. Next: Stage 3 typewell/GR alignment. |

Anchor Kanban card: TBD (create on the JJ board via jj-kanban).
