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

- [ ] **Rules accepted / competition joined**. REQUIRED before data download or submit. One
      manual click on the website (`kaggle` CLI can't accept rules). See "Join" below.
- [ ] Data downloaded to `data/` (blocked on join. 403 until then)
- [ ] EDA on well geometry + logs
- [ ] Baseline model + local CV
- [ ] First notebook submission

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
|. |. |. |. | not started |

Anchor Kanban card: TBD (create on the JJ board via jj-kanban).
