# TitanicMachineLearningFromDisaster

Kaggle competition workspace. Slug: `titanic` · https://www.kaggle.com/competitions/titanic

## Overview

Predict which passengers survived the Titanic shipwreck from passenger attributes. Binary
classification — the "hello world" of Kaggle.

## Evaluation metric

Accuracy — the fraction of test passengers whose survival is predicted correctly.

## Data

- `data/` — `train.csv`, `test.csv`, `gender_submission.csv` (sample submission).
- Target column: `Survived` (0/1). ID column: `PassengerId`.

## Model (v1)

`src/model.py` — a scikit-learn `GradientBoostingClassifier` on engineered features:

- Base: `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`.
- Engineered: `FamilySize` (SibSp + Parch + 1), `IsAlone`, `Title` (parsed from `Name`, rare
  titles collapsed).
- Preprocessing: median impute + scale for numerics, most-frequent impute + one-hot for
  categoricals, in a single `Pipeline`.

**Local 5-fold cross-validated accuracy: 0.8406 ± 0.012** (vs ~0.62 for the old all-zeros
baseline). The public leaderboard score is recorded in `SUBMISSIONS.md` after each submit.

## Run it

```
cd competitions/TitanicMachineLearningFromDisaster
uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt
python src/model.py        # prints CV accuracy, writes submissions/submission.csv
```

## Submit + track

```
python src/submit.py "v1: gradient boosting + engineered features"
```

Submits `submissions/submission.csv`, polls for the public score, and appends a row to
`SUBMISSIONS.md`. Requires Kaggle auth and the competition rules accepted on the website.
Manual equivalent:

```
kaggle competitions submit -c titanic -f submissions/submission.csv -m "<approach>"
kaggle competitions submissions -c titanic
```

> Auth note: the `KGAT_` API token needs kaggle CLI >= 1.8.0 (written to `~/.kaggle/access_token`).
> The legacy `~/.kaggle/kaggle.json` (username + hex key) works on older CLIs.

## Layout

```
data/           train.csv, test.csv, gender_submission.csv
src/            config.py, model.py, submit.py
submissions/    submission.csv (generated)
SUBMISSIONS.md  leaderboard-score log
```

## Experiment log

| Date | Approach | CV accuracy | Public score | Notes |
|---|---|---|---|---|
| 2026-07-13 | GradientBoosting + Title/FamilySize/IsAlone | 0.8406 | 0.76315 | first real model; CV overshoots LB — add held-out val, try RF/XGB, Fare bins |

## Checklist

- [x] Rules accepted on the competition website
- [x] Data downloaded to `data/`
- [x] Baseline model built + validated (CV 0.8406)
- [x] `submission.csv` generated (418 rows, correct format)
- [x] Submitted to Kaggle — public score 0.76315 logged in `SUBMISSIONS.md`
