# MyKaggleCompetitions

One repo holding my Kaggle competition work, one self-contained folder per competition under
`competitions/`. Each folder carries its own data directory, `src/` modeling code, notebooks,
generated submissions, and a `SUBMISSIONS.md` log of real leaderboard scores read back from the
Kaggle CLI (not estimated). Models are scikit-learn based; submissions go out either through a
local `src/submit.py` or, for code competitions, by pushing a notebook kernel with the Kaggle CLI.

## Competitions

| Competition | Task | Approach and models | Best recorded score | Entry point |
|---|---|---|---|---|
| `TitanicMachineLearningFromDisaster` | Binary classification, survival prediction. Metric: accuracy | v1 is a `GradientBoostingClassifier` on `Pclass`/`Sex`/`Age`/`Fare`/`Embarked` plus engineered `FamilySize`, `IsAlone`, `Title`, wrapped in an impute+scale+one-hot `Pipeline`. Later runs used the `kaggle-ml-loop` skill: baselines, Optuna tuning, then voting/blending/stacking ensembles | Public LB **0.78947** (stacking ensemble on raw features). v1 hand-built model scored 0.76315 | `src/model.py`, then `src/submit.py "<message>"` |
| `ROGIIWellboreGeologyPrediction` | Regression. Predict `tvt` (true vertical thickness, ft) in the hidden evaluation zone of ~200 horizontal wells. Metric: RMSE, lower is better | Staged experiments: per-well linear prior `tvt ~ MD + Z` (`baseline.py`), typewell GR alignment heuristics (`stage3_*.py`), a single global `HistGradientBoostingRegressor` over pooled wells validated with GroupKFold by well (`stage4_global_model.py`), a 1D CNN over 41-row windows (`stage4b_cnn_model.py`), an NNLS blend of the three signals (`stage5_ensemble.py`), and K-means cluster features (`stage6_kmeans_clustering.py`) | Public LB **45.196** (Stage 4a gradient-boosted model). Stage 5 blend 45.997, Stage 4b CNN 72.734, Stage 2 linear 80.534. Stage 6 tested worse locally and was not submitted | `src/stage4_global_model.py` for modeling; `notebooks/submission_stage4a.ipynb` for the Kaggle kernel |
| `LLMDetectAIGeneratedText` | Detect whether an essay was written by an LLM | Exploration only. A single notebook loads `data/train_essays.csv` and runs a `DataFrameExplorer` from the `jjutils` package. No model or submission code here yet | `src/data_exploration.ipynb` |

Competition-specific detail, including per-stage diagnostics and why negative results were kept,
lives in each folder's `README.md`, `SUBMISSIONS.md`, and (for ROGII) `context/`.

## Repo layout

```
competitions/<Name>/
  data/          competition data. gitignored, re-download with the kaggle CLI
  src/           config.py plus modeling and submission code
  notebooks/     EDA, champion, and Kaggle submission notebooks (tracked)
  submissions/   generated submission.csv snapshots
  SUBMISSIONS.md leaderboard-score log, newest row at the bottom
  README.md      goal, approach, results for that competition
.claude/         project config: the kaggle and kaggle-ml-loop skills, rules, hooks
CLAUDE.md        working conventions for agents in this repo
TASK.md          append-only log of tasks executed against this project
```

Conventions worth knowing:

- Notebooks under `competitions/*/notebooks/` are tracked on purpose. They are the durable run
  history. `kaggle_run/` (MLflow db, intermediate datasets, models) is gitignored.
- The `kaggle-ml-loop` skill writes a timestamped pair of notebooks per run (EDA + champion) and
  logs one row per run in `notebooks/INDEX.md`.
- ROGII competition data is not committed. It is 1.33 GB and competition-use-only under the rules.

## Requirements

Python 3.10 or newer, plus the Kaggle CLI (2.x for `KGAT_` access tokens).

Per competition, the base dependencies are small:

- Titanic: `pandas`, `scikit-learn` (`competitions/TitanicMachineLearningFromDisaster/requirements.txt`)
- ROGII: `pandas`, `numpy`, `scikit-learn`, `scipy` (`competitions/ROGIIWellboreGeologyPrediction/requirements.txt`)

The `kaggle-ml-loop` pipeline needs more: `mlflow`, `optuna`, `pyyaml`, plus the notebook stack
`seaborn`, `matplotlib`, `nbformat`, `nbconvert`, `jupyter`, `papermill`
(`.claude/skills/kaggle-ml-loop/requirements.txt`).

## Installation

```bash
git clone https://github.com/espin086/MyKaggleCompetitions.git
cd MyKaggleCompetitions/competitions/TitanicMachineLearningFromDisaster
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

Kaggle auth is per machine, not per repo. Put an API token at `~/.kaggle/access_token`
(`KGAT_...`) or the legacy `~/.kaggle/kaggle.json` with `chmod 600`. Do not set placeholder
`KAGGLE_USERNAME` / `KAGGLE_KEY` env vars: non-empty values override the token file and force a
401. `.env-template` at the repo root shows the legacy env-var shape. Verify with:

```bash
kaggle competitions list | head
```

## Usage

Train and submit the Titanic model:

```bash
cd competitions/TitanicMachineLearningFromDisaster
python src/model.py                                    # prints CV accuracy, writes submissions/submission.csv
python src/submit.py "v1: gradient boosting + engineered features"
```

`src/submit.py` submits the file, polls for the public score, and appends a row to
`SUBMISSIONS.md`. The manual equivalent:

```bash
kaggle competitions submit -c titanic -f submissions/submission.csv -m "<approach>"
kaggle competitions submissions -c titanic
```

Run the ROGII stages locally (each prints its own validation RMSE):

```bash
cd competitions/ROGIIWellboreGeologyPrediction
PYTHONPATH=src python3 src/baseline.py              # Stage 2 per-well linear prior
PYTHONPATH=src python3 src/stage4_global_model.py   # Stage 4a global GB model, several minutes
PYTHONPATH=src python3 src/stage5_ensemble.py       # Stage 5 NNLS blend
```

ROGII is a code competition, so scoring happens on Kaggle's infrastructure. Push the notebook and
submit the kernel version:

```bash
kaggle kernels push -p notebooks/kaggle_push
kaggle competitions submit -c rogii-wellbore-geology-prediction -k jjespinoza/<kernel-slug> -v <version>
kaggle competitions submissions -c rogii-wellbore-geology-prediction
```

Run the multi-loop pipeline inside a competition folder (config first, then the scripts in order):

```bash
cp ../../.claude/skills/kaggle-ml-loop/assets/config.yaml ./config.yaml   # edit paths, target, metric, n_loops
python ../../.claude/skills/kaggle-ml-loop/scripts/eda.py --config config.yaml
python ../../.claude/skills/kaggle-ml-loop/scripts/make_datasets.py --config config.yaml
python ../../.claude/skills/kaggle-ml-loop/scripts/train_baselines.py --config config.yaml
python ../../.claude/skills/kaggle-ml-loop/scripts/optimize.py --config config.yaml
python ../../.claude/skills/kaggle-ml-loop/scripts/ensemble.py --config config.yaml
python ../../.claude/skills/kaggle-ml-loop/scripts/select_champion.py --config config.yaml
cp kaggle_run/champion/submission.csv submissions/submission.csv
```

No LICENSE file is present in this repo.
