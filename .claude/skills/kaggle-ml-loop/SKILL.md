---
name: kaggle-ml-loop
description: >
  Autonomous multi-loop Kaggle ML competition pipeline with MLflow tracking and
  scikit-learn. Runs N full passes (default 10) of: EDA → parallel feature-engineering
  agents that design dataset variants → algorithm-to-dataset mapping → baseline model
  training → Optuna hyperparameter optimization of top performers → ensembling
  (voting, weighted blend, stacking) → knowledge distillation that feeds the next
  loop's feature engineering. Ends by selecting a single champion model and writing
  a submission file. Use this skill whenever JJ mentions: "kaggle loop", "run the
  kaggle pipeline", "ml competition workflow", "kaggle agents", "train on this
  competition", "run 10 loops on this dataset", "find the winning model", "automl
  loop", "mlflow kaggle run", or drops a Kaggle train/test CSV and asks for the best
  model. Even casual phrases like "run the loop on this" or "kaggle this dataset"
  should trigger it.
compatibility: Requires Python 3.10+, scikit-learn, mlflow, optuna, pandas, numpy, pyyaml. Parallel FE agents use the Task tool when available (Claude Code); fall back to sequential recipe design otherwise.
---

# Kaggle ML Loop

An orchestrated, self-improving competition pipeline. You (Claude) are the
orchestrator and the creative feature-engineering brain. The scripts are the
deterministic muscle. MLflow is the single source of truth for every result.

## Integration with this repo (MyKaggleCompetitions)

This skill is the **modeling engine**. The lightweight `kaggle` skill owns the
CLI side (auth, scaffolding, submitting, `SUBMISSIONS.md` logging). Use them together:

1. **Scaffold + data** come from the `kaggle` skill: `.claude/scripts/new-competition.sh
   <slug> [Name]` creates `competitions/<Name>/{data,src,notebooks,submissions}` and
   downloads the data. Run this skill **inside that competition folder**.
2. **Run directory lives in the competition folder.** Copy `assets/config.yaml` to
   `competitions/<Name>/config.yaml` and set `run_dir: ./kaggle_run` (so it lands at
   `competitions/<Name>/kaggle_run/`). Point `train_path`/`test_path` at `./data/…`.
3. **Champion → submission handoff.** `select_champion.py` writes
   `kaggle_run/champion/submission.csv`. Copy it to the repo's standard submission path
   so the tracked submit flow picks it up:
   ```bash
   cp kaggle_run/champion/submission.csv submissions/submission.csv
   python src/submit.py "kaggle-ml-loop champion: <one-line description>"   # submits + logs to SUBMISSIONS.md
   ```
   (`src/submit.py` from the competition scaffold submits and appends the public score
   to `SUBMISSIONS.md`. Don't re-implement submitting here.) After scoring, edit the new
   `SUBMISSIONS.md` row to reference the run's two notebooks in `notebooks/` (same names
   logged in `notebooks/INDEX.md`) so a scored submission links back to the EDA + champion
   notebooks that produced it.
4. **Dependencies** beyond the scaffold's pandas/scikit-learn: `mlflow optuna scipy pyyaml`
   plus the notebook stack `seaborn matplotlib nbformat nbconvert jupyter ipykernel`.
   Install into the competition's venv: `uv pip install -r <skill>/requirements.txt`, then
   register the kernel once: `python -m ipykernel install --user --name python3`.
5. **Auth** is Kaggle-CLI auth (needs kaggle CLI ≥ 1.8.0 for `KGAT_` access tokens) — see the
   `kaggle` skill. This skill never calls the Kaggle API itself; it only produces the submission.
6. **`kaggle_run/` is a run artifact** — gitignore it (it holds `mlflow.db`, datasets, models).
   Commit the champion's provenance (`champion.json`) and the final `submissions/submission.csv`
   if you want the scored run reproducible; keep the multi-GB run dir out of git.

Everything below is the general pipeline; the six points above are the only repo-specific glue.

## Inputs you need before starting

1. `train.csv` (or similar) — must contain the target column
2. `test.csv` (optional) — for submission generation
3. Target column name, task type (auto-detected if omitted), competition metric
4. Loop count (default 10) — confirm with the user if not stated

Copy `assets/config.yaml` to the working directory, fill in paths/target/metric/
`n_loops`, and treat it as the run's contract. All scripts read it.

**Data splitting is automatic.** `holdout_fraction` (default 0.2) carves a
stratified, seeded holdout from the training file. Every loop — EDA, feature
engineering, training, Optuna, ensembling — sees only the dev split. The
holdout is touched exactly once, at final champion selection. Never point a
loop script at the holdout.

## Directory contract

All work happens in a run directory (e.g. `./kaggle_run/`):

```
kaggle_run/
├── config.yaml
├── eda/                  # eda.py outputs: report + machine-readable summary
├── recipes/              # loop_<N>/ recipe JSONs from FE agents
├── results/              # loop_<N>/ leaderboards + ensemble results (JSON)
├── knowledge.md          # cross-loop learnings — YOU write this
├── mlflow.db             # MLflow SQLite backend
└── champion/             # final model, params, submission.csv
```

The two explainable notebooks land **outside** the (gitignored) run dir, in the tracked
`competitions/<Name>/notebooks/` folder — a durable, timestamped history logged in
`notebooks/INDEX.md`. See `references/notebook-conventions.md`.

## The loop (repeat for loop 1..N)

### Step 0 — EDA (loop 1 only, re-read every loop)

```bash
python scripts/eda.py --config config.yaml
```

Read `eda/eda_summary.json` and `eda/eda_report.md` fully. This tells you
column types, cardinalities, missingness, skew, target distribution, leakage
suspects, and correlations. Every recipe you design must be grounded in it.

### Step 1 — Parallel feature-engineering agents

Spawn 3–5 parallel subagents (Task tool). Each agent receives:
- the EDA summary
- `references/feature-recipes.md` (recipe schema + op catalog)
- `knowledge.md` (empty on loop 1; mandatory reading on loops 2+)
- a distinct **specialization** so variants diverge:

| Agent | Specialization |
|---|---|
| A | Minimal/raw: impute + encode only (tree-friendly) |
| B | Linear-friendly: scaling, log/power transforms, one-hot |
| C | Dimensionality reduction: PCA / feature selection (kbest, model-based, RFE) |
| D | Interactions & aggregates: polynomial, ratios, group stats, binning |
| E | Wildcard: target encoding, clustering features, anything knowledge.md suggests |

Each agent writes one recipe JSON to `recipes/loop_<N>/<name>.json` following
the schema in `references/feature-recipes.md`. On loops 2+, agents must
explicitly state (in the recipe's `rationale` field) which knowledge.md
learnings they are exploiting or deliberately contradicting. Validate every
recipe compiles before training:

```bash
python scripts/make_datasets.py --config config.yaml --loop <N> --validate
```

Fix any recipe that fails validation before proceeding. If the Task tool is
unavailable, design the 5 recipes yourself sequentially with the same
specializations — do not skip variant diversity.

### Step 2 — Algorithm↔dataset mapping + baseline training

Read `references/algorithm-dataset-map.md`. The training script enforces the
default mapping (trees on raw-ish variants, linear/SVM/KNN only on scaled
variants), but you may override pairings via the recipe's `suggested_models`
field when EDA or knowledge.md justifies it.

```bash
python scripts/train_baselines.py --config config.yaml --loop <N>
```

This cross-validates every valid (dataset variant × model) pair with sensible
default hyperparameters and logs everything to MLflow under experiment
`<competition>/loop_<N>`. It writes `results/loop_<N>/leaderboard.json`.

### Step 3 — Hyperparameter optimization (Optuna)

```bash
python scripts/optimize.py --config config.yaml --loop <N>
```

Takes the top-k pairs from the leaderboard (default k=3), runs Optuna studies
with per-family search spaces, logs every trial as a nested MLflow run, and
appends optimized results to the leaderboard.

### Step 4 — Ensembling

```bash
python scripts/ensemble.py --config config.yaml --loop <N>
```

Builds three ensembles from the top-m optimized models (default m=5):
soft/hard voting, weight-optimized blending (scipy-minimized on OOF
predictions), and stacking with a regularized meta-learner. All CV'd honestly
on out-of-fold predictions and logged to MLflow.

### Step 5 — Knowledge distillation (YOUR job, not a script's)

Read `results/loop_<N>/leaderboard.json` and `results/loop_<N>/ensembles.json`.
Append a dated section to `knowledge.md` covering:

- **What won and why you think it won** (dataset variant, model family, params)
- **What lost** — recipes/ops to avoid or modify
- **Param regions** Optuna converged toward (seed next loop's spaces)
- **Ensemble insight** — did diversity help? which base models correlated?
- **Hypotheses for next loop** — 2–3 concrete recipe ideas

Be specific and quantitative ("target-encoded `city` +0.014 AUC over one-hot"),
never vague ("feature engineering helped"). This file is the entire mechanism
by which loop N+1 outperforms loop N — write it like you'll be graded on it.

Then increment the loop and return to Step 1.

## After the final loop — champion selection

```bash
python scripts/select_champion.py --config config.yaml
```

Two-stage finish designed to defeat CV overfitting:

1. **Finalist round on the holdout.** The top `n_finalists` candidates across
   ALL loops (models and ensembles) are each fit on the dev split and scored
   on the untouched holdout — data no loop, no Optuna trial, and no ensemble
   ever saw. The holdout winner is the champion. With 10 adaptive loops all
   selecting on the same CV folds, CV rank inflates; the holdout is the tiebreak
   that reflects leaderboard reality.
2. **Final refit on the ENTIRE dataset.** The winning configuration
   (recipe + model + params, or full ensemble) is retrained on 100% of the
   training data (dev + holdout) so the deployed model wastes nothing. Saved to
   `champion/model.joblib` with full provenance in `champion/champion.json`
   (CV score, holdout scores for every finalist, refit row counts). If
   `test.csv` exists, writes `champion/submission.csv`.

## After champion selection — build the explainable notebooks

```bash
python scripts/build_notebooks.py --config config.yaml --which both
```

This is **not optional** — every run ends here. It generates two executed, git-tracked
Jupyter notebooks into `competitions/<Name>/notebooks/` (a new timestamped pair per run,
logged in `notebooks/INDEX.md`):

- `eda_<competition>_<ts>.ipynb` — seaborn EDA with survival-signal charts for the
  engineered features.
- `champion_<competition>_<ts>.ipynb` — what the MLflow experiments taught, the score
  trajectory, and how the champion reads the data (permutation + native importance).

Both carry a top-of-notebook links block (datasets, config, run artifacts) and the MLflow
UI command so JJ can jump from notebook → tracker. Requires the notebook deps in
`requirements.txt` and a registered `python3` Jupyter kernel. See
`references/notebook-conventions.md`. On champion→submission handoff, add the run's two
notebook names to the `SUBMISSIONS.md` row.

Report to the user: champion identity, CV score trajectory across loops (did
the knowledge loop actually improve things?), the **paths to both generated notebooks**,
and the MLflow UI command
(`mlflow ui --backend-store-uri sqlite:///kaggle_run/mlflow.db`).

## Non-negotiables

- Every transformation lives inside an sklearn Pipeline so CV never leaks.
  Never materialize target-encoded or scaled data outside a fold.
- One fixed CV splitter (seeded, stratified for classification) for the entire
  run — comparisons across loops are meaningless otherwise. `cv_folds: auto`
  adapts folds/repeats to dataset size (small data gets 10-fold repeated CV for
  a stabler estimate; >100k rows gets 3-fold) but stays fixed once the run starts.
- The holdout split is sacred: no loop step may read it. It exists solely so
  the final champion pick is validated on data the 10-loop search never saw.
- Everything goes to MLflow: params, CV mean/std, per-fold scores, recipe JSON
  as artifact, tags `loop`, `variant`, `model_family`, `stage`.
- Respect compute budgets in config (`optuna_trials`, `optuna_timeout_s`,
  `max_train_seconds_per_fit`). Skip a model that exceeds budget; log the skip.
- If a loop's best score fails to beat the previous loop's, that's signal, not
  failure — record it in knowledge.md and pivot strategies (the wildcard agent
  should get more aggressive).

## Reference files

- `references/feature-recipes.md` — recipe JSON schema, full op catalog, examples. Read before designing any recipe.
- `references/algorithm-dataset-map.md` — model zoo, default params, which models pair with which variant types, Optuna search spaces rationale.
- `references/mlflow-conventions.md` — naming, tags, and how to query results.
- `references/notebook-conventions.md` — the two explainable notebooks: what they contain, naming, where they live, and how `build_notebooks.py` generates them.
