# Notebook Conventions

Every run ends by generating two explainable Jupyter notebooks with
`scripts/build_notebooks.py`. This is the human-facing companion to the JSON/MLflow
pipeline: an EDA notebook you can *see* and a champion notebook that *explains* the run.

## What gets built

| Notebook | Template | Purpose |
|---|---|---|
| `eda_<competition>_<YYYYMMDD-HHMM>.ipynb` | `assets/notebook_templates/eda_template.py` | Seaborn EDA: target balance, univariate/bivariate, correlation, engineered-feature survival previews, auto-filled takeaways. |
| `champion_<competition>_<YYYYMMDD-HHMM>.ipynb` | `assets/notebook_templates/champion_template.py` | Executive summary + score trajectory, what the MLflow experiments taught, what worked/didn't (from `knowledge.md`), how the champion reads the data (permutation + native importance), holdout finalist table, reproduce steps. |

## Where they live

`competitions/<Name>/notebooks/` — **git-tracked** (not a run artifact). Every run produces a
**new timestamped pair**; notebooks are never overwritten, so the folder is a durable history
of every attempt. `notebooks/INDEX.md` logs one row per run (timestamp, both notebook links,
champion kind, CV/holdout scores, the MLflow UI command).

## How they're built

- Assembled from the template specs via `nbformat`, then **executed** with
  `nbconvert`'s `ExecutePreprocessor` (`allow_errors=True`) so charts and MLflow tables render
  inline. A failing cell is kept with its error rather than aborting the run.
- The generator injects a first "parameters" cell (paths, target, id, metric, MLflow URI,
  seed, holdout fraction). Template code cells reference those globals, so a notebook is
  self-contained and re-runnable from Jupyter.
- Requires `seaborn matplotlib nbformat nbconvert jupyter ipykernel` in the competition venv
  and a registered `python3` kernel (`python -m ipykernel install --user --name python3`).

## Required content (both notebooks)

- **A links block** at the top: clickable relative links to the train/test CSVs, `config.yaml`,
  the run dir, and the relevant `eda/` or `champion/` artifacts — plus the **MLflow UI**
  command (`mlflow ui --backend-store-uri sqlite:///<run_dir>/mlflow.db`) and the
  `http://127.0.0.1:5000` URL so you can jump from notebook to the experiment tracker.
- Professional styling: one shared `sns.set_theme(style="whitegrid", context="talk")`,
  consistent palette, titled/labeled axes, `despine`.

## Editing

Change what every notebook contains by editing the template specs
(`assets/notebook_templates/*.py`) — the generator stays generic. Keep template code cells
defensive (guard on column/artifact presence) so the same templates work across competitions.

## Handoff

`SUBMISSIONS.md` rows reference the run's notebook pair (the same names logged in `INDEX.md`),
so a scored submission links straight to the EDA and champion notebooks that produced it.
