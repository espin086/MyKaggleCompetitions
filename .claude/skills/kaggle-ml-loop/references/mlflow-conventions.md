# MLflow Conventions

Tracking URI: `sqlite:///<run_dir>/mlflow.db` (SQLite) unless `mlflow.tracking_uri` is set.
UI: `mlflow ui --backend-store-uri sqlite:///kaggle_run/mlflow.db`

## Structure

- Experiment per loop: `<competition_name>/loop_<N>`
- Parent run `loop_<N>_baselines` with one **nested run per (variant × model)** pair, named `<variant>__<model>`
- Optuna runs: `<variant>__<model>__optuna`, stage tag `optimized`
- Ensemble run: `loop_<N>_ensembles`

## Tags (query on these)

| tag | values |
|---|---|
| `loop` | 1..N |
| `variant` | recipe name |
| `model_family` | zoo key (hist_gbm, logreg, ...) |
| `stage` | baseline \| optimized \| ensemble |

## Metrics

`cv_mean` (primary — always the config metric, higher is better since sklearn
neg_* conventions are used), `cv_std`, `fold_<i>`, `fit_seconds`. Optimized runs
also log `baseline_cv_mean` and `n_trials` so improvement-per-trial is queryable.

## Artifacts

Every model run logs its `recipe.json` — full provenance from any MLflow run
back to the exact preprocessing that produced it.

## Useful queries (Python)

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///<run_dir>/mlflow.db")
runs = mlflow.search_runs(
    experiment_names=[f"comp/loop_{i}" for i in range(1, 11)],
    filter_string="tags.stage = 'optimized'",
    order_by=["metrics.cv_mean DESC"], max_results=10)
```

Note: `select_champion.py` reads the results JSONs rather than querying MLflow,
so champion selection works even if the tracking store is relocated. MLflow is
the audit trail; the JSONs are the pipeline contract.
