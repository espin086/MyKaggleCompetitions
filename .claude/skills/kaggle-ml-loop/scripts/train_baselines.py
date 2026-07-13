"""Baseline training: CV every valid (dataset variant × model) pair with
sensible defaults, log to MLflow, write results/loop_<N>/leaderboard.json.

Run: python scripts/train_baselines.py --config config.yaml --loop 1
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.pipeline import Pipeline

from utils import (compile_recipe, detect_task, encode_target_if_needed, get_cv,
                   load_config, load_recipes, load_train, resolve_metric, run_dir,
                   save_json, setup_mlflow)

# ------------------------------------------------------------------ model zoo

def model_zoo(task: str, rs: int, n_jobs: int) -> dict[str, dict]:
    """name -> {estimator, needs_scaling: bool}. Sensible non-default defaults."""
    if task == "classification":
        from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                                      HistGradientBoostingClassifier, RandomForestClassifier)
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        return {
            "hist_gbm": {"est": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                               early_stopping=True, random_state=rs),
                         "needs_scaling": False},
            "random_forest": {"est": RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                            n_jobs=n_jobs, random_state=rs),
                              "needs_scaling": False},
            "extra_trees": {"est": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                                        n_jobs=n_jobs, random_state=rs),
                            "needs_scaling": False},
            "gbm": {"est": GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                                      subsample=0.9, random_state=rs),
                    "needs_scaling": False},
            "logreg": {"est": LogisticRegression(C=1.0, max_iter=3000, n_jobs=n_jobs),
                       "needs_scaling": True},
            "svm_rbf": {"est": SVC(C=1.0, probability=True, random_state=rs),
                        "needs_scaling": True, "max_rows": 20000},
            "knn": {"est": KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=n_jobs),
                    "needs_scaling": True},
            "naive_bayes": {"est": GaussianNB(), "needs_scaling": True},
        }
    from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                                  HistGradientBoostingRegressor, RandomForestRegressor)
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    return {
        "hist_gbm": {"est": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                                          early_stopping=True, random_state=rs),
                     "needs_scaling": False},
        "random_forest": {"est": RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                                       n_jobs=n_jobs, random_state=rs),
                          "needs_scaling": False},
        "extra_trees": {"est": ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2,
                                                   n_jobs=n_jobs, random_state=rs),
                        "needs_scaling": False},
        "gbm": {"est": GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                                 subsample=0.9, random_state=rs),
                "needs_scaling": False},
        "ridge": {"est": Ridge(alpha=1.0), "needs_scaling": True},
        "elasticnet": {"est": ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000),
                       "needs_scaling": True},
        "svr_rbf": {"est": SVR(C=1.0), "needs_scaling": True, "max_rows": 20000},
        "knn": {"est": KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=n_jobs),
                "needs_scaling": True},
    }


def recipe_is_scaled(recipe: dict) -> bool:
    return any(o["op"] in ("scale", "quantile_transform", "power_transform", "pca")
               for o in recipe["ops"])


def pairs_for(recipes: dict, zoo: dict) -> list[tuple[str, str]]:
    """Default algorithm↔dataset mapping (overridable via recipe['suggested_models'])."""
    out = []
    for rname, recipe in recipes.items():
        suggested = recipe.get("suggested_models")
        scaled = recipe_is_scaled(recipe)
        for mname, spec in zoo.items():
            if suggested is not None:
                if mname in suggested:
                    out.append((rname, mname))
                continue
            if spec["needs_scaling"] and not scaled:
                continue  # never feed unscaled data to distance/linear models
            out.append((rname, mname))
    return out


def cross_validate_pair(pipe, X, y, cv, metric, task, budget_s):
    """Manual CV loop so we can capture OOF predictions and enforce time budget.
    OOF is accumulated and averaged so RepeatedKFold (adaptive CV on small
    data) yields one averaged prediction per sample."""
    from sklearn.base import clone
    from sklearn.metrics import get_scorer
    scorer = get_scorer(metric)
    oof_sum = np.zeros(len(y), dtype=float)
    oof_cnt = np.zeros(len(y), dtype=float)
    fold_scores, t0 = [], time.time()
    for tr, va in cv.split(X, y):
        m = clone(pipe)
        ft = time.time()
        m.fit(X.iloc[tr], y.iloc[tr])
        if time.time() - ft > budget_s:
            raise TimeoutError(f"fold fit exceeded {budget_s}s")
        fold_scores.append(scorer(m, X.iloc[va], y.iloc[va]))
        if task == "classification" and hasattr(m, "predict_proba"):
            proba = m.predict_proba(X.iloc[va])
            oof_sum[va] += proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
        else:
            oof_sum[va] += m.predict(X.iloc[va])
        oof_cnt[va] += 1
    oof = np.divide(oof_sum, oof_cnt, out=np.zeros_like(oof_sum), where=oof_cnt > 0)
    return float(np.mean(fold_scores)), float(np.std(fold_scores)), fold_scores, oof, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--loop", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    X, y = load_train(cfg)
    task = detect_task(cfg, y)
    y, _ = encode_target_if_needed(y, task)
    metric = resolve_metric(cfg, task)
    cv = get_cv(cfg, task, y)
    rs = cfg["evaluation"]["random_state"]
    zoo = model_zoo(task, rs, cfg["training"]["n_jobs"])
    recipes = load_recipes(cfg, args.loop)
    mlflow = setup_mlflow(cfg, args.loop)
    budget = cfg["training"]["max_train_seconds_per_fit"]
    oof_dir = run_dir(cfg) / "results" / f"loop_{args.loop}" / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with mlflow.start_run(run_name=f"loop_{args.loop}_baselines"):
        mlflow.set_tags({"loop": args.loop, "stage": "baseline_parent"})
        mlflow.log_params({
            "cv_splitter": type(cv).__name__, "cv_total_splits": cv.get_n_splits(),
            "metric": metric, "task": task, "dev_rows": len(X),
            "holdout_fraction": cfg["evaluation"].get("holdout_fraction", 0.2),
            "n_recipes": len(recipes)})
        for rname, mname in pairs_for(recipes, zoo):
            spec = zoo[mname]
            if spec.get("max_rows") and len(X) > spec["max_rows"]:
                print(f"SKIP  {rname} × {mname}: {len(X)} rows > {spec['max_rows']} cap")
                continue
            pipe = Pipeline([("pre", compile_recipe(recipes[rname], X, rs)), ("model", spec["est"])])
            tag = f"{rname}__{mname}"
            try:
                mean, std, folds, oof, secs = cross_validate_pair(pipe, X, y, cv, metric, task, budget)
            except Exception as e:
                print(f"FAIL  {tag}: {type(e).__name__}: {e}")
                continue
            np.save(oof_dir / f"{tag}.npy", oof)
            with mlflow.start_run(run_name=tag, nested=True):
                mlflow.set_tags({"loop": args.loop, "variant": rname, "model_family": mname,
                                 "stage": "baseline"})
                mlflow.log_params({f"model__{k}": v for k, v in spec["est"].get_params().items()
                                   if isinstance(v, (int, float, str, bool, type(None)))})
                mlflow.log_metrics({"cv_mean": mean, "cv_std": std, "fit_seconds": secs,
                                    **{f"fold_{i}": s for i, s in enumerate(folds)}})
                mlflow.log_dict(recipes[rname], "recipe.json")
            rows.append({"pair": tag, "variant": rname, "model": mname, "stage": "baseline",
                         "cv_mean": mean, "cv_std": std, "seconds": round(secs, 1)})
            print(f"DONE  {tag}: {metric}={mean:.5f} ±{std:.5f} ({secs:.0f}s)")

    rows.sort(key=lambda r: r["cv_mean"], reverse=True)
    save_json(run_dir(cfg) / "results" / f"loop_{args.loop}" / "leaderboard.json",
              {"metric": metric, "task": task, "loop": args.loop, "results": rows})
    print(f"\nLeaderboard written; best = {rows[0]['pair'] if rows else 'none'}")


if __name__ == "__main__":
    main()
