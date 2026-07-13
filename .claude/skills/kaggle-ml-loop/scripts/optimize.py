"""Optuna hyperparameter optimization for the top-k baseline pairs.

Reads results/loop_<N>/leaderboard.json, optimizes each of the top-k pairs,
logs every trial to MLflow, appends 'optimized' rows to the leaderboard and
saves best OOF predictions for ensembling.

Run: python scripts/optimize.py --config config.yaml --loop 1
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import optuna
from sklearn.pipeline import Pipeline

from train_baselines import cross_validate_pair, model_zoo
from utils import (compile_recipe, detect_task, encode_target_if_needed, get_cv,
                   load_config, load_recipes, load_train, resolve_metric, run_dir,
                   save_json, setup_mlflow)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def search_space(trial: optuna.Trial, family: str, task: str, rs: int, n_jobs: int):
    c = task == "classification"
    if family == "hist_gbm":
        from sklearn.ensemble import (HistGradientBoostingClassifier,
                                      HistGradientBoostingRegressor)
        P = dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 max_iter=trial.suggest_int("max_iter", 100, 800),
                 max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 127),
                 min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 60),
                 l2_regularization=trial.suggest_float("l2_regularization", 1e-8, 10, log=True),
                 early_stopping=True, random_state=rs)
        return (HistGradientBoostingClassifier if c else HistGradientBoostingRegressor)(**P)
    if family in ("random_forest", "extra_trees"):
        from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                                      RandomForestClassifier, RandomForestRegressor)
        P = dict(n_estimators=trial.suggest_int("n_estimators", 200, 800),
                 max_depth=trial.suggest_int("max_depth", 4, 30),
                 min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
                 max_features=trial.suggest_float("max_features", 0.2, 1.0),
                 n_jobs=n_jobs, random_state=rs)
        cls = {("random_forest", True): RandomForestClassifier,
               ("random_forest", False): RandomForestRegressor,
               ("extra_trees", True): ExtraTreesClassifier,
               ("extra_trees", False): ExtraTreesRegressor}[(family, c)]
        return cls(**P)
    if family == "gbm":
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        P = dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 n_estimators=trial.suggest_int("n_estimators", 100, 600),
                 max_depth=trial.suggest_int("max_depth", 2, 8),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 30),
                 random_state=rs)
        return (GradientBoostingClassifier if c else GradientBoostingRegressor)(**P)
    if family == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=trial.suggest_float("C", 1e-3, 100, log=True),
                                  l1_ratio=trial.suggest_float("l1_ratio", 0, 1),
                                  penalty="elasticnet", solver="saga", max_iter=4000, n_jobs=n_jobs)
    if family in ("ridge", "elasticnet"):
        from sklearn.linear_model import ElasticNet, Ridge
        if family == "ridge":
            return Ridge(alpha=trial.suggest_float("alpha", 1e-3, 100, log=True))
        return ElasticNet(alpha=trial.suggest_float("alpha", 1e-4, 10, log=True),
                          l1_ratio=trial.suggest_float("l1_ratio", 0, 1), max_iter=8000)
    if family in ("svm_rbf", "svr_rbf"):
        from sklearn.svm import SVC, SVR
        C = trial.suggest_float("C", 1e-2, 100, log=True)
        gamma = trial.suggest_float("gamma", 1e-4, 1, log=True)
        return SVC(C=C, gamma=gamma, probability=True, random_state=rs) if c else SVR(C=C, gamma=gamma)
    if family == "knn":
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        P = dict(n_neighbors=trial.suggest_int("n_neighbors", 3, 60),
                 weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
                 p=trial.suggest_int("p", 1, 2), n_jobs=n_jobs)
        return (KNeighborsClassifier if c else KNeighborsRegressor)(**P)
    if family == "naive_bayes":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB(var_smoothing=trial.suggest_float("var_smoothing", 1e-11, 1e-6, log=True))
    raise ValueError(f"No search space for {family}")


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
    n_jobs = cfg["training"]["n_jobs"]
    budget = cfg["training"]["max_train_seconds_per_fit"]
    recipes = load_recipes(cfg, args.loop)
    mlflow = setup_mlflow(cfg, args.loop)

    lb_path = run_dir(cfg) / "results" / f"loop_{args.loop}" / "leaderboard.json"
    lb = json.loads(lb_path.read_text())
    top = [r for r in lb["results"] if r["stage"] == "baseline"][: cfg["optimization"]["top_k_models"]]
    oof_dir = lb_path.parent / "oof"

    for row in top:
        rname, family = row["variant"], row["model"]
        tag = f"{rname}__{family}__optuna"
        pre = compile_recipe(recipes[rname], X, rs)
        best = {"score": -np.inf, "params": None, "oof": None}

        def objective(trial: optuna.Trial) -> float:
            est = search_space(trial, family, task, rs, n_jobs)
            pipe = Pipeline([("pre", pre), ("model", est)])
            try:
                mean, std, _, oof, _ = cross_validate_pair(pipe, X, y, cv, metric, task, budget)
            except Exception:
                raise optuna.TrialPruned()
            if mean > best["score"]:
                best.update(score=mean, std=std, params=trial.params, oof=oof)
            return mean

        with mlflow.start_run(run_name=tag):
            mlflow.set_tags({"loop": args.loop, "variant": rname, "model_family": family,
                             "stage": "optimized"})
            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=rs))
            study.optimize(objective, n_trials=cfg["optimization"]["optuna_trials"],
                           timeout=cfg["optimization"]["optuna_timeout_s"])
            mlflow.log_params(best["params"] or {})
            mlflow.log_metrics({"cv_mean": best["score"], "cv_std": best.get("std", 0.0),
                                "n_trials": len(study.trials),
                                "baseline_cv_mean": row["cv_mean"]})
            mlflow.log_dict(recipes[rname], "recipe.json")
        if best["oof"] is not None:
            np.save(oof_dir / f"{tag}.npy", best["oof"])
        lb["results"].append({"pair": tag, "variant": rname, "model": family,
                              "stage": "optimized", "cv_mean": best["score"],
                              "cv_std": best.get("std", 0.0), "params": best["params"]})
        print(f"OPT   {tag}: {row['cv_mean']:.5f} -> {best['score']:.5f} "
              f"({len(study.trials)} trials)")

    lb["results"].sort(key=lambda r: r["cv_mean"], reverse=True)
    save_json(lb_path, lb)
    print("Leaderboard updated with optimized results.")


if __name__ == "__main__":
    main()
