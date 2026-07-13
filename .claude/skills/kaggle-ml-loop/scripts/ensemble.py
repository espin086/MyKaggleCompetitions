"""Ensemble the top-m models of a loop three ways:

1. voting   — soft voting (classification) / mean (regression), rebuilt + CV'd
2. blending — convex weights optimized on OOF predictions (scipy SLSQP)
3. stacking — StackingClassifier/Regressor with regularized meta-learner, CV'd

Writes results/loop_<N>/ensembles.json and logs to MLflow.

Run: python scripts/ensemble.py --config config.yaml --loop 1
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import minimize
from sklearn.pipeline import Pipeline

from optimize import search_space  # reused only for param reconstruction below
from train_baselines import model_zoo
from utils import (compile_recipe, detect_task, encode_target_if_needed, get_cv,
                   load_config, load_recipes, load_train, resolve_metric, run_dir,
                   save_json, setup_mlflow)


def rebuild_estimator(row: dict, task: str, rs: int, n_jobs: int):
    """Recreate the estimator for a leaderboard row (baseline defaults or optuna params)."""
    zoo = model_zoo(task, rs, n_jobs)
    est = zoo[row["model"]]["est"]
    if row.get("params"):
        valid = {k: v for k, v in row["params"].items() if k in est.get_params()}
        est.set_params(**valid)
    return est


def oof_score(oof: np.ndarray, y, metric: str, task: str) -> float:
    from sklearn import metrics as M
    if metric == "roc_auc":
        return float(M.roc_auc_score(y, oof))
    if metric == "accuracy":
        return float(M.accuracy_score(y, (oof > 0.5).astype(int)))
    if metric == "f1":
        return float(M.f1_score(y, (oof > 0.5).astype(int)))
    if metric == "neg_log_loss":
        return float(-M.log_loss(y, np.clip(oof, 1e-7, 1 - 1e-7)))
    if metric == "neg_root_mean_squared_error":
        return float(-np.sqrt(M.mean_squared_error(y, oof)))
    if metric == "neg_mean_absolute_error":
        return float(-M.mean_absolute_error(y, oof))
    if metric == "r2":
        return float(M.r2_score(y, oof))
    raise ValueError(f"OOF scoring not implemented for {metric}")


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
    recipes = load_recipes(cfg, args.loop)
    mlflow = setup_mlflow(cfg, args.loop)

    res_dir = run_dir(cfg) / "results" / f"loop_{args.loop}"
    lb = json.loads((res_dir / "leaderboard.json").read_text())
    m = cfg["ensembling"]["top_m_models"]
    # Dedupe: best row per (variant, model), then take top-m
    seen, top = set(), []
    for r in lb["results"]:
        key = (r["variant"], r["model"])
        if key not in seen:
            seen.add(key)
            top.append(r)
        if len(top) == m:
            break

    named = []
    for r in top:
        pipe = Pipeline([("pre", compile_recipe(recipes[r["variant"]], X, rs)),
                         ("model", rebuild_estimator(r, task, rs, n_jobs))])
        named.append((r["pair"].replace("__optuna", "-opt").replace("__", "-"), pipe, r))

    out = {"metric": metric, "base_models": [r["pair"] for _, _, r in named], "ensembles": []}

    from sklearn.model_selection import cross_val_score

    # 1. Voting -----------------------------------------------------------
    if "voting" in cfg["ensembling"]["methods"] and len(named) >= 2:
        if task == "classification":
            from sklearn.ensemble import VotingClassifier
            ens = VotingClassifier([(n, p) for n, p, _ in named], voting="soft", n_jobs=1)
        else:
            from sklearn.ensemble import VotingRegressor
            ens = VotingRegressor([(n, p) for n, p, _ in named], n_jobs=1)
        scores = cross_val_score(ens, X, y, cv=cv, scoring=metric, n_jobs=1)
        out["ensembles"].append({"method": "voting", "cv_mean": float(scores.mean()),
                                 "cv_std": float(scores.std())})
        print(f"ENS   voting: {scores.mean():.5f} ±{scores.std():.5f}")

    # 2. OOF weight-optimized blending -------------------------------------
    oof_dir = res_dir / "oof"
    oofs, oof_names = [], []
    for _, _, r in named:
        f = oof_dir / f"{r['pair']}.npy"
        if f.exists():
            oofs.append(np.load(f))
            oof_names.append(r["pair"])
    if "blending" in cfg["ensembling"]["methods"] and len(oofs) >= 2:
        P = np.column_stack(oofs)

        def neg_score(w):
            w = np.abs(w) / np.abs(w).sum()
            return -oof_score(P @ w, y, metric, task)

        w0 = np.ones(P.shape[1]) / P.shape[1]
        res = minimize(neg_score, w0, method="SLSQP", bounds=[(0, 1)] * P.shape[1],
                       constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
        w = np.abs(res.x) / np.abs(res.x).sum()
        score = oof_score(P @ w, y, metric, task)
        corr = np.corrcoef(P.T).round(3).tolist() if P.shape[1] > 1 else []
        out["ensembles"].append({"method": "blending", "cv_mean": score,
                                 "weights": dict(zip(oof_names, w.round(4).tolist())),
                                 "oof_correlation": corr})
        print(f"ENS   blending (OOF-optimized weights): {score:.5f}")

    # 3. Stacking -----------------------------------------------------------
    if "stacking" in cfg["ensembling"]["methods"] and len(named) >= 2:
        if task == "classification":
            from sklearn.ensemble import StackingClassifier
            from sklearn.linear_model import LogisticRegression
            ens = StackingClassifier([(n, p) for n, p, _ in named],
                                     final_estimator=LogisticRegression(C=1.0, max_iter=3000),
                                     cv=5, n_jobs=1)
        else:
            from sklearn.ensemble import StackingRegressor
            from sklearn.linear_model import Ridge
            ens = StackingRegressor([(n, p) for n, p, _ in named],
                                    final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=1)
        scores = cross_val_score(ens, X, y, cv=cv, scoring=metric, n_jobs=1)
        out["ensembles"].append({"method": "stacking", "cv_mean": float(scores.mean()),
                                 "cv_std": float(scores.std())})
        print(f"ENS   stacking: {scores.mean():.5f} ±{scores.std():.5f}")

    with mlflow.start_run(run_name=f"loop_{args.loop}_ensembles"):
        mlflow.set_tags({"loop": args.loop, "stage": "ensemble"})
        for e in out["ensembles"]:
            mlflow.log_metric(f"{e['method']}_cv_mean", e["cv_mean"])
        mlflow.log_dict(out, "ensembles.json")
    save_json(res_dir / "ensembles.json", out)
    best = max(out["ensembles"], key=lambda e: e["cv_mean"], default=None)
    print(f"\nBest ensemble: {best['method'] if best else 'none'} "
          f"({best['cv_mean']:.5f})" if best else "")


if __name__ == "__main__":
    main()
