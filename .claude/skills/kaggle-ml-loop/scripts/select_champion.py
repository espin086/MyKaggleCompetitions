"""After the final loop: pick the best result across ALL loops (single models
and ensembles), refit on full training data, save model + provenance, and
generate submission.csv if test data exists.

Run: python scripts/select_champion.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from ensemble import rebuild_estimator
from utils import (compile_recipe, detect_task, encode_target_if_needed, load_config,
                   load_recipes, load_train, resolve_metric, run_dir, save_json,
                   setup_mlflow)


def gather_candidates(cfg: dict):
    """Yield every scored candidate across loops from the results JSONs."""
    rdir = run_dir(cfg) / "results"
    for loop_dir in sorted(rdir.glob("loop_*")):
        loop = int(loop_dir.name.split("_")[1])
        lb = loop_dir / "leaderboard.json"
        if lb.exists():
            for r in json.loads(lb.read_text())["results"]:
                yield {"loop": loop, "kind": "model", "cv_mean": r["cv_mean"], "row": r}
        ens = loop_dir / "ensembles.json"
        if ens.exists():
            data = json.loads(ens.read_text())
            for e in data["ensembles"]:
                yield {"loop": loop, "kind": "ensemble", "cv_mean": e["cv_mean"],
                       "row": e, "base_models": data["base_models"]}


def build_champion_estimator(cand: dict, cfg: dict, X, task: str):
    rs = cfg["evaluation"]["random_state"]
    n_jobs = cfg["training"]["n_jobs"]
    recipes = load_recipes(cfg, cand["loop"])

    def pipe_for(row):
        return Pipeline([("pre", compile_recipe(recipes[row["variant"]], X, rs)),
                         ("model", rebuild_estimator(row, task, rs, n_jobs))])

    if cand["kind"] == "model":
        return pipe_for(cand["row"])

    # Ensemble: rebuild base models from that loop's leaderboard
    lb = json.loads((run_dir(cfg) / "results" / f"loop_{cand['loop']}" /
                     "leaderboard.json").read_text())
    by_pair = {r["pair"]: r for r in lb["results"]}
    base = [(p.replace("__optuna", "-opt").replace("__", "-"), pipe_for(by_pair[p]))
            for p in cand["base_models"] if p in by_pair]
    method = cand["row"]["method"]
    if method == "voting":
        if task == "classification":
            from sklearn.ensemble import VotingClassifier
            return VotingClassifier(base, voting="soft", n_jobs=1)
        from sklearn.ensemble import VotingRegressor
        return VotingRegressor(base, n_jobs=1)
    if method == "blending":
        # Deployable blend = weighted voting with the optimized weights
        weights = [cand["row"]["weights"].get(p, 0.0) for p in cand["base_models"] if p in by_pair]
        if task == "classification":
            from sklearn.ensemble import VotingClassifier
            return VotingClassifier(base, voting="soft", weights=weights, n_jobs=1)
        from sklearn.ensemble import VotingRegressor
        return VotingRegressor(base, weights=weights, n_jobs=1)
    # stacking
    if task == "classification":
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        return StackingClassifier(base, final_estimator=LogisticRegression(C=1.0, max_iter=3000),
                                  cv=5, n_jobs=1)
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import Ridge
    return StackingRegressor(base, final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    X_dev, y_dev_raw = load_train(cfg, split="dev")
    X_hold, y_hold_raw = load_train(cfg, split="holdout")
    X_all, y_all_raw = load_train(cfg, split="all")
    task = detect_task(cfg, y_all_raw)
    metric = resolve_metric(cfg, task)
    # Encode target consistently across splits using the FULL label space
    y_all, le = encode_target_if_needed(y_all_raw, task)
    if le is not None:
        y_dev = pd.Series(le.transform(y_dev_raw))
        y_hold = pd.Series(le.transform(y_hold_raw))
    else:
        y_dev, y_hold = y_dev_raw, y_hold_raw

    cands = list(gather_candidates(cfg))
    if not cands:
        raise SystemExit("No results found — run the loop first.")
    cands.sort(key=lambda c: c["cv_mean"], reverse=True)

    # Score trajectory per loop (did the knowledge loop work?)
    trajectory = {}
    for c in cands:
        trajectory[c["loop"]] = max(trajectory.get(c["loop"], float("-inf")), c["cv_mean"])

    # ---- Finalist round: fit on dev, score on the untouched holdout ----
    from sklearn.metrics import get_scorer
    scorer = get_scorer(metric)
    n_final = cfg.get("champion", {}).get("n_finalists", 5)
    finalists, seen = [], set()
    for c in cands:
        key = (c["loop"], c["kind"],
               c["row"].get("pair") or c["row"].get("method"))
        if key in seen:
            continue
        seen.add(key)
        finalists.append(c)
        if len(finalists) == n_final:
            break

    # ---- MLflow: everything below is tracked in the '<competition>/final' experiment ----
    mlflow = setup_mlflow(cfg, loop=None)
    import mlflow as _mlf  # for mlflow.sklearn
    with mlflow.start_run(run_name="champion_selection") as parent:
        mlflow.set_tags({"stage": "champion_selection"})
        mlflow.log_params({
            "metric": metric, "task": task, "n_finalists": len(finalists),
            "dev_rows": len(X_dev), "holdout_rows": len(X_hold),
            "total_rows": len(X_all),
            "holdout_fraction": cfg["evaluation"].get("holdout_fraction", 0.2)})
        # per-loop best-CV trajectory as a step metric (chart in the UI)
        for lp, sc in sorted(trajectory.items()):
            mlflow.log_metric("best_cv_by_loop", sc, step=lp)

        holdout_results = []
        have_holdout = len(X_hold) > 0
        for c in finalists:
            label = f"loop{c['loop']}/{c['row'].get('pair') or c['row'].get('method')}"
            if not have_holdout:
                holdout_results.append({"candidate": label, "cv_mean": c["cv_mean"],
                                        "holdout": None, "_cand": c})
                continue
            try:
                est = build_champion_estimator(c, cfg, X_dev, task)
                est.fit(X_dev, y_dev)
                score = float(scorer(est, X_hold, y_hold))
            except Exception as e:
                print(f"FAIL  {label}: {type(e).__name__}: {e}")
                with mlflow.start_run(run_name=f"finalist_{label}", nested=True):
                    mlflow.set_tags({"stage": "finalist", "loop": c["loop"],
                                     "kind": c["kind"], "status": "failed"})
                continue
            with mlflow.start_run(run_name=f"finalist_{label}", nested=True):
                mlflow.set_tags({"stage": "finalist", "loop": c["loop"],
                                 "kind": c["kind"],
                                 "variant": c["row"].get("variant", "-"),
                                 "model_family": c["row"].get("model",
                                                              c["row"].get("method", "-"))})
                mlflow.log_metrics({"cv_mean_dev": c["cv_mean"],
                                    "holdout_score": score,
                                    "cv_holdout_gap": c["cv_mean"] - score})
            holdout_results.append({"candidate": label, "cv_mean": c["cv_mean"],
                                    "holdout": round(score, 5), "_cand": c})
            print(f"HOLD  {label}: cv={c['cv_mean']:.5f} holdout={score:.5f}")

        if not holdout_results:
            raise SystemExit("All finalists failed holdout evaluation.")
        key = (lambda r: r["holdout"]) if have_holdout else (lambda r: r["cv_mean"])
        holdout_results.sort(key=key, reverse=True)
        champ = holdout_results[0]["_cand"]

        # ---- Final refit: winning configuration on the ENTIRE dataset ----
        est = build_champion_estimator(champ, cfg, X_all, task)
        est.fit(X_all, y_all)

        out = run_dir(cfg) / "champion"
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(est, out / "model.joblib")
        champion_record = {
            "metric": metric, "loop": champ["loop"], "kind": champ["kind"],
            "detail": champ["row"], "cv_mean_dev": champ["cv_mean"],
            "holdout_score": holdout_results[0]["holdout"],
            "finalists": [{k: v for k, v in r.items() if k != "_cand"}
                          for r in holdout_results],
            "refit_on": f"entire training set ({len(X_all)} rows: "
                        f"{len(X_dev)} dev + {len(X_hold)} holdout)",
            "score_trajectory_by_loop": dict(sorted(trajectory.items())),
        }
        save_json(out / "champion.json", champion_record)

        # Champion nested run: full model artifact + provenance
        with mlflow.start_run(run_name="champion_final", nested=True):
            mlflow.set_tags({"stage": "champion", "loop": champ["loop"],
                             "kind": champ["kind"],
                             "variant": champ["row"].get("variant", "-"),
                             "model_family": champ["row"].get("model",
                                                              champ["row"].get("method", "-"))})
            if champ["row"].get("params"):
                mlflow.log_params(champ["row"]["params"])
            m = {"cv_mean_dev": champ["cv_mean"], "refit_rows": len(X_all)}
            if holdout_results[0]["holdout"] is not None:
                m["holdout_score"] = holdout_results[0]["holdout"]
            mlflow.log_metrics(m)
            mlflow.log_dict(champion_record, "champion.json")
            # cloudpickle serialization: MLflow 3.x's skops default rejects
            # fitted ensembles ("untrusted types": numpy.dtype, Bunch)
            kw = dict(pip_requirements=["scikit-learn", "pandas", "numpy", "scipy"],
                      serialization_format=_mlf.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
            try:
                _mlf.sklearn.log_model(est, name="model", **kw)
            except TypeError:
                _mlf.sklearn.log_model(est, artifact_path="model", **kw)
        mlflow.log_metric("champion_holdout_score",
                          holdout_results[0]["holdout"] or champ["cv_mean"])

    test_path = cfg["data"].get("test_path")
    if test_path:
        try:
            test = pd.read_csv(test_path)
            idc = cfg["data"].get("id_column")
            ids = test[idc] if idc and idc in test.columns else pd.Series(range(len(test)), name="id")
            Xt = test.drop(columns=[c for c in [idc, cfg["data"]["target"]] if c in test.columns])
            if task == "classification" and hasattr(est, "predict_proba"):
                proba = est.predict_proba(Xt)
                pred = proba[:, 1] if proba.shape[1] == 2 else est.predict(Xt)
            else:
                pred = est.predict(Xt)
            if le is not None and pred.ndim == 1 and pred.dtype.kind in "iu":
                pred = le.inverse_transform(pred)
            pd.DataFrame({ids.name or "id": ids, cfg["data"]["target"]: pred}) \
                .to_csv(out / "submission.csv", index=False)
            print(f"submission.csv written ({len(ids)} rows)")
        except FileNotFoundError:
            print("No test file found; skipping submission.")

    hs = holdout_results[0]["holdout"]
    print(f"\nCHAMPION: loop {champ['loop']} {champ['kind']} — "
          f"cv(dev)={champ['cv_mean']:.5f}"
          + (f" | holdout={hs:.5f}" if hs is not None else "")
          + f" | refit on all {len(X_all)} rows")
    print("Trajectory:", dict(sorted(trajectory.items())))
    print(f"Model saved to {out / 'model.joblib'}")


if __name__ == "__main__":
    main()
