"""Shared utilities for kaggle-ml-loop: config, data, CV, MLflow, recipe compiler."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------- config/data

def load_config(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["loop"]["run_dir"] = str(Path(cfg["loop"]["run_dir"]).resolve())
    return cfg


def run_dir(cfg: dict) -> Path:
    p = Path(cfg["loop"]["run_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_train(cfg: dict, split: str = "dev"):
    """Load training data. split: 'dev' (loop work), 'holdout' (final eval), 'all'."""
    df = pd.read_csv(cfg["data"]["train_path"])
    target = cfg["data"]["target"]
    idc = cfg["data"].get("id_column")
    if idc and idc in df.columns:
        df = df.drop(columns=[idc])
    y = df[target]
    X = df.drop(columns=[target])
    frac = cfg["evaluation"].get("holdout_fraction", 0.2)
    if split == "all" or not frac or frac <= 0:
        return X, y
    from sklearn.model_selection import train_test_split
    task = detect_task(cfg, y)
    strat = y if task == "classification" else None
    dev_idx, hold_idx = train_test_split(
        np.arange(len(y)), test_size=frac,
        random_state=cfg["evaluation"]["random_state"], stratify=strat)
    idx = dev_idx if split == "dev" else hold_idx
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def detect_task(cfg: dict, y: pd.Series) -> str:
    task = cfg["data"].get("task", "auto")
    if task != "auto":
        return task
    if y.dtype == object or y.dtype == bool or y.nunique() <= max(20, int(0.05 * len(y))) and y.dtype.kind in "iu":
        return "classification"
    return "regression"


def resolve_metric(cfg: dict, task: str) -> str:
    m = cfg["evaluation"].get("metric", "auto")
    if m != "auto":
        return m
    return "roc_auc" if task == "classification" else "neg_root_mean_squared_error"


def get_cv(cfg: dict, task: str, y=None):
    """CV splitter. cv_folds: 'auto' adapts folds/repeats to dataset size:
    fewer rows -> more folds (+ repeats) for a stabler estimate; huge data ->
    fewer folds since each fold is already large and compute dominates."""
    from sklearn.model_selection import (KFold, RepeatedKFold,
                                         RepeatedStratifiedKFold, StratifiedKFold)
    k_cfg = cfg["evaluation"].get("cv_folds", "auto")
    rs = cfg["evaluation"]["random_state"]
    repeats = 1
    if k_cfg in ("auto", None) and y is not None:
        n = len(y)
        if n < 500:
            k, repeats = 10, 3
        elif n < 2000:
            k, repeats = 10, 2
        elif n < 20000:
            k = 5
        elif n < 100000:
            k = 5
        else:
            k = 3
    else:
        k = int(k_cfg) if k_cfg not in ("auto", None) else 5
    if task == "classification" and y is not None:
        # can't have more folds than the minority class count
        import pandas as _pd
        k = max(2, min(k, int(_pd.Series(y).value_counts().min())))
        if repeats > 1:
            return RepeatedStratifiedKFold(n_splits=k, n_repeats=repeats, random_state=rs)
        return StratifiedKFold(n_splits=k, shuffle=True, random_state=rs)
    if repeats > 1:
        return RepeatedKFold(n_splits=k, n_repeats=repeats, random_state=rs)
    return KFold(n_splits=k, shuffle=True, random_state=rs)


def encode_target_if_needed(y: pd.Series, task: str):
    if task == "classification" and (y.dtype == object or y.dtype == bool):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        return pd.Series(le.fit_transform(y), index=y.index), le
    return y, None

# ------------------------------------------------------------------- mlflow

def setup_mlflow(cfg: dict, loop: int | None = None):
    import mlflow
    uri = cfg["mlflow"].get("tracking_uri") or f"sqlite:///{run_dir(cfg) / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    name = cfg["competition_name"] + (f"/loop_{loop}" if loop is not None else "/final")
    # Pin artifacts under run_dir (sqlite backend otherwise defaults to CWD-relative ./mlruns)
    if mlflow.get_experiment_by_name(name) is None:
        art = (run_dir(cfg) / "mlartifacts" / name.replace("/", "_")).resolve()
        art.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(name, artifact_location=f"file://{art}")
    mlflow.set_experiment(name)
    return mlflow

# ------------------------------------------------------------ recipe compiler
# A recipe is JSON: {"name", "rationale", "suggested_models": [...],
#   "ops": [{"op": <name>, ...params}]}. Ops compile to a leak-safe Pipeline.

NUMERIC_KINDS = "biufc"


def _split_cols(X: pd.DataFrame):
    num = [c for c in X.columns if X[c].dtype.kind in NUMERIC_KINDS]
    cat = [c for c in X.columns if c not in num]
    return num, cat


def compile_recipe(recipe: dict, X: pd.DataFrame, random_state: int = 42):
    """Compile a recipe dict into an sklearn preprocessing Pipeline."""
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import (RFE, SelectFromModel, SelectKBest,
                                           VarianceThreshold, f_classif, f_regression,
                                           mutual_info_classif, mutual_info_regression)
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (FunctionTransformer, KBinsDiscretizer,
                                       MinMaxScaler, OneHotEncoder, OrdinalEncoder,
                                       PolynomialFeatures, PowerTransformer,
                                       QuantileTransformer, RobustScaler, StandardScaler)

    num, cat = _split_cols(X)
    steps = []
    # Base: impute + encode (always first so downstream ops get numeric data)
    base_ops = {o["op"] for o in recipe["ops"]}
    impute_num = next((o for o in recipe["ops"] if o["op"] == "impute_numeric"), {"strategy": "median"})
    impute_cat = next((o for o in recipe["ops"] if o["op"] == "impute_categorical"), {"strategy": "most_frequent"})
    encode = next((o for o in recipe["ops"] if o["op"] in ("onehot", "ordinal", "target_encode")), {"op": "ordinal"})

    num_steps = [("impute", SimpleImputer(strategy=impute_num.get("strategy", "median")))]
    for o in recipe["ops"]:
        if o["op"] == "log1p":
            num_steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
        elif o["op"] == "power_transform":
            num_steps.append(("power", PowerTransformer(method=o.get("method", "yeo-johnson"))))
        elif o["op"] == "quantile_transform":
            num_steps.append(("quantile", QuantileTransformer(
                output_distribution=o.get("output", "normal"), random_state=random_state)))
        elif o["op"] == "scale":
            kind = o.get("kind", "standard")
            num_steps.append(("scale", {"standard": StandardScaler(), "robust": RobustScaler(),
                                        "minmax": MinMaxScaler()}[kind]))
        elif o["op"] == "bin":
            num_steps.append(("bin", KBinsDiscretizer(n_bins=o.get("n_bins", 5),
                                                      encode="ordinal", strategy=o.get("strategy", "quantile"))))

    if encode["op"] == "onehot":
        cat_enc = OneHotEncoder(handle_unknown="ignore", max_categories=encode.get("max_categories", 30),
                                sparse_output=False)
    elif encode["op"] == "target_encode":
        from sklearn.preprocessing import TargetEncoder
        cat_enc = TargetEncoder(random_state=random_state, smooth="auto")
    else:
        cat_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    cat_steps = [("impute", SimpleImputer(strategy=impute_cat.get("strategy", "most_frequent"))),
                 ("encode", cat_enc)]

    pre = ColumnTransformer(
        [("num", Pipeline(num_steps), num), ("cat", Pipeline(cat_steps), cat)],
        remainder="drop", verbose_feature_names_out=False)
    steps.append(("columns", pre))

    # Post-encoding global ops (order as listed in recipe)
    for o in recipe["ops"]:
        if o["op"] == "polynomial":
            steps.append(("poly", PolynomialFeatures(degree=o.get("degree", 2),
                                                     interaction_only=o.get("interaction_only", True),
                                                     include_bias=False)))
        elif o["op"] == "variance_threshold":
            steps.append(("varthresh", VarianceThreshold(threshold=o.get("threshold", 0.0))))
        elif o["op"] == "pca":
            steps.append(("pca", PCA(n_components=o.get("n_components", 0.95),
                                     random_state=random_state)))
        elif o["op"] == "select_kbest":
            score = o.get("score_func", "f")
            task = o.get("task", "classification")
            fn = {("f", "classification"): f_classif, ("f", "regression"): f_regression,
                  ("mi", "classification"): mutual_info_classif,
                  ("mi", "regression"): mutual_info_regression}[(score, task)]
            steps.append(("kbest", SelectKBest(fn, k=o.get("k", 20))))
        elif o["op"] == "select_from_model":
            from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
            est = (ExtraTreesClassifier(n_estimators=100, random_state=random_state)
                   if o.get("task", "classification") == "classification"
                   else ExtraTreesRegressor(n_estimators=100, random_state=random_state))
            steps.append(("sfm", SelectFromModel(est, threshold=o.get("threshold", "median"))))
        elif o["op"] == "rfe":
            from sklearn.linear_model import LogisticRegression, Ridge
            est = (LogisticRegression(max_iter=2000) if o.get("task", "classification") == "classification"
                   else Ridge())
            steps.append(("rfe", RFE(est, n_features_to_select=o.get("n_features", 20), step=0.1)))

    return Pipeline(steps)


def load_recipes(cfg: dict, loop: int) -> dict[str, dict]:
    rdir = run_dir(cfg) / "recipes" / f"loop_{loop}"
    recipes = {}
    for f in sorted(rdir.glob("*.json")):
        r = json.loads(f.read_text())
        recipes[r.get("name", f.stem)] = r
    if not recipes:
        raise FileNotFoundError(f"No recipes in {rdir} — FE agents must run first (see SKILL.md Step 1).")
    return recipes


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
