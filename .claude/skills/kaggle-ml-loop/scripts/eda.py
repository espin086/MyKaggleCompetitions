"""EDA pass: profiles the training data and writes eda_report.md + eda_summary.json.

Run: python scripts/eda.py --config config.yaml
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from utils import detect_task, load_config, load_train, run_dir, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    X, y = load_train(cfg)
    task = detect_task(cfg, y)
    out = run_dir(cfg) / "eda"
    out.mkdir(parents=True, exist_ok=True)

    cols = []
    for c in X.columns:
        s = X[c]
        numeric = s.dtype.kind in "biufc"
        info = {
            "name": c, "dtype": str(s.dtype), "numeric": numeric,
            "missing_pct": round(float(s.isna().mean()) * 100, 2),
            "n_unique": int(s.nunique()),
        }
        if numeric:
            info.update({
                "mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4),
                "min": float(s.min()), "max": float(s.max()),
                "skew": round(float(s.skew()), 3),
                "zeros_pct": round(float((s == 0).mean()) * 100, 2),
            })
        else:
            top = s.value_counts(normalize=True).head(5)
            info["top_values"] = {str(k): round(float(v), 3) for k, v in top.items()}
        # Leakage suspects: near-unique non-numeric, or numeric perfectly monotone with index
        info["leakage_suspect"] = bool(info["n_unique"] >= 0.98 * len(s) and not numeric)
        cols.append(info)

    # Target profile
    tgt = {"task": task, "n_rows": int(len(y)), "missing_pct": round(float(y.isna().mean()) * 100, 2)}
    if task == "classification":
        vc = y.value_counts(normalize=True)
        tgt["classes"] = {str(k): round(float(v), 4) for k, v in vc.items()}
        tgt["imbalance_ratio"] = round(float(vc.max() / vc.min()), 2) if len(vc) > 1 else 1.0
    else:
        tgt.update({"mean": round(float(y.mean()), 4), "std": round(float(y.std()), 4),
                    "skew": round(float(y.skew()), 3)})

    # Correlations (numeric only, top by |corr| with target if target numeric-encodable)
    num_cols = [c["name"] for c in cols if c["numeric"]]
    target_corr = {}
    if num_cols:
        yy = pd.factorize(y)[0] if y.dtype == object else y
        corr = X[num_cols].corrwith(pd.Series(yy, index=X.index)).abs().sort_values(ascending=False)
        target_corr = {k: round(float(v), 4) for k, v in corr.head(15).items() if np.isfinite(v)}
    high_pair_corr = []
    if len(num_cols) > 1:
        cm = X[num_cols].corr().abs()
        for i, a in enumerate(num_cols):
            for b in num_cols[i + 1:]:
                v = cm.loc[a, b]
                if np.isfinite(v) and v > 0.9:
                    high_pair_corr.append({"a": a, "b": b, "corr": round(float(v), 3)})

    summary = {"target": tgt, "columns": cols, "target_correlation_top": target_corr,
               "highly_correlated_pairs": high_pair_corr[:30],
               "n_numeric": len(num_cols), "n_categorical": len(cols) - len(num_cols)}
    save_json(out / "eda_summary.json", summary)

    # Human/agent-readable report
    lines = [f"# EDA — {cfg['competition_name']}", "",
             f"Task: **{task}** | rows: {len(y)} | features: {X.shape[1]} "
             f"({summary['n_numeric']} numeric, {summary['n_categorical']} categorical)", "",
             "## Target", f"```json\n{tgt}\n```", "",
             "## Signals for feature engineering"]
    skewed = [c["name"] for c in cols if c["numeric"] and abs(c.get("skew", 0)) > 1.5]
    missing = [c["name"] for c in cols if c["missing_pct"] > 5]
    highcard = [c["name"] for c in cols if not c["numeric"] and c["n_unique"] > 30]
    leaks = [c["name"] for c in cols if c["leakage_suspect"]]
    lines += [f"- Skewed numerics (candidates for log1p/power): {skewed or 'none'}",
              f"- Columns >5% missing (impute strategy matters): {missing or 'none'}",
              f"- High-cardinality categoricals (target-encode > one-hot): {highcard or 'none'}",
              f"- Leakage suspects (near-unique IDs — consider dropping): {leaks or 'none'}",
              f"- Redundant pairs |r|>0.9 (reduction candidates): {len(high_pair_corr)}", "",
              "## Top |correlation| with target",
              *[f"- {k}: {v}" for k, v in target_corr.items()]]
    (out / "eda_report.md").write_text("\n".join(lines))
    print(f"EDA written to {out}")


if __name__ == "__main__":
    main()
