"""Build the two explainable Jupyter notebooks at the end of a kaggle-ml-loop run.

Produces, into `competitions/<Name>/notebooks/`:
  - eda_<competition>_<YYYYMMDD-HHMM>.ipynb      — seaborn EDA with survival signal
  - champion_<competition>_<YYYYMMDD-HHMM>.ipynb — what MLflow taught + how the champion
                                                    reads the data (permutation importance)

Each notebook is assembled from a template spec in
`assets/notebook_templates/{eda,champion}_template.py`, written as a raw .ipynb, then
executed with nbconvert so all charts and MLflow tables render inline. A failing cell
does not abort the run — the notebook is kept with the error shown. Every build appends
a row to `notebooks/INDEX.md`.

Run: python scripts/build_notebooks.py --config config.yaml [--which eda|champion|both]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import detect_task, load_config, load_train, resolve_metric, run_dir  # noqa: E402

TEMPLATES = Path(__file__).resolve().parent.parent / "assets" / "notebook_templates"


def _rel(target: Path, base: Path) -> str:
    """Relative link from the notebooks dir to a target path (for clickable md links)."""
    try:
        return os.path.relpath(target, base)
    except ValueError:
        return str(target)


def _context(cfg: dict, nb_dir: Path, ts: str) -> dict:
    rd = run_dir(cfg)
    comp_dir = rd.parent
    data = cfg["data"]
    train_p = (comp_dir / data["train_path"]).resolve() if not os.path.isabs(data["train_path"]) else Path(data["train_path"])
    test_p = data.get("test_path")
    test_p = ((comp_dir / test_p).resolve() if test_p and not os.path.isabs(test_p) else Path(test_p)) if test_p else None
    mlflow_uri = cfg["mlflow"].get("tracking_uri") or f"sqlite:///{rd / 'mlflow.db'}"
    return {
        # values
        "competition": cfg["competition_name"],
        "timestamp": ts,
        "target": data["target"],
        "id_col": data.get("id_column"),
        "run_dir_abs": str(rd),
        "comp_dir_abs": str(comp_dir),
        "config_abs": str((comp_dir / "config.yaml").resolve()),
        "train_abs": str(train_p),
        "test_abs": str(test_p) if test_p else None,
        "mlflow_uri": mlflow_uri,
        "mlflow_ui_url": "http://127.0.0.1:5000",
        "metric": resolve_metric(cfg, detect_task(cfg, load_train(cfg, "all")[1])),
        "random_state": cfg["evaluation"]["random_state"],
        "holdout_fraction": cfg["evaluation"].get("holdout_fraction", 0.2),
        # relative links (from notebooks/)
        "train_rel": _rel(train_p, nb_dir),
        "test_rel": _rel(test_p, nb_dir) if test_p else "-",
        "config_rel": _rel((comp_dir / "config.yaml"), nb_dir),
        "run_rel": _rel(rd, nb_dir),
        "eda_report_rel": _rel(rd / "eda" / "eda_report.md", nb_dir),
        "eda_summary_rel": _rel(rd / "eda" / "eda_summary.json", nb_dir),
        "champion_json_rel": _rel(rd / "champion" / "champion.json", nb_dir),
        "model_rel": _rel(rd / "champion" / "model.joblib", nb_dir),
        "submission_rel": _rel(rd / "champion" / "submission.csv", nb_dir),
        "knowledge_rel": _rel(rd / "knowledge.md", nb_dir),
        "comp_rel_root": _rel(comp_dir, nb_dir),
    }


def _params_cell(ctx: dict) -> str:
    p = lambda v: repr(v)
    return "\n".join([
        "# --- parameters (injected by build_notebooks.py) ---",
        f"COMPETITION = {p(ctx['competition'])}",
        f"COMP_DIR = {p(ctx['comp_dir_abs'])}",
        f"RUN_DIR = {p(ctx['run_dir_abs'])}",
        f"CONFIG_PATH = {p(ctx['config_abs'])}",
        f"TRAIN_CSV = {p(ctx['train_abs'])}",
        f"TEST_CSV = {p(ctx['test_abs'])}",
        f"TARGET = {p(ctx['target'])}",
        f"ID_COL = {p(ctx['id_col'])}",
        f"METRIC = {p(ctx['metric'])}",
        f"RANDOM_STATE = {p(ctx['random_state'])}",
        f"HOLDOUT_FRACTION = {p(ctx['holdout_fraction'])}",
        f"MLFLOW_URI = {p(ctx['mlflow_uri'])}",
        f"MLFLOW_DB = {p(ctx['run_dir_abs'] + '/mlflow.db')}",
    ])


def _assemble(kind: str, ctx: dict) -> nbformat.NotebookNode:
    mod_name = f"{kind}_template"
    sys.path.insert(0, str(TEMPLATES))
    mod = __import__(mod_name)
    nb = new_notebook()
    nb.cells.append(new_code_cell(_params_cell(ctx)))
    for celltype, source in mod.notebook_cells(ctx):
        nb.cells.append(new_markdown_cell(source) if celltype == "markdown"
                        else new_code_cell(source))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def _execute(nb: nbformat.NotebookNode, cwd: Path) -> bool:
    try:
        from nbconvert.preprocessors import ExecutePreprocessor
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3", allow_errors=True)
        ep.preprocess(nb, {"metadata": {"path": str(cwd)}})
        return True
    except Exception as e:  # kernel missing, etc. — keep the unexecuted notebook
        print(f"  ! execution skipped ({type(e).__name__}: {e}); notebook kept unexecuted")
        return False


def _log_index(nb_dir: Path, ctx: dict, built: dict, champ: dict | None):
    idx = nb_dir / "INDEX.md"
    if not idx.exists():
        idx.write_text(
            "# Notebook index\n\n"
            "Every kaggle-ml-loop run produces a timestamped EDA + champion notebook pair. "
            "Newest at the bottom.\n\n"
            "| Run (UTC) | EDA notebook | Champion notebook | Champion | CV(dev) | Holdout | MLflow UI |\n"
            "|---|---|---|---|---|---|---|\n")
    cv = f"{champ.get('cv_mean_dev'):.4f}" if champ and champ.get("cv_mean_dev") is not None else "-"
    ho = f"{champ.get('holdout_score'):.4f}" if champ and champ.get("holdout_score") is not None else "-"
    kind = (champ.get("kind") if champ else None) or "-"
    eda = f"[{built['eda'].name}]({built['eda'].name})" if "eda" in built else "-"
    champ_nb = f"[{built['champion'].name}]({built['champion'].name})" if "champion" in built else "-"
    ui = f"`mlflow ui --backend-store-uri {ctx['mlflow_uri']}`"
    with idx.open("a") as f:
        f.write(f"| {ctx['timestamp']} | {eda} | {champ_nb} | {kind} | {cv} | {ho} | {ui} |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--which", choices=["eda", "champion", "both"], default="both")
    ap.add_argument("--timestamp", help="override run timestamp (YYYYMMDD-HHMM)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rd = run_dir(cfg)
    nb_dir = rd.parent / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)
    ts = args.timestamp or datetime.now().strftime("%Y%m%d-%H%M")
    ctx = _context(cfg, nb_dir, ts)
    comp = cfg["competition_name"]

    which = ["eda", "champion"] if args.which == "both" else [args.which]
    built: dict[str, Path] = {}
    for kind in which:
        if kind == "champion" and not (rd / "champion" / "champion.json").exists():
            print("champion.json missing — run select_champion.py first; skipping champion notebook.")
            continue
        print(f"Building {kind} notebook...")
        nb = _assemble(kind, ctx)
        _execute(nb, nb_dir)
        out = nb_dir / f"{kind}_{comp}_{ts}.ipynb"
        nbformat.write(nb, out)
        built[kind] = out
        print(f"  -> {out}")

    champ = None
    cj = rd / "champion" / "champion.json"
    if cj.exists():
        import json
        champ = json.loads(cj.read_text())
    _log_index(nb_dir, ctx, built, champ)
    print(f"\nLogged to {nb_dir / 'INDEX.md'}")
    print(f"MLflow UI: mlflow ui --backend-store-uri {ctx['mlflow_uri']}")


if __name__ == "__main__":
    main()
