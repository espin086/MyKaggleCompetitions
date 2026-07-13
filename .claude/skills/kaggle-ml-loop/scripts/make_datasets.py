"""Validate/compile dataset-variant recipes for a loop.

Recipes are written by the FE agents (see references/feature-recipes.md).
This script compiles each into a Pipeline and smoke-fits it on a sample so
broken recipes fail fast, before training burns compute.

Run: python scripts/make_datasets.py --config config.yaml --loop 1 --validate
"""
from __future__ import annotations

import argparse
import traceback

from utils import (compile_recipe, detect_task, encode_target_if_needed,
                   load_config, load_recipes, load_train)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--loop", type=int, required=True)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    X, y = load_train(cfg)
    task = detect_task(cfg, y)
    y, _ = encode_target_if_needed(y, task)
    recipes = load_recipes(cfg, args.loop)
    rs = cfg["evaluation"]["random_state"]

    n = min(len(X), 2000)
    Xs, ys = X.sample(n, random_state=rs), y.sample(n, random_state=rs)
    ok, bad = [], []
    for name, recipe in recipes.items():
        try:
            pipe = compile_recipe(recipe, X, random_state=rs)
            Xt = pipe.fit_transform(Xs, ys)
            ok.append((name, Xt.shape[1]))
        except Exception as e:
            bad.append((name, f"{type(e).__name__}: {e}"))
            traceback.print_exc()
    for name, nf in ok:
        print(f"OK    {name}: {nf} output features")
    for name, err in bad:
        print(f"FAIL  {name}: {err}")
    if bad:
        raise SystemExit(f"{len(bad)} recipe(s) failed validation — fix before training.")
    print(f"All {len(ok)} recipes valid for loop {args.loop}.")


if __name__ == "__main__":
    main()
