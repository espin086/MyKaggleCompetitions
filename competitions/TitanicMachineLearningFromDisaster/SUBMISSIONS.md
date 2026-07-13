# Submissions log

Public leaderboard scores, newest at the bottom.

| Date (UTC) | Approach | Local score | Public score | Status |
|---|---|---|---|---|
| 2026-07-13 | v1: hand-built GradientBoosting + Title/FamilySize/IsAlone | CV 0.8406 | 0.76315 | complete |
| 2026-07-13 | kaggle-ml-loop champion: stacking ensemble (probabilities. bug) | holdout 0.799 | 0.00000 | complete. invalid (probs not 0/1) |
| 2026-07-13 | kaggle-ml-loop champion: stacking ensemble, 0/1 labels | holdout 0.799 | **0.78947** | complete. best so far |

## Notes

- **kaggle-ml-loop beat the hand-built v1** (0.78947 vs 0.76315, +0.026) and the prior 2024
  export (0.78229). Champion = a stacking ensemble picked on the untouched holdout over the
  higher-CV single model. the holdout tiebreak working as designed.
- **Bug found + fixed:** `select_champion.py` exported positive-class *probabilities* for a
  classification task, which scores 0 on an accuracy competition. Fixed to emit probabilities
  only for probability-scored metrics (roc_auc, log loss) and hard 0/1 labels otherwise.
- Run config: `config.yaml` (1 loop, 3 recipes: minimal / linear_scaled / interactions).
  Reproduce: `python .claude/skills/kaggle-ml-loop/scripts/{eda,make_datasets,train_baselines,optimize,ensemble,select_champion}.py --config config.yaml [--loop 1]`.
- Next: more loops (knowledge distillation), add a Title-extraction recipe (the v1 win), tune
  the ensemble base set.
