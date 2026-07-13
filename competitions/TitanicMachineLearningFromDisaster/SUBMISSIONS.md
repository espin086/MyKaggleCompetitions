# Submissions log

Public leaderboard scores, newest at the bottom.

| Date (UTC) | Approach | Local score | Public score | Status |
|---|---|---|---|---|
| 2026-07-13 | v1: hand-built GradientBoosting + Title/FamilySize/IsAlone | CV 0.8406 | 0.76315 | complete |
| 2026-07-13 | kaggle-ml-loop champion: stacking ensemble (probabilities. bug) | holdout 0.799 | 0.00000 | complete. invalid (probs not 0/1) |
| 2026-07-13 | kaggle-ml-loop champion: stacking ensemble, 0/1 labels | holdout 0.799 | **0.78947** | complete. **best so far** |
| 2026-07-13 | kaggle-ml-loop 5-loop, enriched features, target_encoded hist_gbm (optuna) | holdout 0.8156 | 0.75837 | complete. regressed on public LB |
| 2026-07-13 | robust soft-voting RF+HistGBM+LogReg, ordinal enriched features | CV 0.844 | 0.75598 | complete. regressed on public LB |

Notebooks for the 5-loop run: [`notebooks/eda_titanic_20260713-0916.ipynb`](notebooks/eda_titanic_20260713-0916.ipynb) ·
[`notebooks/champion_titanic_20260713-0916.ipynb`](notebooks/champion_titanic_20260713-0916.ipynb) (see `notebooks/INDEX.md`).

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
- **2026-07-13 5-loop enriched run. CV↔LB gap, did NOT beat 0.78947.** Added `src/prep.py`
  (Title, FamilySize/IsAlone, TicketGroupSize, FarePerPerson, Deck, Age-by-Title impute) and
  ran 5 loops. Local scores went UP (champion holdout 0.8156, robust ensemble CV 0.844) but
  both public submissions came DOWN (0.758, 0.756). Classic Titanic overfit: the 179-row
  holdout and 5-fold CV both overestimate; richer features + target encoding memorize signal
  that doesn't transfer to the 418-row public set. **The raw-feature stacking ensemble
  (0.78947) remains the best public score and stays on the leaderboard** (Kaggle keeps the
  best). Lesson for this competition: prefer simpler feature sets and ensembles; treat the
  local holdout as a weak proxy. The champion notebook visualizes this exact gap (finalist
  CV-vs-holdout chart, permutation importance).
- **Deliverable this run:** the pipeline now auto-generates two explainable notebooks per run
  (EDA + champion) into `notebooks/`, git-tracked and timestamped, logged in `INDEX.md`. See
  the repo `CLAUDE.md` § notebooks and `.claude/skills/kaggle-ml-loop/references/notebook-conventions.md`.
