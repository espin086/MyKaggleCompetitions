# Algorithm ↔ Dataset Mapping

Which model families typically perform best on which dataset variants, and why.
`train_baselines.py` enforces the hard rule automatically; use `suggested_models`
in a recipe to override the soft guidance.

## Hard rule (enforced)

Models with `needs_scaling: true` (logreg, elasticnet/ridge, SVM, KNN, GaussianNB)
are **never** paired with a variant that lacks `scale`, `quantile_transform`,
`power_transform`, or `pca`. Distance and gradient-based models on unscaled
features produce garbage baselines that pollute the leaderboard.

## Soft guidance (typical winners by variant type)

| Variant type | Best bets | Usually wasted effort |
|---|---|---|
| Minimal/raw (impute+ordinal) | hist_gbm, random_forest, extra_trees, gbm | linear, KNN, SVM (blocked anyway) |
| Scaled + one-hot | logreg/ridge/elasticnet, svm_rbf, knn | trees (invariant to monotone scaling — redundant compute) |
| PCA-reduced | svm_rbf, knn, logreg | trees (PCA destroys axis-aligned splits trees exploit) |
| Selected subset (kbest/SFM) | everything — selection helps all families | — |
| Target-encoded high-cardinality | hist_gbm, gbm, logreg | one-hot alternatives on same columns |
| Polynomial interactions + selection | logreg, ridge, elasticnet | trees (they learn interactions natively) |

## Model zoo defaults (train_baselines.py)

- **hist_gbm** — the workhorse; sklearn's LightGBM-alike. Almost always top-3 on tabular. 300 iters, lr 0.08, early stopping.
- **random_forest / extra_trees** — 400 trees, min_samples_leaf=2. Strong, low-variance, great ensemble diversity vs boosting.
- **gbm** — classic GradientBoosting; slower but different bias profile than hist_gbm, useful ensemble member.
- **logreg / ridge / elasticnet** — the linear baseline; shines on scaled + engineered features, cheap, decorrelated from trees.
- **svm_rbf / svr_rbf** — capped at 20k rows (O(n²)). Excellent on small/medium PCA'd data.
- **knn** — k=15, distance weights. Rarely wins alone; adds real ensemble diversity.
- **naive_bayes** — near-free; occasionally surprising on independent-ish features.

## Optuna search-space rationale (optimize.py)

- Learning rates log-uniform (0.01–0.3) — sensitivity is multiplicative.
- Tree depth/leaves ranges deliberately wide; regularization (min_samples_leaf, l2) matters more than depth on tabular data.
- Elasticnet l1_ratio spans full 0–1 so Optuna decides ridge-vs-lasso.
- SVM C and gamma both log-scale over 4 decades — the classic grid.
- Seed next loop's intuition from where TPE converges (record it in knowledge.md).

## Ensemble composition advice

Blends win on **decorrelated errors**, not raw strength. From the top-m, prefer
mixing families (a boosted tree + a bagged tree + a linear model) over three
boosted trees with different seeds. Check the `oof_correlation` matrix in
ensembles.json — base models correlated > 0.97 add nothing; note that in
knowledge.md and diversify next loop.
