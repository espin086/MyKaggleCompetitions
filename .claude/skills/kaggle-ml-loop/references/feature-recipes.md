# Feature Recipe Schema & Op Catalog

Every FE agent outputs one JSON file to `recipes/loop_<N>/<name>.json`. Recipes
compile to leak-safe sklearn Pipelines — all fitting happens inside CV folds,
so target encoding, scaling, PCA, and selection never see validation data.

## Schema

```json
{
  "name": "linear_scaled_v2",
  "rationale": "EDA shows 6 skewed numerics; knowledge.md loop-3 found quantile > standard scaling (+0.003 AUC). Trying quantile + interactions.",
  "suggested_models": ["logreg", "svm_rbf", "knn"],
  "ops": [
    {"op": "impute_numeric", "strategy": "median"},
    {"op": "impute_categorical", "strategy": "most_frequent"},
    {"op": "quantile_transform", "output": "normal"},
    {"op": "onehot", "max_categories": 30},
    {"op": "polynomial", "degree": 2, "interaction_only": true},
    {"op": "select_kbest", "score_func": "f", "task": "classification", "k": 40}
  ]
}
```

- `name` — unique within the loop; becomes the MLflow `variant` tag.
- `rationale` — REQUIRED. Cite the EDA finding or knowledge.md entry driving each choice. On loops 2+, state which learning you exploit or contradict.
- `suggested_models` — optional. Omit to use the default algorithm↔dataset mapping; provide a list to override it (see algorithm-dataset-map.md).
- `ops` — ordered list. Impute/encode ops are hoisted into the column stage; post-encoding ops (polynomial, pca, selection) apply in listed order.

## Op catalog

| op | params (defaults) | notes |
|---|---|---|
| `impute_numeric` | strategy: median\|mean\|constant | |
| `impute_categorical` | strategy: most_frequent\|constant | |
| `ordinal` | — | default categorical encoding; fine for trees |
| `onehot` | max_categories: 30 | linear/distance models; cap cardinality |
| `target_encode` | — | high-cardinality cats; internally CV'd by sklearn TargetEncoder |
| `log1p` | — | right-skewed non-negative numerics |
| `power_transform` | method: yeo-johnson\|box-cox | general skew fix |
| `quantile_transform` | output: normal\|uniform | robust to outliers |
| `scale` | kind: standard\|robust\|minmax | REQUIRED before linear/SVM/KNN unless quantile/power used |
| `bin` | n_bins: 5, strategy: quantile | discretize numerics |
| `polynomial` | degree: 2, interaction_only: true | feature count explodes fast — pair with selection |
| `variance_threshold` | threshold: 0.0 | drop constants |
| `pca` | n_components: 0.95 (variance) or int | variable reduction; always scale first |
| `select_kbest` | score_func: f\|mi, task, k: 20 | univariate selection |
| `select_from_model` | task, threshold: "median" | ExtraTrees importance-based |
| `rfe` | task, n_features: 20 | greedy backward elimination (slow) |

## Design principles

1. One recipe = one hypothesis. Don't stack every op — you won't know what worked.
2. The minimal recipe (`impute + ordinal`) must exist every loop as the control.
3. Order matters: transform → encode → interact → reduce/select.
4. If knowledge.md says an op hurt twice, stop proposing it.
5. Feature count sanity: `polynomial` on 50 columns yields ~1,275 interactions — always follow with `select_kbest` or `select_from_model`.
