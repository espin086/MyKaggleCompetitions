# Agent brief. ROGII Wellbore Geology Prediction

Read this first. It is the orientation for any agent (or the kaggle-ml-loop) working this
competition. The deeper files are: `01-overview.md`, `02-data-dictionary.md`,
`03-rules-and-constraints.md`, `04-discussion-intel.md`, `05-plan-of-attack.md`.

## What you're predicting

For ~200 horizontal wells, predict `tvt` (True Vertical Thickness, ft) at every 1-ft station
inside each well's **evaluation zone**. the stretch of the lateral where `TVT_input` is
`NaN`. RMSE against the hidden truth, lower is better. Public LB leader is ~4.86 RMSE; a naive
constant/linear baseline is far worse.

## Why this is not a plain tabular problem

`tvt` is the geologic position of the drill bit. how deep inside the rock column the bit sits
as the horizontal well snakes through folded, faulted layers. It is a **structured sequence /
signal-matching** problem, not IID rows:

1. Each well is an ordered sequence along Measured Depth (`MD`). Neighboring stations are
   highly correlated; `tvt` moves smoothly except at faults.
2. The **typewell** is a vertical "template" of the rock column: `GR` vs depth with formation
   labels. The core task is aligning the lateral's `GR` trace to the typewell's `GR` template
   to read off geologic position. a correlation / dynamic-time-warping problem.
3. Geometry matters: `X, Y, Z, MD` give the well path; `Z` (true vertical depth) plus the
   local dip of the layers largely determines `tvt`. A useful prior is `tvt ≈ linear(MD, Z)`
   per well, then correct with the GR/typewell match.

The community consensus (see `04-discussion-intel.md`): **pure XGBoost on flat features hits a
wall.** What moves the needle is spatial + sequential context. 1D CNNs over the GR sequence,
typewell log-matching / DTW, per-well linear priors, and forward-simulation of the log.

## Hard constraints (from the rules. do not violate)

- **Code competition:** you submit a **Kaggle Notebook**, not a CSV via CLI. Notebook must run
  ≤ 9h (CPU or GPU) with **internet disabled** and write `submission.csv`.
- **5 submissions/day**, 2 final submissions selectable.
- **Data is competition-use-only**. never redistribute, never commit `data/` to git.
- External public data / pretrained models are allowed if freely available to all.

## The fast path

1. Join the competition + download data (see README. blocked until rules accepted).
2. Build a per-well loader; confirm the evaluation-zone mask (`TVT_input.isna()`).
3. Ship a dumb baseline first (per-well `tvt ≈ linear(MD, Z)` fit on the known zone,
   extrapolated into the eval zone) to lock the submission pipeline end-to-end.
4. Then layer the GR/typewell alignment + a 1D-CNN sequence model. See `05-plan-of-attack.md`.

Local validation: hold out whole wells (group by well hash), and within a well mask a
contiguous span to imitate the eval zone. random-row CV will lie because rows leak.
