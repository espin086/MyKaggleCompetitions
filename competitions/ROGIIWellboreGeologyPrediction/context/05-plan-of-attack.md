# Plan of attack. ROGII Wellbore Geology Prediction

Staged plan. Each stage ends with a verifiable check (per the verify-with-real-data rule).
Ship the pipeline before the model. a submitting baseline beats a perfect model that can't
submit. Deadline 2026-08-05; ~5 submissions/day.

## Stage 0. Enter + load (blocked on Join)
1. Accept rules / Join on the website (manual, one click). Download + unzip data (README).
2. Write a per-well loader in `src/`: given a well hash, return the horizontal_well df + the
   typewell df. Enumerate `data/train/*__horizontal_well.csv` → ~700 well hashes; `data/test/`
   → the visible examples (real test swapped in on rerun).
   - **Verify:** counts of train/test wells; a well's df has expected columns; the eval-zone
     mask `TVT_input.isna()` selects a contiguous span, not scattered rows.

## Stage 1. EDA (answers that shape the model)
- Distribution of `tvt`; per-well range; how big is the eval zone vs the known zone.
- Is `tvt` smooth in `MD`? Where are the jumps (faults)? Plot a few wells + their `.png`.
- Correlation of `tvt` with `Z`, `MD`, `GR`. Fit per-well `tvt ~ MD + Z`. how good is the
  linear prior alone (R², residual std)?
- Typewell: does lateral `GR` visibly match typewell `GR` when shifted by `tvt`? This validates
  the whole log-matching premise.
  - **Verify:** a notebook/figure per question saved to `context/eda/` or `src/`.

## Stage 2. Submitting baseline (lock the pipeline)
- Per-well linear prior: fit `tvt = a·MD + b·Z + c` on the known-zone rows, predict the eval
  zone. Assemble `submission.csv` with `id = {hash}_{row_index}`.
- Build the **Kaggle notebook** that reproduces this offline (internet off) and writes
  `submission.csv`. This is the real submission artifact. get it green early.
  - **Verify:** submission row count matches sample_submission; first Kaggle submission returns
    a public score; log it in `SUBMISSIONS.md`. This proves the end-to-end path works.
- **Local CV that doesn't lie:** GroupKFold by well hash; within held-out wells, mask a
  contiguous span to imitate the eval zone. Track local RMSE vs public LB gap.

## Stage 3. Signal matching (typewell alignment). TESTED, underperformed; feed as feature not gate

Two heuristic variants were tested against the real per-well eval zones (same 773-well
harness as Stage 2), both **worse than the Stage 2 linear prior**:

| Approach | Local RMSE (50-well sample) | vs Stage 2 (70.23 same sample) |
|---|---|---|
| Pointwise nearest-GR match within a ±100ft window of the linear prior (`src/stage3_gr_match.py`) | 74.72 | worse |
| Windowed shape-match (±5-row window, SSE) + per-well confidence gate (`src/stage3_windowed_match.py`) | 75.06 | worse |

**Why:** diagnosed on well `000d7d20`. its true eval-zone span is only 15 ft, but the
typewell is noisy (GR std 26, range 28-158) over ~650 ft at 0.5 ft resolution, and the
linear prior was *already* tight there (RMSE 10). A wide/noisy search window gives the
matcher enough freedom to latch onto spurious GR coincidences instead of real structure.
A follow-up diagnostic (median |lateral GR − typewell GR at known TVT| across 50 wells)
showed typewell-fit quality is fairly **uniform** (~7.7 GR units, not bimodal). so a
simple per-well trust/no-trust gate has no signal to key off either. Two informed attempts
without a reinforcing trend; per `verify-with-real-data.md` this stopped the heuristic-gate
line rather than burn a third blind parameter guess.

**Verdict, revised:** don't hand-tune a GR-match heuristic further. The signal is still
probably useful, but only as an *input feature* to a learned model (Stage 4) that can weigh
it per-region rather than a fixed threshold. see Stage 4's `windowed_match` /
`match_minus_prior` features, which feed the same matcher's output into the boosted model
instead of gating it by hand. Real DTW alignment (proper dynamic time warping over the full
known-zone sequence, not a fixed-radius local window) remains untried and is the more
faithful reading of the hengck23 DTW thread. worth a real attempt if Stage 4/5 plateau.
Watch for shared/duplicate typewells across wells (RMorrison thread) if revisiting this.

## Stage 4a. Global gradient-boosted model. TESTED, real improvement (67.09 → 52.90)

Before the 1D-CNN lift, tried the simpler "let a learned model combine the existing signals"
step: one `HistGradientBoostingRegressor` trained across **all** wells' real eval zones
pooled together (not per-well), with `linear_prior`, `windowed_match`,
`match_minus_prior`, geometry (`MD/X/Y/Z`), `GR`, `dist_from_known_boundary`,
`eval_zone_frac`, and `known_zone_rows` as features (`src/stage4_global_model.py`).
Validated with 5-fold **GroupKFold by well** (never train and score on the same well). directly comparable to Stage 2's 67.09 since both use the full 773-well real-eval-zone set.

| Signal | RMSE (773 wells, full set) |
|---|---|
| `linear_prior` alone (Stage 2) | 67.09 |
| `windowed_match` alone (Stage 3, confirms the 50-well finding at full scale) | 71.98 |
| **Stage 4a: HistGradientBoostingRegressor combining both + features (OOF)** | **52.90** |

**Why it worked where Stage 3's hand-tuned gate didn't:** the model learns *when* to trust
the GR-match signal (and by how much) per region of feature space, rather than a single
fixed threshold. Confirms the Stage 3 verdict above. the GR-match signal had real
information, it just needed a learner instead of a heuristic to extract it.

**Caveat:** per-fold RMSE ranged 46.7. 63.8. real well-to-well difficulty variance
(structural complexity), not fold noise; the closest-to-plausible test wells still likely
score somewhere in that range, still far from the ~4.86 LB leader. Runtime: ~11 min for the
full 773-well feature build + 5-fold train (feature build dominates. the windowed-match
step is O(rows) with per-row Python overhead; vectorizing it would speed later iterations).

**Submitted and confirmed with a real Kaggle score.** Pushed as a Kaggle Notebook
(`notebooks/submission_stage4a.ipynb`) via `kaggle kernels push`, ran clean on Kaggle's
infrastructure (0/773 train wells failed), submitted via
`kaggle competitions submit -k jjespinoza/rogii-stage-4a-global-gradient-boosted-model -v 1`.
**Public LB: 45.196**, vs Stage 2's 80.534. a real 44% improvement, and the public-LB gain
tracked the local-CV gain in the same direction (no CV/LB divergence) - the local validation
methodology (real per-well eval zones, GroupKFold by well) is trustworthy. Full detail in
`../SUBMISSIONS.md`.

## Stage 4b. Sequence model. TESTED, underperformed Stage 4a (3 attempts, hard-capped)

A 1D CNN over local windows (41 consecutive eval-zone rows) of the Stage 4a feature set,
convolving where Stage 4a treated every row independently (`src/stage4b_cnn_model.py`).
Three attempts, GroupKFold-by-well on a 30-well local sample (fast iteration before
committing to the full 773-well run):

| Attempt | Features | Local RMSE (30-well GroupKFold) |
|---|---|---|
| 1 | All 11 Stage 4a features (incl. raw `MD/X/Y/Z`) | 81.02 |
| 2 | Dropped `X/Y`, added dropout + weight decay | 84.67 (regressed) |
| 3 | Kept only `GR` + `linear_prior`/`windowed_match`/`match_minus_prior`/`eval_zone_frac` | **70.18** (best, still worse than Stage 4a) |

**Diagnosis:** attempts 1-2 converged to near-zero TRAIN loss within a single epoch while
held-out RMSE stayed terrible - textbook well-identity memorization, not sequence learning.
With only ~700 wells (≈25 per training fold), the CNN's capacity is enough to memorize
"well at this (X, Y) / this MD range / this known_zone_rows count → target ≈ this value"
instead of learning transferable GR-shape patterns. Attempt 3 stripped every scalar that
could serve as a well-identity key and improved meaningfully (81.02/84.67 → 70.18), which
confirms the diagnosis - but it's still well short of **Stage 4a's 52.90 local / 45.196
public LB**. Even the cleanest feature set didn't have enough data (and target-scaling was
also needed just to get the model to converge to the right output magnitude at all - raw
~11-12k targets converge far slower than scaled ones from a random init).

Per `verify-with-real-data.md`, 3 attempts is the hard cap regardless of outcome. Submitted
attempt 3 anyway for a real, honest data point (see `../SUBMISSIONS.md`) rather than only
estimate it - it is NOT the recommended model; Stage 4a remains the best real result.

**Submitted and confirmed: public LB 72.734.** Pushed as a Kaggle Notebook
(`notebooks/submission_stage4b.ipynb`) via `kaggle kernels push`, ran clean on Kaggle's
infrastructure (0/773 train wells failed, 1-epoch CNN trained on all 3.78M rows), submitted
via `kaggle competitions submit -k jjespinoza/rogii-stage-4b-1d-cnn-sequence-model -v 1`.
Worse than Stage 4a's 45.196, better than Stage 2's 80.534. Local (70.18) and public (72.73)
tracked closely - no CV/LB divergence, so the negative result is trustworthy, not a fluke of
local validation. Full detail in `../SUBMISSIONS.md`.

**Revised verdict for future work:** a naive per-row-window CNN on this small a well count
doesn't beat gradient boosting on the same signals. A genuine sequence-to-sequence approach
(predict the whole eval zone at once, or pretrain on a self-supervised task using ALL rows -
not just eval-zone-labeled ones - across all 773 wells) has more data to learn from and is
the more faithful reading of what the top public kernels (hengck23 MTP thread) are doing;
this windowed-per-row CNN was a lighter first attempt and it hit its ceiling.

## Stage 5. Ensemble. TESTED, submitted, small net regression on public LB

Original plan: "blend physics prior + DTW match + CNN." Revised given Stage 4b's real
result (public LB 72.734, clearly worse than Stage 4a's 45.196) - the CNN was excluded from
the default blend since including a much-worse model was more likely to hurt than help
(`src/stage5_ensemble.py`).

**What was built:** a non-negative least-squares blend of `linear_prior` (Stage 2),
`windowed_match` (Stage 3), and Stage 4a's `HistGradientBoostingRegressor`, using leak-free
Stage 4a out-of-fold predictions (5-fold GroupKFold) as the third input so stacking a
meta-model on top doesn't leak. Weights: `linear_prior` 0.123, `windowed_match` 0.096,
`stage4a_pred` 0.781 - Stage 4a dominates but the other two aren't zeroed out.

**Local validation (the honest number, not just in-sample):** weights fit on 4/5 of wells'
OOF predictions, evaluated on the held-out 1/5 -> **RMSE 52.03** vs Stage 4a alone's 52.90,
a real ~1.6% local improvement.

**Submitted and confirmed - but it's a small net REGRESSION on the real leaderboard:**
public LB **45.997** vs Stage 4a alone's **45.196** (~1.8% worse). Pushed via
`kaggle kernels push` (`notebooks/submission_stage5.ipynb`, kernel
`jjespinoza/rogii-stage-5-ensemble-blend`), ran clean (0/773 wells failed), submitted via
`kaggle competitions submit -k jjespinoza/rogii-stage-5-ensemble-blend -v 1`.

**Follow-up investigation (`src/stage5b_blend_variance.py`), answering "is the ~1 RMSE
point local gap real, or noise?"** with actual repeated-CV data rather than guessing:

Repeated the held-out split **10 times with genuinely random well-to-fold assignment**
(first attempt used sklearn's `GroupKFold`, which turned out to be fully deterministic
regardless of input order - shuffling the well list before calling it did nothing, caught
because all 10 "repeats" came back bit-for-bit identical; fixed with manual random fold
assignment). Also tested ridge blending on standardized features (the first ridge attempt
used raw ~11,000-magnitude features, where alpha up to 100 was negligible next to the Gram
matrix's scale and did nothing - fixed by standardizing before ridge).

**Result: the local ensemble improvement is real and NOT noise.** Across 10 independent
random splits, the NNLS blend beat Stage 4a alone in **every single repeat** - mean
improvement -0.69 RMSE (range -0.31 to -0.91), std 0.207, **t-stat ~ -10.6**. Ridge
(standardized, alpha 0.01-10, all near-identical to each other) gave a slightly smaller but
equally consistent improvement (-0.60, t-stat ~ -9.6). Regularization doesn't meaningfully
change the picture - NNLS's non-negativity constraint isn't costing accuracy vs a
shrinkage-based blend.

**So the CV/LB divergence isn't a "local estimate too imprecise" problem - it's something
else.** The most likely explanation: per the competition rules, **the public leaderboard
itself is scored on a representative SAMPLE of the test data**, not the full test set (the
private leaderboard, which determines final standing, uses a different, larger sample). A
single public submission's score is therefore itself a noisy point estimate of the true
private-test performance, not ground truth to calibrate against. Getting one public score
that lands unfavorably for an approach that is robustly better in repeated local validation
is exactly the kind of outcome that small-sample public-LB noise would produce.

**Follow-up: resubmitted the identical Stage 5 predictions a second time** (same
deterministic model, bit-for-bit identical `submission.csv`) to test whether the public
score itself varies per submission. **Confirmed identical: 45.997 both times.** This rules
out submission-to-submission randomness as the explanation - the public leaderboard is
scored against a FIXED sample of the test data, not a re-drawn one each time. So a re-submit
of the same model gives zero new information; the divergence is more likely a genuine
(if small) difference between the training wells' distribution and the specific fixed
public sample, not noise that would average out with more submissions of the same thing.

**Revised recommendation:** the local evidence for the NNLS blend remains strong (10/10
random splits favor it, t-stat ~-10.6), but the one real public score we have for it is
confirmed stable at 45.997, worse than Stage 4a's 45.196. Stage 4a alone remains the safer
choice to lead with. Resolving which is truly better needs either a genuinely different/
improved model, or the private leaderboard at competition close - not another identical
resubmission.

**Not done:** uncertainty estimation (per-prediction confidence) and picking 2 diverse final
submissions for judging - deferred since which model is actually best is now genuinely
unresolved between Stage 4a and the Stage 5 blend, not because there's nothing to diversify.

## Stage 6. K-means cluster features - planned

Add K-means clustering as a preprocessing step on the Stage 4a feature set, then feed the
assigned cluster (+ distance to centroid) into the gradient-boosted model as extra features,
on the hypothesis that clusters capture geological "regimes" (e.g. flat vs faulted terrain)
that let the model specialize its splits.

1. **Elbow analysis first.** Fit KMeans for k = 2..15 on the standardized Stage 4a feature
   set (leak-free care needed: fit within CV folds, not on the full dataset, for the actual
   validated model - the elbow plot itself can use the full dataset since it's just
   descriptive/exploratory, not a trained artifact).
2. **Search range from the elbow.** Explore k values a few steps below and above the elbow
   point, not just the elbow itself - the "best" k for downstream supervised performance
   doesn't always coincide with the unsupervised elbow.
3. **Tune k with Optuna**, objective = leak-free GroupKFold-by-well OOF RMSE of the
   downstream `HistGradientBoostingRegressor` (KMeans fit within each training fold only,
   cluster label passed as a native categorical feature via `categorical_features=`).
4. **Verify before submitting:** does the best-k-augmented model actually beat Stage 4a's
   52.90 local / 45.196 public LB? If yes, submit and confirm with a real score. If no,
   document the negative result anyway - it goes in the experiment log either way, per the
   standing rule that every real attempt gets logged, win or lose.

## Tooling note. reuse the repo's kaggle-ml-loop where it fits
The `kaggle-ml-loop` skill automates the tabular loop (EDA → recipes → MLflow → champion).
It fits Stages 1. 2 (EDA + the linear/tabular baseline) but **not** the sequence/DTW stages,
which are custom. Use it to bootstrap the baseline fast, then hand-build Stages 3. 5.

## Risks / watchlist
- **CV that leaks:** random-row CV will look great and fail on LB. Group by well; mask spans.
- **Visible-test overfit:** the visible `test/` is copied from train. don't tune to it.
- **Private rescore:** a prior rescore happened; trust GroupKFold over early public LB.
- **Notebook runtime:** ≤ 9h, internet off. Keep inference lean; preload weights as a Dataset.
