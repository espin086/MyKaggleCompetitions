# ROGII - Wellbore Geology Prediction

Kaggle competition workspace. Slug: `rogii-wellbore-geology-prediction`
https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction

**Type:** Featured **Code Competition** (notebook submission, internet disabled) ·
**Prize:** $50,000 · **Final deadline:** 2026-08-05 · **Entry deadline:** 2026-07-29

## Goal in one line

Predict `tvt` (True Vertical Thickness, ft) along the hidden "evaluation zone" of ~200
horizontal oil-and-gas wells, given each well's trajectory + gamma-ray log and a vertical
reference log (typewell). Lower RMSE wins.

## Status

- [x] **Rules accepted / competition joined**
- [x] Data downloaded to `data/` (773 train wells, 3 visible test wells; real ~200-well test
      set is swapped in when the notebook reruns)
- [ ] EDA on well geometry + logs
- [x] Baseline model + local CV (Stage 2 linear prior. see result below, not competitive yet)
- [x] Submission notebook built + executed locally (`notebooks/submission.ipynb`)
- [x] First notebook submission pushed to Kaggle (Stage 2 baseline, pipeline-proving)
- [x] Stage 3 typewell/GR alignment tested (two heuristic variants, both underperformed. see
      `context/05-plan-of-attack.md`) - informed the Stage 4 feature design instead
- [x] Stage 4a global gradient-boosted model. real improvement, OOF RMSE 52.90 (see below)
- [x] Stage 4a submitted to Kaggle. **real public score 45.196** (vs Stage 2's 80.534). see
      `SUBMISSIONS.md`. **Best model so far.**
- [x] Stage 4b 1D CNN sequence model tested (3 attempts) and submitted. real public score
      **72.734** - worse than Stage 4a, kept as an honest documented result (see below)
- [x] Stage 5 ensemble (linear prior + GR-match + Stage 4a, CNN excluded) tested and
      submitted: real public score **45.997** - a small net regression vs Stage 4a alone,
      despite a small local held-out improvement. See below - Stage 4a remains recommended.
- [x] Stage 6 K-means cluster features tested (elbow + Optuna-tuned k): **negative result**,
      53.05 vs Stage 4a's 52.90 baseline - not submitted, logged per the standing rule.
- [x] Stage 7 guarded physical override (adapted from a public kernel) tested and submitted:
      real public score **45.196** - bit-for-bit identical to Stage 4a alone. The guard is
      safe by construction and fired correctly on local sanity-check wells, but did NOT fire
      on the real hidden test set (no train/test well-ID overlap currently exists). Confirmed
      negative for this specific exploit; cost nothing since it never regresses.
- [x] Stage 8 cross-well spatial prior (`FormationPlaneKNN`, adapted from the same public
      kernel) tested and submitted: real public score **45.221** - a tiny, ~noise-level
      difference from Stage 4a alone (45.196), despite a real local improvement in every
      GroupKFold fold (52.57 vs 52.90). First stage to use ANY cross-well information. A
      real bug (catastrophic unstable extrapolation, RMSE 22,032 alone) was diagnosed and
      fixed with a guard before this result. Not adopted as the new best model.

## Baseline result (Stage 2. per-well linear prior `tvt ~ MD + Z`)

`src/baseline.py` fits `tvt = a*MD + b*Z + c` per well on that well's own known rows
(`TVT_input` not null), then predicts that same well's real evaluation zone (`TVT_input`
null). no synthetic masking; every train well already carries its own real eval zone, so
this is the actual test-time procedure. Run: `PYTHONPATH=src python3 src/baseline.py`.

**Overall local RMSE: 67.09** (median per-well: 33.07) vs. the public LB leader's ~4.86.
Not competitive. confirms the forum consensus (`context/04-discussion-intel.md`): a flat
linear/tabular fit plateaus hard. Root cause, confirmed by inspecting the worst well
(`4a335117`, RMSE 410): the eval zone is a long extrapolation. known zone spans ~1,700 ft of
MD, the eval zone spans ~5,000 ft beyond it. so a straight-line fit blows up wherever the
well crosses a fault or dip. Best wells (RMSE ~1.5-2.5) are ones where the geology stays
flat clear through the eval zone; worst wells (RMSE 280-410) hit structural breaks.

**Verdict:** pipeline is proven end-to-end (real per-well fit -> real eval-zone scoring), but
the model itself needs the typewell/GR alignment (Stage 3) before it's submission-worthy.
a linear-only submission would score far off the leaderboard.

## Stage 3 result (typewell/GR alignment. tested, underperformed)

Two heuristic variants (`src/stage3_gr_match.py` pointwise, `src/stage3_windowed_match.py`
windowed+gated) both scored WORSE than the plain Stage 2 linear prior on a 50-well sample
(74.72 and 75.06 vs 70.23). Root cause and full diagnostic write-up in
`context/05-plan-of-attack.md` Stage 3 section - short version: GR alone is too noisy a
discriminator for a hand-tuned local match/gate at this data's resolution. The signal turned
out to still be useful, just not as a heuristic - see Stage 4a below, which feeds the same
matcher's output into a learned model instead of gating it by hand.

## Stage 4a result (global gradient-boosted model. real improvement)

`src/stage4_global_model.py` trains ONE `HistGradientBoostingRegressor` across all 773
wells' real eval zones pooled together, using `linear_prior`, `windowed_match`,
`match_minus_prior`, geometry, `GR`, and eval-zone-position features as inputs. Validated
with 5-fold **GroupKFold by well** (leak-free - never trains and scores on the same well).

**Overall out-of-fold RMSE: 52.90** vs. Stage 2's 67.09 (linear alone) and 71.98
(windowed-match alone, full-scale confirmation of the Stage 3 finding) - a genuine 21%
improvement by letting the model learn when to trust each signal instead of a fixed
threshold. Per-fold RMSE ranged 46.7-63.8 (real well-to-well difficulty variance). Still far
from the ~4.86 LB leader - this is a flat/tabular model over per-row features, not a true
sequence model; Stage 4b (1D CNN / proper DTW) is what's expected to close most of the
remaining gap. Runtime: ~11 min full pipeline (feature build dominates).

Run: `PYTHONPATH=src python3 src/stage4_global_model.py` (takes several minutes - the
windowed-match feature has per-row Python overhead; worth vectorizing before iterating on it
further).

**Submitted and scored on Kaggle: public LB 45.196** (vs Stage 2's 80.534 - a real 44%
improvement, tracking the local CV gain in the same direction, no CV/LB divergence). Pushed
via `kaggle kernels push` (`notebooks/submission_stage4a.ipynb`, kernel
`jjespinoza/rogii-stage-4a-global-gradient-boosted-model`), ran clean on Kaggle's own
infrastructure (0 wells failed), submitted via `kaggle competitions submit -k ... -v 1`. Full
detail in `SUBMISSIONS.md`.

## Stage 4b result (1D CNN sequence model. tested, real result, still worse than 4a)

`src/stage4b_cnn_model.py` convolves over a local window of 41 consecutive eval-zone rows
(instead of Stage 4a treating each row independently). Three attempts on a 30-well
GroupKFold sample before committing to the full run:

| Attempt | Features | Local RMSE (30-well) |
|---|---|---|
| 1 | All 11 Stage 4a features (incl. raw MD/X/Y/Z) | 81.02 |
| 2 | Dropped X/Y, added dropout + weight decay | 84.67 (regressed) |
| 3 | GR + model-output signals only (identity features stripped) | **70.18** |

**Diagnosis:** attempts 1-2 hit near-zero training loss within 1 epoch while held-out RMSE
stayed terrible - the CNN was memorizing per-well identity from absolute location/geometry
(only ~25 wells per fold makes this trivial), not learning transferable GR-shape patterns.
Attempt 3 stripped every well-identifying scalar and improved meaningfully, confirming the
diagnosis - but still fell short of Stage 4a. Per `verify-with-real-data.md`, 3 attempts is
the hard cap regardless of outcome.

**Submitted anyway for a real, honest data point: public LB 72.734** - worse than Stage 4a's
45.196, better than Stage 2's 80.534. Local (70.18) and public (72.73) track closely, no
CV/LB divergence - confirming the validation methodology holds even for a negative result.
**Not the recommended model.** Stage 4a remains the best submitted approach. Full detail in
`context/05-plan-of-attack.md` and `SUBMISSIONS.md`.

## Stage 5 result (ensemble. tested, submitted, small net regression on public LB)

`src/stage5_ensemble.py` blends `linear_prior` (Stage 2), `windowed_match` (Stage 3), and
Stage 4a's GB model via non-negative least squares, using leak-free Stage 4a OOF predictions
(the CNN is excluded, given its 72.734 result). Weights: 0.123 / 0.096 / 0.781 - Stage 4a
dominates but the other two aren't zeroed out.

**Held-out local RMSE: 52.03** (weights fit on 4/5 of wells, evaluated on the other 1/5) vs
Stage 4a alone's 52.90 - a real ~1.6% local improvement.

**Submitted: public LB 45.997** vs Stage 4a alone's 45.196 - a small net REGRESSION
(~1.8% worse), despite the local held-out gain.

**Follow-up (`src/stage5b_blend_variance.py`) - is that gap real or noise?** Repeated the
held-out split 10 times with genuinely random well-to-fold assignment (a first attempt used
sklearn's `GroupKFold`, which turned out to be fully deterministic regardless of input
order - caught because all 10 "repeats" were bit-for-bit identical, fixed with manual random
fold assignment) and compared NNLS against ridge on standardized features. **Result: the
local improvement is real, not noise** - the blend beat Stage 4a alone in every single one
of 10 independent random splits (mean -0.69 RMSE, t-stat ~-10.6). Ridge gave a near-identical,
equally consistent result. So the CV/LB divergence isn't a "local estimate too imprecise"
problem - more likely the public leaderboard (scored on a *sample* of the test data, per the
rules) is itself a noisy point estimate. **Which model is actually best is genuinely
unresolved** - Stage 4a (45.196) is the safer choice since it's the only real confirmed
number for that model, but the blend isn't disproven. Full detail in
`context/05-plan-of-attack.md` and `SUBMISSIONS.md`.

## Stage 6 result (K-means cluster features. tested, negative, not submitted)

`src/stage6_kmeans_clustering.py` adds K-means cluster label + distance-to-centroid as extra
features on top of Stage 4a's feature set. Elbow analysis (k=2..15) found a soft elbow at
**k=6**; Optuna (10 trials, TPE) searched k in [3, 11] using leak-free 3-fold GroupKFold
(KMeans fit within each training fold only), then the best k got an honest 5-fold
confirmation.

**Every single trial scored worse than the no-cluster baseline.** Best k=8: final 5-fold OOF
RMSE **53.05** vs Stage 4a's **52.90** - a small but consistent regression. The gradient-
boosted model's own tree splits already capture whatever structure clustering would add;
an explicit cluster assignment is redundant, not new information. **Not submitted** - no
local evidence of improvement anywhere in the tested range would make spending a submission
worthwhile. Logged here and in the experiment log per the standing rule that every real
attempt gets recorded, win or lose. Full detail in `context/05-plan-of-attack.md`.

## Stage 7 result (guarded physical override, adapted from a public kernel. tested, confirmed inert)

Pulled and studied `lightningv08/rogii-dual-pipeline-self-verifying`
(`context/external-kernels/README.md`) and extracted its **guarded physical override**:
exact TVT reconstruction from a well's formation-contact columns, applied to a test well
only if that well's ID also appears in train AND the reconstruction self-verifies against
the test well's own known prefix at runtime (RMSE < 1 ft). By construction this can only
help or be a no-op - it never overrides without passing self-verification first.

`src/stage7_guarded_override.py` fired correctly on all 3 local visible test wells (known
train-copies): verify RMSE ~0.01 ft, override RMSE ~0.005 ft vs true TVT.

**Submitted: public LB 45.196 - bit-for-bit identical to Stage 4a alone.** The guard did NOT
fire on the real hidden test set - no train/test well-ID overlap exists in the current
(post-rescore) hidden test data. A real, confirmed negative result for this specific exploit
on this competition's current data; not a bug (proven correct by the local sanity check),
and it cost nothing to check since a safe-by-construction override that doesn't fire is a
no-op. Full detail in `context/05-plan-of-attack.md` and `SUBMISSIONS.md`.

## Stage 8 result (cross-well spatial prior, adapted from a public kernel. tested, ~noise-level)

The one genuinely novel signal left untried: every prior stage treats each well in complete
isolation. `FormationPlaneKNN` (same source kernel as Stage 7) fits a locally-weighted plane
of each formation's depth vs (X, Y) using the K=10 nearest OTHER wells, giving a cross-well
estimate of what a formation depth "should be" at any location
(`src/stage8_spatial_prior.py`).

**Attempt 1 (unguarded):** raw plane-fit estimate alone scored RMSE **22,032** -
catastrophically unstable. Diagnosed on well `09ec2ca9`: near-collinear neighbor wells make
the local weighted least-squares system ill-conditioned, producing wildly exploded
extrapolations (raw estimate -51,563 ft vs true ~11,051 ft).

**Attempt 2 (guarded fix):** fall back to `linear_prior` for any row whose spatial estimate
disagrees with it by more than 200 ft. Raw feature alone dropped to a sane RMSE **70**.
Added as a 12th feature to Stage 4a: **5-fold GroupKFold OOF RMSE 52.57 vs 52.90 baseline -
improved in EVERY fold**, a real, reinforcing local signal.

**Submitted: public LB 45.221** vs Stage 4a alone's **45.196** - a tiny (+0.025) difference,
~20x smaller than Stage 5's confirmed-real 0.8-point divergence, and consistent with noise
rather than a real effect. Neither a clear win nor loss - not adopted as the new best model,
but the guarded-KNN feature-engineering pattern is sound and reusable. Full detail in
`context/05-plan-of-attack.md` and `SUBMISSIONS.md`.

## Submission notebook

`notebooks/submission.ipynb` is the actual submission artifact for this **Code Competition**
(rules require submitting a Notebook, not a CLI file-upload). It's self-contained: same
Stage 2 per-well linear fit as `src/baseline.py`, no internet access needed, auto-detects
Kaggle's mounted input path (`/kaggle/input/rogii-wellbore-geology-prediction/`) vs. this
repo's local `../data/` for local testing, and writes `submission.csv` after asserting the
id set exactly matches `sample_submission.csv` with no NaNs.

**Verified locally** (executed end-to-end with `jupyter nbconvert --execute`, output saved
at `submissions/submission_stage2_linear.csv`): 14,151 rows, ids match sample_submission
exactly, `tvt` range ~11,589-12,216 (sane, no NaNs). This is the same non-competitive Stage 2
baseline (local RMSE 67.09) - it locks the submit pipeline, not a leaderboard attempt.

### To actually submit on Kaggle

1. Upload `notebooks/submission.ipynb` as a new Kaggle Notebook, attach the competition
   dataset (`rogii-wellbore-geology-prediction`) as input.
2. **Turn internet off** in notebook settings (required by the rules).
3. Click **Save Version → Save & Run All (Commit)** - not a plain interactive run. Running
   cells in the editor writes `submission.csv` into that live session only; Kaggle's submit
   picker only sees output files attached to a **committed Version**. Wait for the commit to
   finish, then confirm `submission.csv` shows up under that Version's Output tab (check the
   Version's own Logs if it doesn't - a bare `assert` failing on the real hidden test set,
   not just the 3 visible example wells, will silently produce nothing; the notebook code was
   hardened on 2026-07-13 to degrade instead of hard-crash for exactly this reason).
4. **Check your score with the CLI** (more reliable than clicking through the website UI):

   ```
   kaggle competitions submit -c rogii-wellbore-geology-prediction \
     -f submission.csv -k jjespinoza/<NOTEBOOK-SLUG> -v <VERSION-NUMBER> \
     -m "<short note on the approach>"
   ```

   `-k` is the notebook's Kaggle slug (`jjespinoza/<notebook-name>` from its URL), `-v` is the
   committed Version number to pull `submission.csv` from (not the CLI file-submit path used
   for non-Code Competitions - this variant submits a notebook Version's output file).
5. Read the score back with `kaggle competitions submissions -c rogii-wellbore-geology-prediction`
   and log it in the Experiment log below - don't report a number the CLI hasn't returned.

You get 5 submissions/day - this pipeline-proving baseline is a fine use of one of today's,
but save most of them for after Stage 3 (typewell/GR alignment) actually moves the needle.

## Join (one-time, manual. unblocks everything else)

The CLI returns `403` on download/submit until the rules are accepted in the browser.
Open https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/rules → click
**Join Competition** → accept. Then:

```
cd competitions/ROGIIWellboreGeologyPrediction
kaggle competitions download -c rogii-wellbore-geology-prediction -p data/
cd data && unzip -q -o '*.zip' && rm -f *.zip && cd ..
```

> Requires kaggle CLI ≥ 2.x (the `KGAT_` access token in `~/.kaggle/access_token` needs it).
> Installed globally via `uv tool install kaggle` at `~/.local/bin/kaggle`.

## The data (per-well, not flat CSVs)

`data/train/` (~700 wells) and `data/test/` (~200 wells). Each well = an 8-char hash:

| File | What |
|---|---|
| `{hash}__horizontal_well.csv` | The lateral: `MD`, `X`, `Y`, `Z`, `GR`, `TVT_input`, and (train only) `TVT` + formation-depth columns `ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA`. |
| `{hash}__typewell.csv` | Vertical reference log: `TVT` (depth index), `GR`, `Geology` (formation label). Used to correlate the lateral's GR against a known column. |
| `{hash}.png` | (train only) cross-section visualization of the well path + geology. |

**Target:** `tvt` for rows in the evaluation zone, where `TVT_input` is `NaN`. Everywhere
else `TVT_input` is a copy of the true `TVT` and can be used as a feature.

**Submission:** `id,tvt` where `id = {hash}_{row_index}`. See `data/sample_submission.csv`.
The visible `test/` folder is a stand-in copied from train; the real hidden test set is
swapped in when the notebook is rerun.

Full field-by-field dictionary and strategy notes are in `context/`. read those before
modeling. Agents working this competition should start at `context/00-agent-brief.md`.

## Layout

```
data/           per-well train/ + test/ CSVs, sample_submission.csv (gitignored. re-download)
src/            config.py, model + submission code (baseline.py, stage3_*.py, stage4_*.py)
notebooks/      Kaggle submission notebooks + kaggle_push/ (kernel-metadata.json for CLI push)
submissions/    generated submission.csv snapshots (local-test and real Kaggle-scored copies)
context/        agent-facing brief, data dictionary, rules, discussion intel, plan of attack
SUBMISSIONS.md  real leaderboard-score log (Kaggle CLI, not estimated)
```

## Experiment log

| Date | Approach | Local CV RMSE | Public LB | Notes |
|---|---|---|---|---|
| 2026-07-13 | Per-well linear `tvt ~ MD + Z` (Stage 2 baseline) | 67.09 (median 33.07) | **80.534** | Pipeline proven end-to-end. Long extrapolation into eval zone breaks on faults. |
| 2026-07-13 | Pointwise GR/typewell match (Stage 3a) | 74.72 (50-well sample) | not submitted | Worse than Stage 2 - GR alone too noisy for a single-point match. |
| 2026-07-13 | Windowed GR shape-match + gate (Stage 3b) | 75.06 (50-well sample) | not submitted | Still worse - per-well typewell-fit quality too uniform for a simple gate to key off. |
| 2026-07-13 | Global gradient-boosted model (Stage 4a) | 52.90 (773-well GroupKFold OOF) | **45.196** | Real improvement, 21% local / 44% public LB reduction over Stage 2. CV and LB gains tracked in the same direction. Combines linear prior + GR-match signal + geometry via a learned model instead of a hand-tuned gate. **Best model so far.** |
| 2026-07-14 | 1D CNN sequence model (Stage 4b) | 70.18 (30-well GroupKFold, best of 3 attempts) | **72.734** | Worse than Stage 4a. 3 attempts (81.02, 84.67, 70.18) - CNN memorized well identity on the first two; stripping identity features helped but not enough. Local/public tracked closely (no divergence). Honest negative result, documented and submitted anyway. |
| 2026-07-14 | Ensemble: NNLS blend of linear prior + GR-match + Stage 4a (Stage 5) | 52.03 (held-out GroupKFold) | **45.997** | Small net REGRESSION vs Stage 4a alone (45.196), despite a small local held-out gain (52.90 to 52.03). Genuine small CV/LB divergence. Stage 4a alone remains the recommended model. Repeated-CV follow-up (`src/stage5b_blend_variance.py`) showed the blend actually beats Stage 4a in 10/10 random splits (t-stat ~-10.6) - resubmitting confirmed the public score is bit-for-bit stable, ruling out per-submission noise. Which model is truly best remains unresolved; see `SUBMISSIONS.md`. |
| 2026-07-14 | K-means cluster features (Stage 6): cluster label + distance-to-centroid added to Stage 4a's feature set, k tuned via Optuna around the elbow (k=6) | 53.05 (best k=8, 5-fold GroupKFold OOF) | not submitted | Worse than Stage 4a's no-cluster baseline (52.90) - every one of 10 Optuna trials across k=3-11 scored worse. HistGradientBoostingRegressor's own tree splits already capture whatever structure clustering would add; an explicit cluster label is redundant, not new signal. No local evidence of improvement anywhere in the tested range, so not submitted - logged per the standing rule that every real attempt gets recorded. |
| 2026-07-15 | Guarded physical override (Stage 7): exact TVT-from-formation-contacts reconstruction, self-verified before applying, layered on Stage 4a - adapted from a public kernel | n/a (post-processing override) | **45.196** (identical to Stage 4a) | Safe-by-construction guard fired correctly on local sanity-check wells but did NOT fire on the real hidden test set - no train/test well-ID overlap currently exists. Confirmed negative for this exploit, zero cost since it never regresses. |
| 2026-07-15 | Cross-well spatial prior (Stage 8): `FormationPlaneKNN` formation-depth plane fit from K=10 nearest wells, guarded fallback to `linear_prior`, added as a Stage 4a feature - adapted from the same public kernel | 52.57 (5-fold GroupKFold OOF, every fold improved) | **45.221** | Real, consistent local improvement (-0.33 vs 52.90 baseline, every fold) but tiny (+0.025) ~noise-level difference on public LB - not clearly better or worse. First stage to use cross-well information. Fixed a real bug along the way: an unguarded attempt was catastrophically unstable (RMSE 22,032 alone) from ill-conditioned local plane fits. |

Anchor Kanban card: TBD (create on the JJ board via jj-kanban).
