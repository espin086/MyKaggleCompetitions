# ROGII - Wellbore Geology Prediction - Submissions log

Real scores from `kaggle competitions submissions -c rogii-wellbore-geology-prediction`.
Lower RMSE is better. Public LB leader for reference: ~4.86 (as of 2026-07-13).

| Date | Kaggle ref | Approach | Local CV RMSE | Public LB | Notes |
|---|---|---|---|---|---|
| 2026-07-13 | 54662772 | Stage 2: per-well linear `tvt ~ MD + Z` | 67.09 (median 33.07) | **80.534** | Pipeline-proving baseline. Extrapolates a straight line into the eval zone; breaks on faults. |
| 2026-07-13 | 54664284 | Stage 4a: global `HistGradientBoostingRegressor` (linear prior + windowed GR/typewell match + geometry as features, GroupKFold-by-well OOF) | 52.90 (773-well OOF) | **45.196** | 44% improvement over Stage 2. Public LB gain (80.53 to 45.2) tracked the local CV gain (67.09 to 52.90) in the same direction - no CV/LB divergence. Still a flat/tabular model, not a true sequence model. **Best model so far.** |
| 2026-07-14 | 54671461 | Stage 4b: 1D CNN over local windows (`GR` + model-output signals only, identity features stripped after 2 failed attempts) | 70.18 (30-well GroupKFold) | **72.734** | Real, honest result - worse than Stage 4a (45.196), better than Stage 2 (80.534). Local (70.18) and public (72.73) track closely, no CV/LB divergence, confirming the validation methodology even for a negative result. 3 attempts tested (81.02, 84.67, 70.18 local); hard-capped per `verify-with-real-data.md`. NOT the recommended model - kept for the honest record. Stage 4a remains best. |
| 2026-07-14 | 54672595 | Stage 5: NNLS ensemble of `linear_prior` + `windowed_match` + Stage 4a GB (CNN excluded, per Stage 4b's result). Weights: 0.123 / 0.096 / 0.781. | 52.03 (held-out GroupKFold, honest estimate) | **45.997** | Small CV/LB divergence: local held-out CV suggested a modest improvement over Stage 4a (52.90 to 52.03, ~1.6%), but the public LB is actually slightly WORSE (45.196 to 45.997, ~1.8% worse). At this small an effect size, the local gain didn't generalize. **Stage 4a alone remains the best real submitted model** - the blend isn't recommended as-is. See `src/stage5b_blend_variance.py` for a follow-up investigating whether this gap is noise. |
| 2026-07-14 | 54694271 | Stage 5 v2: identical predictions to the row above (same deterministic model, resubmitted as-is) | 52.03 (unchanged) | **45.997** (identical) | Resubmitted the exact same predictions to test whether the public score is stable. **Confirmed identical, bit-for-bit** - Kaggle scores the public leaderboard against a FIXED sample of the test set, not a re-drawn one per submission. Rules out submission-to-submission randomness as an explanation for the CV/LB gap; only a genuinely different model or the private leaderboard at competition close can resolve which approach is actually better. |
| 2026-07-15 | 54712045 | Stage 7: Stage 4a base model + a **guarded physical override** (exact TVT-from-formation-contacts reconstruction, self-verified against each test well's own known prefix before applying) - adapted from public kernel `lightningv08/rogii-dual-pipeline-self-verifying` | n/a (guard is a post-processing override, not a separately-trained model) | **45.196** (identical to Stage 4a alone) | The guard is safe by construction (only overrides after self-verification passes) - it fired correctly on all 3 local visible test wells (known train-copies, verify RMSE ~0.01 ft) but produced an **identical score to Stage 4a alone** on the real hidden test set, meaning it did NOT fire there - no train/test well-ID overlap in the current (post-rescore) hidden test data. A real, confirmed negative result: this specific exploit doesn't apply to this competition's current test set, but it cost nothing (no regression) to check. |
| 2026-07-15 | 54713031 | Stage 8: Stage 4a + a **guarded cross-well spatial prior** feature (`FormationPlaneKNN` - locally-weighted plane fit of formation depths vs (X,Y) from the K=10 nearest OTHER wells, guarded to fall back to `linear_prior` when the estimate is unstable) - adapted from the same public kernel. First stage to use ANY cross-well information; every prior stage treated wells in complete isolation. | 52.57 (5-fold GroupKFold OOF, improved in EVERY fold vs Stage 4a's 52.90) | **45.221** | A real, consistent local improvement (-0.33 RMSE, every fold) came back as a tiny, likely-noise-level difference on the public LB (+0.025 vs Stage 4a's 45.196) - not a meaningful gain OR loss. Root-caused and fixed one real bug along the way: an unguarded first attempt was catastrophically unstable (raw feature RMSE 22,032 alone, from ill-conditioned local plane fits when neighbor wells are near-collinear) - the guard (fallback to `linear_prior` beyond 200ft disagreement) fixed it to RMSE 70 alone and is what made the augmented model improve locally at all. |

## How each was submitted

All six went through the real Code Competition path: pushed as a Kaggle Notebook (`kaggle
kernels push`), executed on Kaggle's own infrastructure against
`/kaggle/input/competitions/rogii-wellbore-geology-prediction/`, then submitted via
`kaggle competitions submit -f submission.csv -k jjespinoza/<slug> -v <version>`. Score
read back with `kaggle competitions submissions -c rogii-wellbore-geology-prediction` -
not guessed or estimated.

- Stage 2: `notebooks/submission.ipynb`
- Stage 4a: `notebooks/submission_stage4a.ipynb`, kernel
  `jjespinoza/rogii-stage-4a-global-gradient-boosted-model`
- Stage 4b: `notebooks/submission_stage4b.ipynb`, kernel
  `jjespinoza/rogii-stage-4b-1d-cnn-sequence-model`
- Stage 5: `notebooks/submission_stage5.ipynb`, kernel
  `jjespinoza/rogii-stage-5-ensemble-blend`
- Stage 7: `notebooks/submission_stage7.ipynb`, kernel
  `jjespinoza/rogii-stage-7-guarded-physics-override`
- Stage 8: `notebooks/submission_stage8.ipynb`, kernel
  `jjespinoza/rogii-stage-8-cross-well-spatial-prior`

## Current recommendation

**Which model is actually best is genuinely unresolved** between Stage 4a alone (45.196,
the only real submitted number for that model) and the Stage 5 blend. Follow-up analysis
(`src/stage5b_blend_variance.py`) repeated the held-out validation 10 times with genuinely
random well-to-fold splits (a first attempt used sklearn's `GroupKFold`, which turned out to
be deterministic regardless of input order - fixed): the blend beat Stage 4a alone in
**every single repeat** (mean -0.69 RMSE, t-stat ~-10.6) - a real, low-variance local signal,
not noise. Ridge regularization on standardized features gave a near-identical, equally
consistent result.

Resubmitted the identical Stage 5 predictions a second time to test whether the public score
was itself noisy per-submission - **confirmed bit-for-bit identical (45.997 both times)**.
Kaggle scores the public leaderboard against a fixed test-data sample, not a re-drawn one per
submission, so submission-to-submission randomness is ruled out as the explanation. The
divergence is more likely a genuine difference between the training wells' distribution and
the specific (fixed) public sample - not something a re-submit of the same model can resolve.
Only a genuinely different/improved model, or the private leaderboard at competition close,
can settle which approach is actually better. Lead with Stage 4a for now since it's the
safer, confirmed number, but don't treat the blend as disproven - the local evidence for it
remains strong.
