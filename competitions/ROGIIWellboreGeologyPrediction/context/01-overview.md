# Overview (from the competition Overview tab, 2026-07-13)

**Host:** ROGII (oil-and-gas software, Houston TX). Featured Code Competition.
**Prizes:** $50,000 total. 1st $25k, 2nd $13k, 3rd $7k, 4th $5k. Plus two $2,500 Working
Note Awards (optional writeup, medal-zone teams, was due 2026-07-06).

## The problem

Drilling a horizontal well is navigating underground without a map. the path runs through
rock layers you can't see. ~10,000 horizontal wells are drilled worldwide per year, still
largely steered by manual expert interpretation. Small deviations from the target zone waste
resources. Subsurface measurements (wells, seismic, logging tools) only show part of the
picture; layers start stacked "like a layer cake" but bend and break along faults, so it's
hard to know where the bit sits within the formation.

Task: build ML models that predict the geology encountered along a horizontal wellbore. identify favorable layers from drilling data to guide well placement.

## Evaluation

Root Mean Squared Error on the target `tvt`:

> RMSE = sqrt( (1/n) · Σ (yᵢ − ŷᵢ)² )

Lower is better. Submission format:

```
id,tvt
000d7d20_1442,0.0
000d7d20_1443,0.0
...
```

One row per prediction point in the (hidden) test set. `id = {WELLNAME}_{row_index}`.

## Timeline (all 23:59 UTC)

- 2026-05-05. Start
- 2026-07-06. Working-note award deadline (optional, passed)
- 2026-07-29. Entry deadline (must accept rules by this date) + team-merger deadline
- 2026-08-05. **Final submission deadline**

## Code requirements

Submissions are made through **Notebooks**. For the Submit button to activate after a commit:

- CPU or GPU notebook ≤ 9 hours runtime
- **Internet access disabled**
- Freely & publicly available external data allowed, including pretrained models
- Submission file must be named `submission.csv`

## Scoring nuance

Public leaderboard = a representative sample of test data; **Private leaderboard = the private
test set and determines final standing.** There was a "Private Test Update and Rescore"
(pinned discussion by Ryan Holbrook) and a "[EDIT] Dataset issue - Fixed!". check those
threads before trusting any historical LB numbers.
