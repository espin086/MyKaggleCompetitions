# Submissions log

Public leaderboard scores for each submission, newest at the bottom. Rows are appended
automatically by `python src/submit.py "message"` once a submission is scored.

| Date (UTC) | Message | Public score | Status |
|---|---|---|---|
| 2026-07-13 | v1: gradient boosting + engineered features (local 5-fold CV 0.8406 ± 0.012) | 0.76315 | complete |

Note: the earlier 2024 export scored 0.78229. v1's CV (0.8406) overshoots the public
leaderboard (0.76315) — a sign of optimistic CV / mild overfit. Next: a held-out validation
split, lighter model / regularization, and feature review before the next submit.
