# Rules & operational constraints (from the Rules tab, 2026-07-13)

Sponsor: ROGII (Houston TX). Governing law: Texas. Full legal text is on the Rules tab; this
file captures only what changes how you work.

## Operational limits

- **Team size:** max 5. Mergers allowed until 2026-07-29.
- **Submissions:** **5 per day.** Select up to **2** final submissions for private-LB judging.
- **One account only.** No submitting from multiple accounts.

## Code / submission mechanics (this is a Code Competition)

- Submit via **Kaggle Notebook**, not a prediction-file upload.
- Notebook must run **≤ 9 hours** (CPU or GPU) with **internet disabled**.
- Output file must be named exactly `submission.csv`.
- No hand-labeling or human prediction of the test records.

## Data use

- **Competition use only.** Do not transmit, duplicate, publish, or redistribute the
  competition data to anyone not in the competition. → `data/` stays gitignored, never pushed.
- **External data & tools allowed** if publicly available and equally accessible to all
  participants at minimal cost (the "Reasonableness Standard"). Pretrained models OK. AutoML
  tools OK if properly licensed.

## Code sharing

- **No private sharing** of code/data outside your team.
- Public sharing is allowed **only** on the competition's Kaggle forums/notebooks, and by
  sharing you license it under an OSI-approved license. If you use open-source code in your
  model it must be under an OSI-approved license permitting commercial use.

## Winner obligations (if you place)

- Deliver full training + inference code, documentation, and a reproducible environment
  description. Non-exclusive perpetual license to the sponsor. Possible recorded call.
- Standard eligibility: 18+, not in sanctioned regions, tax forms (W-9/W-8BEN).

## Practical takeaways for JJ

- Because it's notebook-submit + internet-off, the local repo is for **development and local
  CV only**. The final artifact is a self-contained Kaggle notebook that loads any pretrained
  weights as a Kaggle Dataset (uploaded ahead of time) rather than downloading at runtime.
- Budget the 5/day submissions: spend early ones locking the pipeline, save late-competition
  ones for real model comparisons. Pick 2 diverse finals (don't pick two near-identical subs).
