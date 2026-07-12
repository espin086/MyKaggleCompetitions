---
name: kaggle
description: Use when working with Kaggle in this repo — downloading competition data, listing/entering competitions, training a model, and submitting predictions via the kaggle CLI. Triggers on "download the data", "submit to kaggle", "check the leaderboard", "enter competition", "kaggle", or working inside a competitions/<Name>/ folder.
---

# Kaggle CLI workflow

This repo holds one folder per competition under `competitions/`. Each is self-contained
(`data/`, `src/`, submission files). This skill drives the `kaggle` CLI for the full loop:
pull data → build a model → submit → read the score.

## Auth (once per machine)

Credentials come from `~/.kaggle/kaggle.json` (`chmod 600`) or the `KAGGLE_USERNAME` /
`KAGGLE_KEY` env vars wired in `.claude/settings.local.json` (gitignored — never commit the
key). Verify with:

```
kaggle competitions list | head
```

If it errors with `401` or "credentials", auth isn't set — stop and tell JJ to create a token
at kaggle.com → Settings → Create New API Token, then place `kaggle.json` in `~/.kaggle/`.

## Layout for a competition

```
competitions/<CompetitionName>/
  data/        raw + generated CSVs (train, test, submission)
  src/         config.py, model.py, notebooks
```

The Kaggle competition slug (e.g. `titanic`, `llm-detect-ai-generated-text`) is the URL tail
from kaggle.com/competitions/<slug>. Keep it in `src/config.py` so commands don't hardcode it.

## Core commands

Run these from the competition folder (`cd competitions/<Name>`).

- **Download data:** `kaggle competitions download -c <slug> -p data/ && unzip -o data/*.zip -d data/`
- **List your competitions:** `kaggle competitions list -s <search>`
- **Submit:** `kaggle competitions submit -c <slug> -f data/submission.csv -m "<short note on the approach>"`
- **Check submissions + score:** `kaggle competitions submissions -c <slug>`
- **Leaderboard (top):** `kaggle competitions leaderboard -c <slug> --show | head`

## Rules

- **Accept the competition rules on the website first** — `submit`/`download` return `403`
  until the rules are accepted in the browser for that competition. Surface this to JJ; the CLI
  can't click the accept button.
- **Never commit `data/` if it's large or license-restricted** — check the competition's data
  license before adding raw files to git; prefer re-downloading via the CLI.
- **Every submission message names the approach** (model + key feature), so the submissions log
  is a readable experiment history.
- **Verify before claiming a score** — after `submit`, run `kaggle competitions submissions` and
  read the actual public score back; don't report a number the CLI hasn't returned.
