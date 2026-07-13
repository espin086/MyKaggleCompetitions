---
name: kaggle
description: Use when working with Kaggle in this repo — downloading competition data, listing/entering competitions, training a model, and submitting predictions via the kaggle CLI. Triggers on "download the data", "submit to kaggle", "check the leaderboard", "enter competition", "kaggle", or working inside a competitions/<Name>/ folder.
---

# Kaggle CLI workflow

This repo holds one folder per competition under `competitions/`. Each is self-contained
(`data/`, `src/`, submission files). This skill drives the `kaggle` CLI for the full loop:
pull data → build a model → submit → read the score.

## CLI version (must be ≥ 2.x)

The modern Kaggle token is the `KGAT_...` **access token** at `~/.kaggle/access_token`, which
requires **kaggle CLI ≥ 1.8 / 2.x**. That CLI needs **Python ≥ 3.11**, so the system-Python
`pip install kaggle` (Python 3.9) silently pins you to the ancient 1.7.x, which can't read the
new token and returns `401` on every call. This bit us once — the global install is now:

```
uv tool install --python 3.11 "kaggle>=2.0"     # lands at ~/.local/bin/kaggle (first on PATH)
kaggle --version                                 # must print 2.x, not 1.7.x
```

If `kaggle --version` shows 1.7.x, an old shadow copy is winning on PATH — remove it
(`pip3 uninstall kaggle`) so `~/.local/bin/kaggle` resolves.

## Auth (once per machine)

Credentials come from `~/.kaggle/access_token` (`KGAT_...`, needs CLI ≥ 2.x) or the legacy
`~/.kaggle/kaggle.json` (`chmod 600`). Do **not** put placeholder `KAGGLE_USERNAME` /
`KAGGLE_KEY` values in `.claude/settings.local.json` — non-empty env vars override the token
file and force a `401`. Verify with:

```
kaggle competitions list | head
```

If it errors with `401` or "credentials": first check the CLI version (above); then check for
placeholder env vars; then have JJ create a token at kaggle.com → Settings → Create New API
Token and place it in `~/.kaggle/`.

## Layout for a competition

```
competitions/<CompetitionName>/
  data/        raw + generated CSVs (train, test, submission)
  src/         config.py, model.py, notebooks
```

The Kaggle competition slug (e.g. `titanic`, `llm-detect-ai-generated-text`) is the URL tail
from kaggle.com/competitions/<slug>. Keep it in `src/config.py` so commands don't hardcode it.

## Joining a NEW competition (the systematized intake)

When JJ points at a competition (a URL, a screenshot, or a name) and wants to enter it, run
this end-to-end. The output is a self-contained competition folder whose `context/` package
lets any agent (or the `kaggle-ml-loop` skill) start modeling without re-researching.
`competitions/ROGIIWellboreGeologyPrediction/` is the reference example.

1. **Resolve the slug** — the URL tail from `kaggle.com/competitions/<slug>`. Confirm it and
   the reward/deadline with `kaggle competitions list -s <keyword>`.

2. **Scaffold the folder** — `competitions/<CamelCaseName>/{data,src,submissions,context}`.
   Write `src/config.py` (slug, target, id, metric, paths) and a `README.md` (goal, type,
   deadline, join steps, layout, empty experiment-log table).

3. **Gather ALL the competition info** (this is the point — build context once, reuse forever).
   Kaggle pages are JS-rendered, so WebFetch returns only the title — use the browser
   (`claude-in-chrome`, JJ is logged in) and `get_page_text` on each tab:
   - `/overview` → task, **evaluation metric (verbatim)**, submission format, prizes, timeline,
     and whether it's a **Code Competition** (notebook-submit, internet-off, runtime cap).
   - `/data` → the full field-by-field data dictionary and file layout.
   - `/rules` → submission limits/day, team size, data-use (redistribution ban), external-data
     policy, winner obligations.
   - `/discussion?sort=votes` → capture the top threads with vote counts + authors; they are
     the meta-strategy map (which approaches work, known data issues, rescore notices).

4. **Write the `context/` package** — one file each, agent-facing:
   `00-agent-brief.md` (entry point: what to predict, why it's hard, hard constraints, fast
   path), `01-overview.md`, `02-data-dictionary.md`, `03-rules-and-constraints.md`,
   `04-discussion-intel.md`, `05-plan-of-attack.md` (staged plan, each stage with a
   verify-check; ship a submitting baseline before the fancy model).

5. **Gitignore the data if it's large or license-restricted** — competition data is almost
   always "competition use only"; add `competitions/<Name>/data/` to the root `.gitignore` and
   re-download via CLI rather than committing (see Rules below).

6. **Join + download** — accepting the competition rules is a **binding legal agreement**;
   surface it to JJ and let him click **Join Competition** on the website (the CLI can't, and
   it's his agreement to accept). `download`/`submit` return `403` until then. Once joined:
   `kaggle competitions download -c <slug> -p data/ && cd data && unzip -q -o '*.zip' && rm *.zip`.

7. **Create the anchor Kanban card** on the JJ board via `jj-kanban`, and branch + PR the new
   folder/skill changes per `.claude/rules/branch-and-pr.md`.

## Core commands

Run these from the competition folder (`cd competitions/<Name>`).

- **Download data:** `kaggle competitions download -c <slug> -p data/ && unzip -o data/*.zip -d data/`
- **List your competitions:** `kaggle competitions list -s <search>`
- **Submit:** `kaggle competitions submit -c <slug> -f data/submission.csv -m "<short note on the approach>"`
- **Check submissions + score:** `kaggle competitions submissions -c <slug>`
- **Leaderboard (top):** `kaggle competitions leaderboard -c <slug> --show | head`

## Rules

- **Code Competitions submit via Notebook, not CLI file-upload.** If the overview says
  "submissions must be made through Notebooks" (internet-off, runtime cap, `submission.csv`),
  the `kaggle competitions submit -f ...` path does NOT apply — the submission artifact is a
  committed Kaggle Notebook. The local repo is for development + local CV only; port the final
  pipeline into a self-contained notebook (preload any weights as a Kaggle Dataset).
  - **The competition-data mount path is `/kaggle/input/competitions/<slug>/`**, NOT
    `/kaggle/input/<slug>/` — that second form is where an *attached Dataset* lands, and a
    Code Competition's own data is mounted under the `competitions/` subpath instead (confirmed
    from a live Kaggle session's own `os.walk('/kaggle/input')` output on ROGII). Write the
    notebook's path-detection to check `/kaggle/input/competitions/<slug>/` first, and always
    have JJ paste the actual `os.walk` output from a real Kaggle session before trusting a
    guessed path — this bit us once already.
- **Accept the competition rules on the website first** — `submit`/`download` return `403`
  until the rules are accepted in the browser for that competition. Surface this to JJ; the CLI
  can't click the accept button.
- **Never commit `data/` if it's large or license-restricted** — check the competition's data
  license before adding raw files to git; prefer re-downloading via the CLI.
- **Every submission message names the approach** (model + key feature), so the submissions log
  is a readable experiment history.
- **Verify before claiming a score** — after `submit`, run `kaggle competitions submissions` and
  read the actual public score back; don't report a number the CLI hasn't returned.
