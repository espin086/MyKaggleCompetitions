---
name: branch-and-pr
status: active
---

# Rule: branch then pull request — never commit to main directly

## What this rule says

Never commit or push straight to `main` in this repo. Every change ships through a
short-lived feature branch and a pull request that a human reviews and merges:

1. **Branch off up-to-date main.** `git checkout main && git pull` then
   `git checkout -b <type>/<short-topic>` (e.g. `feat/titanic-xgboost`,
   `chore/kaggle-auth`). Types: `feat`, `fix`, `chore`, `refactor`, `docs`.
2. **Commit on the branch.** Conventional-commit messages
   (`type(scope): summary`), one logical change per commit, ending with the
   `Co-Authored-By: Claude Opus 4.8` trailer.
3. **Push the branch** (`git push -u origin <branch>`), never `main`.
4. **Open a PR** with `gh pr create` (or the GitHub MCP): title = the change,
   body = what changed, why, and how it was verified. Base is `main`.
5. **Stop there.** A human reviews and merges. The agent does not merge its own
   PR or push to `main`, and never force-pushes.

## Guards (also enforced in `.claude/settings.json`)

- `Bash(git push origin main:*)` and force-push are denied. Branch pushes and
  `gh pr` are allowed.
- If a push to `main` is ever needed, that is a human decision — surface it, don't
  route around the deny.

## Why

`main` is the shared, deployable history for a repo more than one person (or one
machine) touches. A PR gives a review gate, a CI hook point, and a readable record
of each change, and it keeps a bad commit from landing on `main` unnoticed.

## When this rule does NOT apply

- The `~/.claude` config repo, which has its own standing
  `auto-commit-push-claude-repo.md` rule (commit straight to `main`). That override
  is scoped to that repo only and does not reach here.

## Applies to

Every agent, skill, and bare edit whose changes land in this repo.
