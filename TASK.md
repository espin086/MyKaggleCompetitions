# {{Project Name}}. task log

The running history of every task executed on this project. The `jj-work` skill
(`/jj-work <CARD-ID>`) reads this file before it starts a card. to avoid repeating work and to
build on prior stages. and appends a row here when the card is done. Append-only, newest at the
bottom.

This is separate from the other root files:
- `README.md`. quick reference: the goal, the anchor Kanban card, the deadline.
- `04-action-plan/great-plan.md`. the hardened phase-by-phase plan (from the `jj-master-plan` loop).
- `TASK.md` (this file). what has actually been *done*, card by card.

One row per executed card: the date, the Kanban card ID (plain text. Drive doesn't resolve
Obsidian wikilinks), a one-line summary, and the deliverable path (usually under `05-deliverables/`).

## Tasks executed

| Date | Card | What was done | Deliverable |
|---|---|---|---|
| {{YYYY-MM-DD}} | {{JJ-000}} | {{one line. what this card produced}} | {{05-deliverables/<file>}} |
