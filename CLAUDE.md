# CLAUDE.md — {{Project Name}}

Workspace-wide preferences (writing style, project isolation, calendar rules, notification norms) live in `~/.claude/CLAUDE.md`. Drive-specific rules live in `My Drive/CLAUDE.md`. This file documents only what is specific to **{{Project Name}}** plus a progressive-disclosure index into this project's own `.claude/` config.

**Project isolation.** This is a **personal** project — JJ-personal / Haplos only. Never put HD or FCO content here. If a project ever needs HD or FCO material, it does not belong in personal `Projects/` — it lives on the firewalled HD/FCO machine.

## Repo layout

Each Kaggle competition lives in its own subfolder under `competitions/`, self-contained
with its data, notebooks, models, and submission pipeline. Use the `kaggle` CLI to pull
competition data and submit predictions.

## The goal in one paragraph

{{What this project is and why it matters. The outcome that means "done". Frame the stakes in opportunity cost / marginal value where money or time is involved.}}

## Decision rule (locked)

{{If this is a decision, the single rule that ends it — e.g. "Enroll only if discount ≥ 10%." If it's not a decision project, delete this section.}}

## Folder structure

```
competitions/            One subfolder per Kaggle competition, each self-contained.
  TitanicMachineLearningFromDisaster/
  LLMDetectAIGeneratedText/
```

Each competition folder holds its own data, notebooks, models, and submission
pipeline. Add a new competition by creating a new subfolder under `competitions/`.

Root files: `README.md` (quick reference — goal, anchor card, deadline), `TASK.md` (the
executed-task log — see below), this `CLAUDE.md`, and the project's own `.claude/` config
(see below).

## TASK.md — the executed-task log

`TASK.md` at the project root is the running history of every task worked on this project. The
`jj-work` skill (`/jj-work <CARD-ID>`) maintains it: it **reads** `TASK.md` before starting a
card — to avoid repeating work and to build on prior stages — and **appends a row** when the card
is done (date, Kanban card ID, one-line summary, deliverable path). Append-only, newest at the
bottom. It is deliberately separate from `README.md` (which holds the goal/anchor/deadline) so
the history and the orientation don't fight in one file. The card ID is plain text there — Drive
doesn't resolve Obsidian wikilinks.

## This project's `.claude/` config (progressive disclosure)

The project ships its own `.claude/` folder for anything specific to **{{Project Name}}** —
config that shouldn't live in the global `~/.claude/` because it only applies here. Each
subfolder has its own `README.md` explaining what belongs in it and how to add to it; read the
specific one when a task touches that area, not before.

| Folder | Holds | Read when |
|---|---|---|
| Skills | @.claude/skills/README.md | Adding a project-specific reusable workflow. |
| Hooks | @.claude/hooks/README.md | Enforcing a project-specific automated guard. |
| Agents | @.claude/agents/README.md | Defining a project-specific subagent role. |
| Rules | @.claude/rules/README.md | Adding a project-specific cross-cutting constraint. |
| Memory | @.claude/memory/README.md | Recording a project-specific fact, feedback, or decision worth recalling later. |
| Settings + MCP | `.claude/settings.json` | Project tool permissions and hook wiring. MCP servers (Playwright, GitHub, Context7, etc.) are connected account-wide — this file only grants the project permission to use them. |

All five subfolders start empty except for their `README.md`. Don't inline a skill, hook, agent,
rule, or memory body into this file — add the file in its folder and reference it.

## Kanban sync — the read/write contract with the board and the vault

Most projects here are spawned from a card on JJ's Obsidian Kanban board (JJ or haplos). This
project folder is a **satellite** of that card, not a replacement for it — the board stays the
source of truth for task state; this folder holds the actual work product. Three things must
stay true at all times:

1. **This project links to its card, not the other way around only.** The anchor card ID goes
   in `README.md` as a `[[id-slug|ID]]` wikilink (see the template line there). If the project
   spans multiple cards (a big engagement), list each one in `04-action-plan/`.
2. **The card's detail note links back to this folder.** When `jj-kanban` creates or enriches
   the anchor card, its `Tasks/**` note carries a `## Links` line pointing at this project (a
   `My Drive/Projects/{{Project Name}}` path or, once deliverables exist, the specific file).
   This is the same card↔source-note pattern as `~/.claude/rules/kanban-card-linking.md` —
   just pointed at a Drive folder instead of a vault note.
3. **Read before you write; write when you finish a stage.** Before starting a card's work,
   read `TASK.md` (what's already been done here) and the card's `## Update` history (what's
   already been decided or found) so the two don't diverge. When a stage of work finishes:
   - **Read from Kanban/Obsidian:** any new `## Update` on the card, any new note the board
     side has added, so this folder never falls behind board-side decisions.
   - **Write back to Kanban/Obsidian:** append a `TASK.md` row here, then call `jj-kanban
     --comment <ID> "..."` (or `--done <ID>` if the card is finished) so the board reflects
     what this folder now contains. If the work produced a note worth keeping independent of
     the card (a decision writeup, a research summary), save it via `jj-note` so it gets
     enriched and backlinked in the vault, and link it from `04-action-plan/` or `TASK.md`.

**Enforcement.** `.claude/hooks/kanban-obsidian-sync-check.sh` runs on session `Stop` and
reminds Claude to do step 3's write-back if `TASK.md`, `README.md`, `04-action-plan/`, or
`05-deliverables/` changed since the last sync (tracked via the `.claude/.last-kanban-sync`
marker, touched after a sync). It's a reminder, not a hard block — JJ can defer a sync and end
the session anyway, but the reminder won't clear until the sync happens.

## Source of truth

| What | Where |
|------|-------|
| Source brief (why this matters) | {{`~/Documents/obsidian/Areas/.../<brief>.md`}} |
| Related Drive folders | {{e.g. `Areas/Finance/Trackers/`}} |
| Kanban board | {{JJ or haplos — `Projects/Kanban - <board>.md` in the Obsidian vault}} |

## Action items (current)

Tasks are tracked on JJ's Obsidian Kanban boards (via the `jj-kanban` skill) and linked per
`~/.claude/rules/kanban-card-linking.md` — every task reference is a clickable
`[[id-slug|ID]]` wikilink to its `Tasks/**` detail note.

1. {{First next step.}}
2. {{…}}

## Working conventions

- **Research uses Playwright in HEADED mode.** When this project needs live web research — checking a form, a policy, a price, availability, a competitor, anything behind JS or a login — drive a real browser via the Playwright MCP in **headed** mode (visible window, `headless: false`), not headless. JJ wants to watch it run, and headed rendering handles gated/JS-heavy pages that headless trips on. Take screenshots into `02-data-and-documents/`. Still obey the source tiers in `~/.claude/rules/high-value-research-sources.md`.
- **Full Disk Access required.** This folder is under GoogleDrive-CloudStorage; reads/writes fail with "Operation not permitted" without it.
- **Google Workspace stubs.** `.gdoc/.gsheet/.gslides` files are pointers, not content — open in browser.
- **Naming.** Documents: `YYYY_{{project}}_{doc-type}.pdf`. Screenshots/confirmations: `YYYY-MM-DD_{{project}}_{event}.png`.
- **Writing style.** Follow the AI-tells ban list in `~/.claude/CLAUDE.md`. Plain words, no padding.
- **Artifacts to the vault.** Any written brief/report/summary this project produces also lands in the vault root `~/Documents/obsidian/` per `~/.claude/rules/skill-output-to-vault.md`.

## Open questions

- {{Unknowns to resolve before the decision can be made.}}
