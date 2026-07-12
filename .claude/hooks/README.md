# hooks/

Shell scripts the Claude Code harness runs automatically on lifecycle events
(`PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `Stop`, `SessionStart`, etc.). Hooks
run whether or not Claude "remembers" to do the thing. this is the only reliable way
to enforce an automated behavior ("always X before Y", "never Y without Z").

## When to add a hook here vs global

- **Here** (`.claude/hooks/`): the guard is specific to this project (e.g. block writes
  to a project's `02-data-and-documents/` without a source citation, warn before
  touching a specific sensitive file in this project).
- **Global** (`~/.claude/hooks/`): the guard applies everywhere (sensitive-folder
  warnings, no-HD-FCO-email, anti-AI writing checks). Don't re-implement a global hook
  here. it already fires on every project.

## Wiring a hook

1. Write the script here, executable (`chmod +x`).
2. Register it in `.claude/settings.json` under `hooks`, e.g.:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [{ "type": "command", "command": ".claude/hooks/my-guard.sh" }]
      }
    ]
  }
}
```

3. Test it manually before trusting it. hooks fail silently in ways that are easy to
   miss (wrong matcher, wrong exit code, path not executable).

## Exit code contract

- `0`. allow, no message.
- Non-zero on `PreToolUse`. blocks the tool call; stderr is shown to Claude as the
  reason.
- Any hook can print a `system-reminder`-style message to stdout to inject context
  without blocking.

## Shipped with this template

- **`kanban-obsidian-sync-check.sh`** (Stop, async, non-blocking). reminds Claude to sync
  `TASK.md` / `README.md` / `04-action-plan/` / `05-deliverables/` changes back to the source
  Kanban card and the Obsidian vault before the session ends, if those files changed more
  recently than the `.claude/.last-kanban-sync` marker. See the project `CLAUDE.md` section
  "Kanban sync" for the full contract this hook enforces.

Add project-specific hooks alongside it as the project needs them.
