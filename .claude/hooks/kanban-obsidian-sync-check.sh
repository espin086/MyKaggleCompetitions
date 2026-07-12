#!/usr/bin/env bash
# Stop hook: reminds Claude to sync this project's state back to the Kanban
# board and Obsidian vault before the session ends, if project files changed
# and no sync has happened since.
#
# Non-blocking (async, informational) — matches the house pattern in
# ~/.claude/hooks/jj-obsidian-para-router.sh. It nudges via a printed
# reminder rather than blocking Stop, because most projects here are worked
# interactively and JJ wants the session to end even if he defers the sync.
#
# Marker file: .claude/.last-kanban-sync — touch this (or let jj-kanban /
# jj-work touch it) after posting the Kanban update, so this hook goes quiet
# until the next real change.

set -uo pipefail

cwd="${CLAUDE_PROJECT_DIR:-$PWD}"
cd "$cwd" 2>/dev/null || exit 0

marker=".claude/.last-kanban-sync"
watched_paths=(TASK.md README.md 04-action-plan 05-deliverables)

# Nothing to compare against yet — treat "never synced" as in-scope only if
# there's already project content worth syncing.
marker_epoch=0
[ -f "$marker" ] && marker_epoch=$(stat -f %m "$marker" 2>/dev/null || stat -c %Y "$marker" 2>/dev/null || echo 0)

newest_change=0
for p in "${watched_paths[@]}"; do
  [ -e "$p" ] || continue
  while IFS= read -r -d '' f; do
    m=$(stat -f %m "$f" 2>/dev/null || stat -c %Y "$f" 2>/dev/null || echo 0)
    [ "$m" -gt "$newest_change" ] && newest_change="$m"
  done < <(find "$p" -type f -print0 2>/dev/null)
done

[ "$newest_change" -eq 0 ] && exit 0
[ "$newest_change" -le "$marker_epoch" ] && exit 0

cat <<EOF
{"systemMessage": "Project files changed since the last Kanban sync (TASK.md, README.md, 04-action-plan/, or 05-deliverables/). Before ending this session: (1) update the source Kanban card via jj-kanban --comment/--done with a one-line summary and the deliverable path, (2) confirm the card's detail note in Tasks/** links back here (obsidian:// or a relative path) and this project's README/TASK.md links the card as [[id-slug|ID]], (3) touch .claude/.last-kanban-sync so this reminder clears. See this project's CLAUDE.md section \"Kanban sync\" for the full contract."}
EOF
exit 0
