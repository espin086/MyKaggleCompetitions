# memory/

Persistent, file-based memory scoped to this project. facts, feedback, and
project-state notes worth recalling in a future session on this project, separate
from JJ's global cross-project memory.

## Types of memory (same model as global memory)

| Type | What it holds |
|---|---|
| `user` | How JJ specifically works this project. his role here, what he already knows, what he wants explained vs skipped. |
| `feedback` | Corrections and confirmations from JJ about *how* to work this project. what approach he validated, what he shut down and why. |
| `project` | Facts about this project's state. decisions made, deadlines, who's involved, why a direction was chosen. This is the type used most here. |
| `reference` | Pointers to where up-to-date info lives for this project (a tracker sheet, a Drive folder, a specific person's notes). |

## File shape

One file per memory, plus an index:

```
memory/
  MEMORY.md          # index. one line per memory, links to the file
  project_<slug>.md  # the memory body
```

`MEMORY.md` entry: `- [Title](project_slug.md). one-line hook`.

Memory file frontmatter:

```yaml
---
name: project-slug
description: One-line summary for relevance matching
metadata:
  type: project   # user | feedback | project | reference
---
```

## What NOT to put here

- Anything derivable by reading the project's own files (`README.md`, `great-plan.md`
  successor, `TASK.md`). that's the project's live state, not memory.
- Ephemeral in-progress task details. those belong in `TASK.md` or the Kanban card,
  not memory.
- Anything already covered by a global memory in `~/.claude/.../memory/`. don't fork it.

## Placeholder

No project memories saved yet. Delete this note once the first one is added.
