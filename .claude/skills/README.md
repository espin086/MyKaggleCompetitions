# skills/

Project-scoped Claude Code skills — reusable playbooks for tasks specific to **this
project** that don't belong in the global `~/.claude/skills/` library.

## When to add a skill here vs global

- **Here** (`.claude/skills/`): the workflow only makes sense for this project — a
  project-specific research pattern, a repeatable data-extraction step, a deliverable
  template unique to this engagement.
- **Global** (`~/.claude/skills/`): the workflow is reusable across projects (task
  triage, email processing, Kanban writes). Don't duplicate a global skill here —
  reference it instead.

## Structure

Each skill is a folder with a `SKILL.md`:

```
skills/
  my-skill-name/
    SKILL.md
```

`SKILL.md` frontmatter:

```yaml
---
name: my-skill-name
description: One line — what it does and when to use it. Specific enough that
  Claude picks it without being told.
---
```

Body: the instructions Claude follows when the skill fires. Keep it a playbook
(steps, checks, output format), not prose about the topic.

## Naming

No fixed prefix requirement at the project level (the `jj-` prefix is a global-config
convention, not required here). Name for clarity: `kebab-case`, verb-first where it
helps triggering ("extract-invoice-totals", not "invoices").

## Placeholder

No project skills yet. Delete this note once the first one is added.
