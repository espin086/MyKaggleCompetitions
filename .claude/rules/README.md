# rules/

Cross-cutting behavioral rules for this project — the same progressive-disclosure
pattern JJ's global config uses: `CLAUDE.md` stays a lean index (one line per rule),
and the full rule body lives here in its own file. Loaded every session; read the
specific file when a task touches that rule.

## When to add a rule here vs global

- **Here** (`.claude/rules/`): a constraint specific to this project (e.g. "always
  cite the source document page number", "never quote client-confidential figures
  outside `05-deliverables/`").
- **Global** (`~/.claude/rules/`): the constraint applies to every project JJ touches
  (writing style, HD/FCO isolation, calendar rules, verify-with-real-data). Those are
  already loaded from `~/.claude/CLAUDE.md` — don't restate them here.

## Structure

One file per rule, referenced from the project's root `CLAUDE.md` by a one-line row
plus an `@`-path so it progressive-discloses:

```
rules/
  my-rule-name.md
```

```markdown
---
name: my-rule-name
status: active
---

# Rule: <name>

What it says, why (the incident or preference behind it), and how to apply it.
```

## Adding a rule

1. Write the rule file here.
2. Add one row to the project `CLAUDE.md`'s rules table with an `@.claude/rules/<file>.md`
   pointer — never inline the rule body in `CLAUDE.md` itself.

## Placeholder

No project-specific rules yet. Delete this note once the first one is added.
