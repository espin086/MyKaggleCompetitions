# agents/

Project-scoped subagent definitions — specialized personas the `Agent` tool can
dispatch (a fresh context, a narrower toolset, a specific system prompt) for
recurring sub-tasks inside this project.

## When to add an agent here vs global

- **Here** (`.claude/agents/`): a role that only makes sense inside this project (a
  "deed-reviewer" agent for a real-estate decision project, a "vendor-comparison"
  agent for a specific research engagement).
- **Global** (`~/.claude/agents/`): a role reusable across projects (`jj-code-backend`,
  `jj-research`, `jj-code-security`). Don't fork a global agent here — invoke it directly.

## Structure

One markdown file per agent:

```
agents/
  my-agent-name.md
```

Frontmatter:

```yaml
---
name: my-agent-name
description: What it's for and when to dispatch it — specific enough that Claude
  or JJ picks it correctly without guessing.
tools: Read, Grep, Glob      # narrow to what the role actually needs
model: sonnet                 # optional override; omit to inherit the session model
---
```

Body: the agent's system prompt — its scope, what it refuses, its output format.

## Design notes

- Keep agents narrow. A read-only reviewer should not have `Write`/`Edit`.
- State explicitly what the agent refuses to do (mirrors the global agents' pattern
  of refusing HD/FCO work, refusing to invoke `/code` directly, etc.) — this prevents
  scope creep when the agent is dispatched with a terse prompt.

## Placeholder

No project agents yet. Delete this note once the first one is added.
