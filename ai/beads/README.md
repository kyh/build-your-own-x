# Build Your Own Beads

A minimal reproduction of [Beads](https://github.com/steveyegge/beads) — a git-backed, dependency-aware issue tracker designed for AI coding agents — in a single TypeScript file.

This is a learning exercise. The goal is to understand how agent memory systems work by building one from scratch: issue tracking, hash-based IDs, dependency graphs, and the "ready queue" that tells agents what to work on next.

## The problem Beads solves

AI coding agents have what Steve Yegge calls the "50 First Dates" problem — every session starts from zero. The agent doesn't remember what it worked on yesterday, what's blocked, or what's ready. Markdown TODO files don't scale: they're not machine-readable, dependencies aren't queryable, and agents waste tokens parsing them.

Beads replaces markdown plans with a structured, dependency-aware task graph. The key insight: instead of asking the LLM to analyze a dependency graph (expensive, error-prone), compute it deterministically in code and give the agent only what's actionable.

## Usage

```sh
# from repo root
pnpm install
npx tsx ai/beads/beads.ts <command>
```

Issues are stored in `.beads/issues.jsonl` in the current directory — just add `.beads/` to your project's git repo for version-controlled task tracking.

## Commands

```sh
bd create <title>              # Create an issue
bd create <title> -t bug auth  # Create with tags
bd list                        # List open issues
bd list --all                  # Include closed issues
bd show <id>                   # Show details + dependency info
bd update <id> --status in-progress
bd close <id>                  # Close an issue
bd link <from> <to>            # "from" blocks "to"
bd unlink <from> <to>          # Remove dependency
bd ready                       # Show unblocked issues (the key command)
bd ready --json                # JSON output for programmatic use
```

## Example workflow

```sh
$ bd create "Implement login page"
Created bd-a3f8: Implement login page

$ bd create "Add auth middleware" -t backend
Created bd-7c01: Add auth middleware

$ bd create "Build dashboard UI"
Created bd-e5b2: Build dashboard UI

# Dashboard depends on both login and auth
$ bd link bd-a3f8 bd-e5b2
Linked: bd-a3f8 (Implement login page) blocks bd-e5b2 (Build dashboard UI)

$ bd link bd-7c01 bd-e5b2
Linked: bd-7c01 (Add auth middleware) blocks bd-e5b2 (Build dashboard UI)

# What should I work on? Only unblocked issues
$ bd ready
  2 issue(s) ready:
  [ ] bd-a3f8  Implement login page
  [ ] bd-7c01  Add auth middleware (backend)

# Complete login, check again
$ bd close bd-a3f8
Closed bd-a3f8: Implement login page

$ bd ready
  1 issue(s) ready:
  [ ] bd-7c01  Add auth middleware (backend)
  # Dashboard still blocked — auth middleware isn't done yet

$ bd close bd-7c01
$ bd ready
  1 issue(s) ready:
  [ ] bd-e5b2  Build dashboard UI
  # Now dashboard is unblocked
```

## Architecture

### Issue model

Each issue has an ID, title, description, status, tags, and a list of issue IDs it blocks. That's it — minimal by design.

```typescript
type Issue = {
  id: string;           // hash-based, e.g. "bd-a3f8"
  title: string;
  description: string;
  status: "open" | "in-progress" | "closed";
  tags: string[];
  blocks: string[];     // IDs of issues this blocks
  createdAt: string;
  updatedAt: string;
};
```

### Hash-based IDs

IDs are generated from a SHA-256 hash of the title + timestamp + random salt, truncated to 4 hex characters (`bd-a3f8`). This is important for multi-agent workflows: if two agents create issues on different git branches, sequential IDs (1, 2, 3) would collide on merge, but hash-based IDs won't.

### JSONL storage

Issues are stored as JSON Lines — one JSON object per line. This format is ideal for git because:
- Adding an issue appends one line (clean diff)
- Modifying an issue changes one line
- Merge conflicts are rare and easy to resolve

The real Beads uses Dolt (a version-controlled SQL database) for more sophisticated storage, but JSONL + git gives us the same version control benefits with zero dependencies.

### Dependency graph

Dependencies are stored as a `blocks` array on each issue. `A.blocks = ["B"]` means "A blocks B" — B can't be worked on until A is closed.

The `getTransitiveBlockers` function computes the full blocking chain using BFS. If A blocks B and B blocks C, then A transitively blocks C. This computation is done in code, not by the LLM — that's the key architectural insight of Beads. Graph algorithms are deterministic and free; LLM inference is expensive and unreliable for structured reasoning.

### The ready queue

`getReadyIssues` is the most important function. It filters open issues to only those with zero open transitive blockers, sorted by creation date. This gives agents a deterministic answer to "what should I do next?" without burning any tokens on analysis.

```
All issues
  -> filter: not closed
  -> filter: no open transitive blockers
  -> sort: oldest first
  = ready queue
```

### Storage layout

```
.beads/
  issues.jsonl    # One issue per line, git-trackable
```

## What the real Beads adds

Our reproduction covers the core concepts. The real Beads (~130k lines of Go) adds:

- **Dolt database** — Version-controlled SQL with cell-level merge
- **Hierarchical subtasks** — `bd-a3f8.1`, `bd-a3f8.2` under a parent
- **More dependency types** — Related, conditional blocking, discovered-from links
- **Thread-based messaging** — Comments and discussions on issues
- **Memory compaction** — Summarizing closed issues to save context window space
- **MCP server mode** — Direct integration with Claude and other agents
- **Multi-agent conflict resolution** — Sophisticated merge strategies

## Things to try

- **Add subtask IDs** — Generate child IDs like `bd-a3f8.1` under a parent issue
- **Add compaction** — When closing an issue, summarize its history into a short blurb
- **Add an MCP server** — Expose the commands as MCP tools for direct agent integration
- **Add priority** — Numeric priority field, use it to sort the ready queue
- **Add cycle detection** — Prevent circular dependencies (A blocks B blocks A)
