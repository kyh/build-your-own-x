# Build Your Own OpenClaw

A minimal reproduction of [OpenClaw](https://github.com/nichochar/openclaw) — a persistent, multi-agent AI assistant with tools, memory, and sessions — in a single TypeScript file.

## What it does

- **Multi-agent system** — Jarvis (general assistant) and Scout (research specialist), routed by prefix commands
- **Tool use** — Agents can run shell commands, read/write files, and manage long-term memory
- **Persistent sessions** — Conversations stored as JSONL files, survive restarts
- **Context compaction** — Automatically summarizes old messages when context gets too long
- **Long-term memory** — Keyword-searchable markdown files shared across agents and sessions
- **Permission controls** — Safe command allowlist with persistent user approvals
- **Scheduled heartbeats** — Daily 07:30 agent trigger via `setInterval`

## Usage

```sh
# from repo root
npm install
export ANTHROPIC_API_KEY=sk-...
npx tsx ai/openclaw/openclaw.ts
```

## REPL commands

| Command              | Description                          |
| -------------------- | ------------------------------------ |
| `/research <query>`  | Route message to Scout (researcher)  |
| `/new`               | Reset session (start fresh)          |
| `/quit`              | Exit                                 |

## How it works

The core loop is `runAgentTurn`:

1. Load session history from JSONL
2. Compact if over ~100k estimated tokens
3. Call Claude with system prompt (soul), tools, and messages
4. If Claude returns `tool_use`, execute each tool and feed results back
5. Repeat until Claude returns `end_turn` (max 20 iterations)
6. Every message is appended to the session file as it happens

```
User input
  -> resolveAgent (route to Jarvis or Scout)
  -> runAgentTurn
       -> loadSession (JSONL)
       -> compactSession (summarize if needed)
       -> loop: claude API -> serialize -> tool execution -> feed back
  -> print response
```

### Workspace layout

```
~/.mini-openclaw/
  sessions/          # JSONL conversation files
  memory/            # Markdown memory files (shared across agents)
  exec-approvals.json  # Persistent command approvals
```

### Key design decisions

- **Single-threaded** — No locks needed. Node's event loop handles concurrency naturally.
- **`execSync` for commands** — Blocks the event loop, which is fine for a REPL.
- **Type guards over assertions** — Tool `input` from the SDK is `unknown`. Runtime guards (`isRecord`, `getString`) validate at the boundary instead of using `as` casts.
- **`setInterval` for heartbeats** — Checks time every 60s with a date guard to prevent double-firing.
