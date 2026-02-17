# Build Your Own OpenClaw

A minimal reproduction of [OpenClaw](https://github.com/nichochar/openclaw) — a persistent AI assistant with tools, memory, skills, and sessions — in a single TypeScript file.

This is a learning exercise. The goal is to understand how persistent AI assistants work by building one from scratch: sessions, tool use, memory, skills, permissions, compaction, and scheduled tasks. Every concept maps directly to a function you can read and modify.

## What it does

- **Tool use** — Run shell commands, read/write files, and manage long-term memory
- **Installable skills** — Markdown instruction files that teach the agent new capabilities
- **Persistent sessions** — Conversations stored as JSONL files, survive restarts
- **Context compaction** — Automatically summarizes old messages when context gets too long
- **Long-term memory** — Keyword-searchable markdown files shared across sessions
- **Permission controls** — Safe command allowlist with persistent user approvals
- **Scheduled heartbeats** — Daily agent trigger that runs autonomously on a timer

## Usage

```sh
# from repo root
pnpm install
export ANTHROPIC_API_KEY=sk-...
npx tsx ai/openclaw/openclaw.ts
```

Requires Node.js >= 17 (for `readline/promises`).

## REPL commands

| Command              | Description                          |
| -------------------- | ------------------------------------ |
| `/new`               | Reset session (start fresh)          |
| `/quit`              | Exit                                 |

## Architecture

```
┌──────────────────────┐
│   Scratch Pads       │
│                      │
│  soul.md             │    ┌───────────────────────┐
│                      │    │  Skills (Installable) │
│  memory/*.md         │    │                       │
│  sessions/*.jsonl    │    │  skills/*.md          │
│  exec-approvals.json │    │  loaded on demand via │
│                      │    │  file tools           │
└──────────┬───────────┘    └───────────┬───────────┘
       read│write                       │
           │                  installed│loaded
           ▼                            │
  ┌──────────────────────────────┐      │
  │                              │◄─────┘
  │        Agent Loop            │
  │       (runAgentTurn)         │
  │                              │        ┌──────────────┐
  │   load session               │        │              │
  │   compact if needed          │───────►│  UI/Whatsapp │
  │   call Claude ◄──► tools     │◄───────│   (Gateway)  │
  │   loop until end_turn        │        │              │
  │                              │        └──────────────┘
  │                              │
  └──────────────┬───────────────┘
                 ▲
                 │
  ┌──────────────┴───────────────┐
  │   Heartbeat: setInterval     │
  │   (WAKE UP! every 60s)       │
  │                              │
  │   07:30 daily ──► agent turn │
  └──────────────────────────────┘
```

### The agent loop

The core of the system is `runAgentTurn`. Every interaction — whether from a user or a scheduled heartbeat — flows through this single function:

1. **Load session** — Read JSONL conversation history from disk
2. **Compact if needed** — If estimated tokens exceed ~100k, summarize older messages via a separate Claude call and replace them with a summary
3. **Append user message** — Add to session and persist immediately
4. **Call Claude** — Send system prompt (the agent's "soul"), tools, and full message history
5. **Handle response** — If Claude returns `tool_use`, execute each tool, feed results back, and loop. If `end_turn`, return the text response
6. **Max 20 iterations** — Safety limit to prevent runaway tool loops

```
User input
  -> runAgentTurn
       -> loadSession (JSONL)
       -> compactSession (summarize if needed)
       -> loop: Claude API -> serialize -> tool execution -> feed back
  -> print response
```

OpenClaw uses [Pi](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent) as their coding agent

### Sessions

Each session is a JSONL file — one JSON object per line, each representing a message. Messages are appended as they happen (`appendMessage`), so if the process crashes mid-conversation you lose at most one message. Compaction and session resets use `saveSession` to overwrite the full file.

Session keys like `repl` get sanitized to filesystem-safe names (`repl.jsonl`). The REPL and heartbeats use different session keys, so their histories stay separate.

### Tools

Five tools are defined in the `TOOLS` array, each with a JSON schema that Claude uses to generate structured input:

| Tool | What it does |
| --- | --- |
| `run_command` | Execute a shell command (with permission checks) |
| `read_file` | Read a file (truncated to 10k chars) |
| `write_file` | Write a file (creates parent directories) |
| `save_memory` | Write a markdown file to `~/.mini-openclaw/memory/` |
| `memory_search` | Keyword search across all memory files |

The SDK returns tool input as `unknown`. Rather than using type assertions, we validate at the boundary with `isRecord` and `getString` guards.

### Skills

Skills are installable markdown files that teach the agent new capabilities. They live in `~/.mini-openclaw/skills/` and the agent manages them with its existing file tools — no dedicated skill tools needed.

The agent can write new skills, read them when relevant, list the directory to see what's installed, and delete ones that are no longer needed. Like memory, skills are loaded on demand rather than always present in context, so they scale without bloating the system prompt.

### Permission controls

When the agent tries to run a command:

1. Check if the base command is in `SAFE_COMMANDS` (e.g. `ls`, `git`, `node`) — allow immediately
2. Check if the exact command was previously approved — allow
3. Otherwise, prompt the user interactively and persist their decision to `exec-approvals.json`

Approvals are stored per exact command string. Approving `curl https://example.com` doesn't approve `curl https://evil.com`.

### Context compaction

Sessions grow over time. When `estimateTokens` (a rough chars/4 heuristic) exceeds 100k:

1. Split messages in half — older half and recent half
2. Send the older half to Claude with a "summarize this" prompt
3. Replace the older half with a single summary message
4. Overwrite the session file with the compacted history

The agent keeps its knowledge but the token count drops significantly. This happens transparently before each turn.

### Heartbeats

Heartbeats let the agent act without user input. They're configured in `~/.mini-openclaw/heartbeats.json` (created with a default 07:30 entry on first run):

```json
[
  { "time": "07:30", "prompt": "Good morning! Give me a motivational quote." },
  { "time": "12:00", "prompt": "Remind me to take a break." }
]
```

A `setInterval` runs every 60 seconds, reads the config, and fires each heartbeat once per day at its scheduled time:

```
every 60s:
  for each heartbeat in heartbeats.json:
    if time === heartbeat.time AND not yet fired today:
      runAgentTurn("cron:{time}", heartbeat.prompt)
```

The agent manages heartbeats by reading and writing `heartbeats.json` with its existing file tools — no dedicated heartbeat tools needed. Each heartbeat uses its own session key (e.g. `cron:0730`) so histories stay separate.

Since Node.js is single-threaded, heartbeat timers only fire between await points. If the user is mid-conversation, the heartbeat queues up and fires after the current turn completes. No locks needed.

### Workspace layout

```
~/.mini-openclaw/
  soul.md                # System prompt (created on first run, editable)
  heartbeats.json        # Scheduled heartbeats (created on first run)
  sessions/              # JSONL conversation files
    repl.jsonl
    cron_0730.jsonl
  memory/                # Markdown memory files
    user-preferences.md
    research-findings.md
  skills/                # Installable skill files (loaded on demand)
    code-review.md
    debugging.md
  exec-approvals.json    # Persistent command approvals
```

## Things to try

- **Add a new tool** — Add a schema to `TOOLS` and a case in `executeTool`
- **Swap the memory search** — Replace keyword matching with embeddings for semantic search
- **Add a new channel** — The agent loop is decoupled from the REPL; you could wire it to a Discord bot or HTTP API
