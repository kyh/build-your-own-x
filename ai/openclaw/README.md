# Build Your Own OpenClaw

A minimal reproduction of [OpenClaw](https://github.com/nichochar/openclaw) — a persistent, multi-agent AI assistant with tools, memory, and sessions — in a single TypeScript file.

This is a learning exercise. The goal is to understand how persistent AI assistants work by building one from scratch: sessions, tool use, memory, permissions, compaction, multi-agent routing, and scheduled tasks. Every concept maps directly to a function you can read and modify.

## What it does

- **Multi-agent system** — Two agents (Jarvis and Scout) with different personalities, routed by prefix commands
- **Tool use** — Agents can run shell commands, read/write files, and manage long-term memory
- **Persistent sessions** — Conversations stored as JSONL files, survive restarts
- **Context compaction** — Automatically summarizes old messages when context gets too long
- **Long-term memory** — Keyword-searchable markdown files shared across agents and sessions
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
| `/research <query>`  | Route message to Scout (researcher)  |
| `/new`               | Reset session (start fresh)          |
| `/quit`              | Exit                                 |

## Architecture

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
  -> resolveAgent (route to Jarvis or Scout)
  -> runAgentTurn
       -> loadSession (JSONL)
       -> compactSession (summarize if needed)
       -> loop: Claude API -> serialize -> tool execution -> feed back
  -> print response
```

### Sessions

Each session is a JSONL file — one JSON object per line, each representing a message. Messages are appended as they happen (`appendMessage`), so if the process crashes mid-conversation you lose at most one message. Compaction and session resets use `saveSession` to overwrite the full file.

Session keys like `agent:main:repl` get sanitized to filesystem-safe names (`agent_main_repl.jsonl`). Different agents and heartbeats use different session keys, so their histories stay separate.

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

### Permission controls

When the agent tries to run a command:

1. Check if the base command is in `SAFE_COMMANDS` (e.g. `ls`, `git`, `node`) — allow immediately
2. Check if the exact command was previously approved — allow
3. Otherwise, prompt the user interactively and persist their decision to `exec-approvals.json`

Approvals are stored per exact command string. Approving `curl https://example.com` doesn't approve `curl https://evil.com`.

### Multi-agent routing

Messages are routed by `resolveAgent` based on prefix commands. `/research <query>` routes to Scout (the researcher agent), everything else goes to Jarvis (the main agent). Each agent has:

- **A soul** — System prompt that defines personality and behavior
- **A model** — Which Claude model to use
- **A session prefix** — Keeps conversation histories separate

Agents share the same memory directory, so Scout can save research findings that Jarvis can later search for.

### Context compaction

Sessions grow over time. When `estimateTokens` (a rough chars/4 heuristic) exceeds 100k:

1. Split messages in half — older half and recent half
2. Send the older half to Claude with a "summarize this" prompt
3. Replace the older half with a single summary message
4. Overwrite the session file with the compacted history

The agent keeps its knowledge but the token count drops significantly. This happens transparently before each turn.

### Heartbeats

Heartbeats let the agent act without user input. A `setInterval` runs every 60 seconds and checks the current time:

```
every 60s:
  if time === "07:30" AND date !== lastHeartbeatDate:
    lastHeartbeatDate = today
    runAgentTurn("cron:morning-check", "Good morning! ...", mainAgent)
```

The `lastHeartbeatDate` guard prevents double-firing (the 60s interval might hit 07:30 twice). The heartbeat uses its own session key (`cron:morning-check`) so it doesn't pollute the REPL conversation.

Since Node.js is single-threaded, the heartbeat timer only fires between await points. If the user is mid-conversation (waiting on a Claude API call), the heartbeat queues up and fires after the current turn completes. No locks needed.

### Workspace layout

```
~/.mini-openclaw/
  sessions/              # JSONL conversation files
    agent_main_repl.jsonl
    agent_researcher_repl.jsonl
    cron_morning-check.jsonl
  memory/                # Markdown memory files (shared across agents)
    user-preferences.md
    research-findings.md
  exec-approvals.json    # Persistent command approvals
```

## Things to try

- **Add a new agent** — Add an entry to `AGENTS` and a routing rule in `resolveAgent`
- **Add a new tool** — Add a schema to `TOOLS` and a case in `executeTool`
- **Change the heartbeat schedule** — Modify the time check in `setupHeartbeats`
- **Swap the memory search** — Replace keyword matching with embeddings for semantic search
- **Add a new channel** — The agent loop is decoupled from the REPL; you could wire it to a Discord bot or HTTP API
