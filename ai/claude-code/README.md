# Mini Claude Code

A minimal clone of [Claude Code](https://docs.anthropic.com/en/docs/claude-code) in ~250 lines. Implements the core agentic loop: send a prompt to Claude, let it call tools (read/write files, run shell commands), feed results back, repeat until done.

## How it works

```
User prompt
    ↓
┌─────────────────────────────┐
│  Claude API (tool_use mode) │
│  System: "You are a helpful │
│  coding assistant..."       │
└──────────┬──────────────────┘
           │
     ┌─────▼─────┐
     │ end_turn?  │──yes──→ Print response
     └─────┬──────┘
           │ no (tool_use)
           ▼
   ┌───────────────┐
   │ Execute tools  │
   │ (with permission│
   │  checks)       │
   └───────┬───────┘
           │
           ▼
   Append tool results
   as "user" message
           │
           └──→ Loop back to Claude API
```

### Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read a file's contents |
| `write_file` | Create or overwrite a file |
| `list_files` | List directory contents |
| `run_command` | Execute a shell command (30s timeout) |

### Permission system

- **`write_file`** — always prompts for confirmation
- **`run_command`** — prompts if the command contains dangerous patterns (`rm`, `sudo`, `chmod`, `mv`, `cp`, `>`, `>>`)
- **`read_file`**, **`list_files`** — no confirmation needed

## Usage

```bash
# Requires ANTHROPIC_API_KEY env var
export ANTHROPIC_API_KEY=sk-ant-...

# TypeScript
npx tsx ai/claude-code/claude-code.ts
```
