#!/usr/bin/env npx tsx
// mini-openclaw.ts - A minimal OpenClaw clone in TypeScript
// Run: npx tsx ai/openclaw/openclaw.ts

import Anthropic from "@anthropic-ai/sdk";
import { execSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
  appendFileSync,
  readdirSync,
} from "node:fs";
import { homedir } from "node:os";
import { join, dirname } from "node:path";
import { createInterface, type Interface } from "node:readline/promises";

// ─── Types ───

type Heartbeat = {
  time: string;
  prompt: string;
};

type Approvals = {
  allowed: string[];
  denied: string[];
};

type SerializedBlock =
  | { type: "text"; text: string }
  | { type: "tool_use"; id: string; name: string; input: unknown };

// ─── Configuration ───

const WORKSPACE = join(homedir(), ".mini-openclaw");
const SESSIONS_DIR = join(WORKSPACE, "sessions");
const MEMORY_DIR = join(WORKSPACE, "memory");
const SKILLS_DIR = join(WORKSPACE, "skills");
const APPROVALS_FILE = join(WORKSPACE, "exec-approvals.json");
const HEARTBEATS_FILE = join(WORKSPACE, "heartbeats.json");

// ─── Agent ───

const MODEL = "claude-sonnet-4-5-20250929";
const SOUL_PATH = join(WORKSPACE, "soul.md");

const DEFAULT_SOUL = `\
You are Jarvis, a personal AI assistant.
Be genuinely helpful. Skip the pleasantries. Have opinions.
You have tools — use them proactively.

## Skills

Skills are markdown instruction files in ${SKILLS_DIR} — manage them with your file tools.
Write new skills there, read them when relevant, list the directory to see what's installed.

## Memory

Use save_memory to store important information across sessions.
Use memory_search at the start of conversations to recall context.

## Heartbeats

You can schedule daily heartbeats that trigger you autonomously at specific times.
Heartbeats are configured in ${HEARTBEATS_FILE} — read and write it with your file tools.
`;

/** Load the agent's soul (system prompt) from soul.md, creating the default on first run. */
function loadSoul(): string {
  if (!existsSync(SOUL_PATH)) {
    writeFileSync(SOUL_PATH, DEFAULT_SOUL);
  }
  return readFileSync(SOUL_PATH, "utf-8");
}

// ─── Tools ───

const TOOLS = [
  {
    name: "run_command",
    description: "Run a shell command",
    input_schema: {
      type: "object" as const,
      properties: {
        command: { type: "string" as const, description: "The command to run" },
      },
      required: ["command"],
    },
  },
  {
    name: "read_file",
    description: "Read a file from the filesystem",
    input_schema: {
      type: "object" as const,
      properties: {
        path: { type: "string" as const, description: "Path to the file" },
      },
      required: ["path"],
    },
  },
  {
    name: "write_file",
    description: "Write content to a file (creates directories if needed)",
    input_schema: {
      type: "object" as const,
      properties: {
        path: { type: "string" as const, description: "Path to the file" },
        content: {
          type: "string" as const,
          description: "Content to write",
        },
      },
      required: ["path", "content"],
    },
  },
  {
    name: "save_memory",
    description: "Save important information to long-term memory",
    input_schema: {
      type: "object" as const,
      properties: {
        key: {
          type: "string" as const,
          description: "Short label (e.g. 'user-preferences')",
        },
        content: {
          type: "string" as const,
          description: "The information to remember",
        },
      },
      required: ["key", "content"],
    },
  },
  {
    name: "memory_search",
    description: "Search long-term memory for relevant information",
    input_schema: {
      type: "object" as const,
      properties: {
        query: {
          type: "string" as const,
          description: "What to search for",
        },
      },
      required: ["query"],
    },
  },
] satisfies Anthropic.Tool[];

// ─── Client ───

const client = new Anthropic();

// ─── Type Guards ───

/** Narrow unknown values to a plain object. */
function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** Validate JSONL-deserialized data as a MessageParam (boundary guard). */
function isMessageParam(v: unknown): v is Anthropic.MessageParam {
  if (!isRecord(v)) return false;
  if (v.role !== "user" && v.role !== "assistant") return false;
  return v.content !== undefined;
}

/** Safely extract a string from an unknown tool input object, or throw. */
function getString(obj: Record<string, unknown>, key: string): string {
  const val = obj[key];
  if (typeof val !== "string") {
    throw new Error(`Expected string for "${key}", got ${typeof val}`);
  }
  return val;
}

// ─── Permission Controls ───

const SAFE_COMMANDS = new Set([
  "ls",
  "cat",
  "head",
  "tail",
  "wc",
  "date",
  "whoami",
  "echo",
  "pwd",
  "which",
  "git",
  "python",
  "node",
  "npm",
]);

/** Load the persistent approval/denial list from disk. */
function loadApprovals(): Approvals {
  if (existsSync(APPROVALS_FILE)) {
    const raw: unknown = JSON.parse(readFileSync(APPROVALS_FILE, "utf-8"));
    if (
      isRecord(raw) &&
      Array.isArray(raw.allowed) &&
      Array.isArray(raw.denied)
    ) {
      return {
        allowed: raw.allowed.filter((x): x is string => typeof x === "string"),
        denied: raw.denied.filter((x): x is string => typeof x === "string"),
      };
    }
  }
  return { allowed: [], denied: [] };
}

/** Persist a user's approval or denial of a specific command. */
function saveApproval(command: string, approved: boolean): void {
  const approvals = loadApprovals();
  const key = approved ? "allowed" : "denied";
  if (!approvals[key].includes(command)) {
    approvals[key].push(command);
  }
  writeFileSync(APPROVALS_FILE, JSON.stringify(approvals, null, 2));
}

/** Check if a command is safe, previously approved, or needs user approval. */
function checkCommandSafety(
  command: string,
): "safe" | "approved" | "needs_approval" {
  const baseCmd = command.trim().split(/\s+/)[0] ?? "";
  if (SAFE_COMMANDS.has(baseCmd)) return "safe";
  const approvals = loadApprovals();
  if (approvals.allowed.includes(command)) return "approved";
  return "needs_approval";
}

// ─── Tool Execution ───

/** Execute a tool by name, dispatching to the appropriate handler. */
async function executeTool(
  name: string,
  toolInput: unknown,
  rl: Interface,
): Promise<string> {
  if (!isRecord(toolInput)) return "Error: invalid tool input";

  switch (name) {
    case "run_command": {
      const cmd = getString(toolInput, "command");
      const safety = checkCommandSafety(cmd);
      if (safety === "needs_approval") {
        const confirm = await rl.question(
          `\n  ⚠️  Command: ${cmd}\n  Allow? (y/n): `,
        );
        if (confirm.trim().toLowerCase() !== "y") {
          saveApproval(cmd, false);
          return "Permission denied by user.";
        }
        saveApproval(cmd, true);
      }
      try {
        const output = execSync(cmd, {
          encoding: "utf-8",
          timeout: 30_000,
          stdio: ["pipe", "pipe", "pipe"],
        });
        return output || "(no output)";
      } catch (err: unknown) {
        if (isRecord(err) && typeof err.status === "number") {
          const stdout = typeof err.stdout === "string" ? err.stdout : "";
          const stderr = typeof err.stderr === "string" ? err.stderr : "";
          return stdout + stderr || `Command exited with code ${err.status}`;
        }
        return `Error: ${err instanceof Error ? err.message : String(err)}`;
      }
    }

    case "read_file": {
      const filePath = getString(toolInput, "path");
      try {
        return readFileSync(filePath, "utf-8").slice(0, 10_000);
      } catch (err: unknown) {
        return `Error: ${err instanceof Error ? err.message : String(err)}`;
      }
    }

    case "write_file": {
      const filePath = getString(toolInput, "path");
      const content = getString(toolInput, "content");
      try {
        mkdirSync(dirname(filePath), { recursive: true });
        writeFileSync(filePath, content);
        return `Wrote to ${filePath}`;
      } catch (err: unknown) {
        return `Error: ${err instanceof Error ? err.message : String(err)}`;
      }
    }

    case "save_memory": {
      const key = getString(toolInput, "key");
      const content = getString(toolInput, "content");
      try {
        mkdirSync(MEMORY_DIR, { recursive: true });
        writeFileSync(join(MEMORY_DIR, `${key}.md`), content);
        return `Saved to memory: ${key}`;
      } catch (err: unknown) {
        return `Error saving memory: ${err instanceof Error ? err.message : String(err)}`;
      }
    }

    case "memory_search": {
      const query = getString(toolInput, "query").toLowerCase();
      const words = query.split(/\s+/);
      const results: string[] = [];
      const MAX_RESULTS = 10;
      try {
        if (existsSync(MEMORY_DIR)) {
          for (const fname of readdirSync(MEMORY_DIR)) {
            if (!fname.endsWith(".md")) continue;
            const content = readFileSync(join(MEMORY_DIR, fname), "utf-8");
            if (words.some((w) => content.toLowerCase().includes(w))) {
              results.push(`--- ${fname} ---\n${content}`);
              if (results.length >= MAX_RESULTS) break;
            }
          }
        }
      } catch (err: unknown) {
        return `Error searching memory: ${err instanceof Error ? err.message : String(err)}`;
      }
      return results.length > 0
        ? results.join("\n\n")
        : "No matching memories found.";
    }

    default:
      return `Unknown tool: ${name}`;
  }
}

// ─── Session Management ───

/** Convert a session key (e.g. "agent:main:repl") to a safe filesystem path. */
function getSessionPath(sessionKey: string): string {
  const safeKey = sessionKey.replace(/[:/]/g, "_");
  return join(SESSIONS_DIR, `${safeKey}.jsonl`);
}

/** Load conversation history from a JSONL file on disk. */
function loadSession(sessionKey: string): Anthropic.MessageParam[] {
  const path = getSessionPath(sessionKey);
  const messages: Anthropic.MessageParam[] = [];
  if (!existsSync(path)) return messages;
  const lines = readFileSync(path, "utf-8").split("\n");
  for (const line of lines) {
    if (!line.trim()) continue;
    try {
      const parsed: unknown = JSON.parse(line);
      if (isMessageParam(parsed)) {
        messages.push(parsed);
      }
    } catch {
      console.warn(`  ⚠ Skipping malformed JSONL line in session "${sessionKey}"`);
    }
  }
  return messages;
}

/** Append a single message to the session file (crash-safe append-only write). */
function appendMessage(
  sessionKey: string,
  message: Anthropic.MessageParam,
): void {
  appendFileSync(getSessionPath(sessionKey), JSON.stringify(message) + "\n");
}

/** Overwrite the session file with the full message list. */
function saveSession(
  sessionKey: string,
  messages: Anthropic.MessageParam[],
): void {
  const content = messages.map((m) => JSON.stringify(m)).join("\n") + "\n";
  writeFileSync(getSessionPath(sessionKey), content);
}

// ─── Compaction ───

/** Rough token estimate: ~4 chars per token. */
function estimateTokens(messages: Anthropic.MessageParam[]): number {
  return messages.reduce((sum, m) => sum + JSON.stringify(m).length, 0) / 4;
}

/** When context exceeds ~100k tokens, summarize older messages to free space. */
async function compactSession(
  sessionKey: string,
  messages: Anthropic.MessageParam[],
): Promise<Anthropic.MessageParam[]> {
  if (estimateTokens(messages) < 100_000) return messages;

  const split = Math.floor(messages.length / 2);
  const old = messages.slice(0, split);
  const recent = messages.slice(split);

  console.log("\n  📦 Compacting session history...");

  const summary = await client.messages.create({
    model: MODEL,
    max_tokens: 2000,
    messages: [
      {
        role: "user",
        content:
          "Summarize this conversation concisely. Preserve key facts, " +
          "decisions, and open tasks:\n\n" +
          JSON.stringify(old, null, 2),
      },
    ],
  });

  const summaryText = summary.content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("");

  const compacted: Anthropic.MessageParam[] = [
    { role: "user", content: `[Conversation summary]\n${summaryText}` },
    ...recent,
  ];

  saveSession(sessionKey, compacted);
  return compacted;
}

// ─── Content Serialization ───

/** Convert SDK response content blocks to plain JSON-serializable objects for JSONL storage. */
function serializeContent(
  content: Anthropic.ContentBlock[],
): SerializedBlock[] {
  const serialized: SerializedBlock[] = [];
  for (const block of content) {
    if (block.type === "text") {
      serialized.push({ type: "text", text: block.text });
    } else if (block.type === "tool_use") {
      serialized.push({
        type: "tool_use",
        id: block.id,
        name: block.name,
        input: block.input,
      });
    }
  }
  return serialized;
}

// ─── Agent Loop ───

/** Run a full agent turn: load session, call LLM in a loop (up to 20 iterations), handle tool use. */
async function runAgentTurn(
  sessionKey: string,
  userText: string,
  rl: Interface,
): Promise<string> {
  let messages = loadSession(sessionKey);
  messages = await compactSession(sessionKey, messages);

  const userMsg: Anthropic.MessageParam = { role: "user", content: userText };
  messages.push(userMsg);
  appendMessage(sessionKey, userMsg);

  for (let i = 0; i < 20; i++) {
    const response = await client.messages.create({
      model: MODEL,
      max_tokens: 4096,
      system: loadSoul(),
      tools: TOOLS,
      messages,
    });

    const content = serializeContent(response.content);
    const assistantMsg: Anthropic.MessageParam = {
      role: "assistant",
      content,
    };
    messages.push(assistantMsg);
    appendMessage(sessionKey, assistantMsg);

    if (response.stop_reason === "end_turn") {
      return response.content
        .filter((b): b is Anthropic.TextBlock => b.type === "text")
        .map((b) => b.text)
        .join("");
    }

    if (response.stop_reason === "tool_use") {
      const toolResults: Anthropic.ToolResultBlockParam[] = [];
      for (const block of response.content) {
        if (block.type !== "tool_use") continue;
        const inputPreview = JSON.stringify(block.input).slice(0, 100);
        console.log(`  🔧 ${block.name}: ${inputPreview}`);
        const result = await executeTool(block.name, block.input, rl);
        console.log(`     → ${result.slice(0, 150)}`);
        toolResults.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: result,
        });
      }

      const resultsMsg: Anthropic.MessageParam = {
        role: "user",
        content: toolResults,
      };
      messages.push(resultsMsg);
      appendMessage(sessionKey, resultsMsg);
    }
  }

  return "(max turns reached)";
}

// ─── Cron / Heartbeats ───

const DEFAULT_HEARTBEATS: Heartbeat[] = [
  {
    time: "07:30",
    prompt: "Good morning! Check today's date and give me a motivational quote.",
  },
];

/** Validate an unknown value as a Heartbeat. */
function isHeartbeat(v: unknown): v is Heartbeat {
  return isRecord(v) && typeof v.time === "string" && typeof v.prompt === "string";
}

/** Load heartbeats from config, creating the default on first run. */
function loadHeartbeats(): Heartbeat[] {
  if (!existsSync(HEARTBEATS_FILE)) {
    writeFileSync(HEARTBEATS_FILE, JSON.stringify(DEFAULT_HEARTBEATS, null, 2));
  }
  const raw: unknown = JSON.parse(readFileSync(HEARTBEATS_FILE, "utf-8"));
  if (!Array.isArray(raw)) return [];
  return raw.filter(isHeartbeat);
}

/** Check all configured heartbeats every 60s, firing each once per day at its scheduled time. */
function setupHeartbeats(rl: Interface): void {
  const lastFired = new Map<string, string>();

  setInterval(() => {
    const now = new Date();
    const time = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;
    const date = now.toISOString().slice(0, 10);

    const heartbeats = loadHeartbeats();
    for (let i = 0; i < heartbeats.length; i++) {
      const hb = heartbeats[i];
      if (hb === undefined) continue;
      const heartbeatKey = `${hb.time}:${i}`;
      if (hb.time === time && lastFired.get(heartbeatKey) !== date) {
        lastFired.set(heartbeatKey, date);
        const sessionKey = `cron:${hb.time.replace(":", "")}_${i}`;
        console.log(`\n⏰ Heartbeat: ${hb.time}`);
        runAgentTurn(sessionKey, hb.prompt, rl)
          .then((result) => {
            console.log(`🤖 ${result}\n`);
          })
          .catch((err: unknown) => {
            console.error(`  ❌ Heartbeat failed: ${err instanceof Error ? err.message : String(err)}`);
          });
      }
    }
  }, 60_000);
}

// ─── REPL ───

/** Initialize workspace, start heartbeats, and run the interactive REPL. */
async function main(): Promise<void> {
  for (const dir of [WORKSPACE, SESSIONS_DIR, MEMORY_DIR, SKILLS_DIR]) {
    mkdirSync(dir, { recursive: true });
  }

  const rl = createInterface({ input: process.stdin, output: process.stdout });

  setupHeartbeats(rl);

  let sessionKey = "repl";

  console.log("Mini OpenClaw");
  console.log(`  Workspace: ${WORKSPACE}`);
  console.log("  Commands: /new (reset), /quit\n");

  while (true) {
    let userInput: string;
    try {
      userInput = await rl.question("You: ");
    } catch {
      console.log("\nGoodbye!");
      break;
    }

    userInput = userInput.trim();
    if (!userInput) continue;

    if (["/quit", "/exit", "/q"].includes(userInput.toLowerCase())) {
      console.log("Goodbye!");
      break;
    }

    if (userInput.toLowerCase() === "/new") {
      const timestamp = new Date()
        .toISOString()
        .replace(/[-:T]/g, "")
        .slice(0, 14);
      sessionKey = `repl:${timestamp}`;
      console.log("  Session reset.\n");
      continue;
    }

    try {
      const response = await runAgentTurn(sessionKey, userInput, rl);
      console.log(`\n🤖 ${response}\n`);
    } catch (err: unknown) {
      console.error(`\n  ❌ Turn failed: ${err instanceof Error ? err.message : String(err)}\n`);
    }
  }

  rl.close();
  process.exit(0);
}

main();
