#!/usr/bin/env npx tsx
// mini-claude-code.ts - A minimal Claude Code clone in TypeScript
// Run: npx tsx ai/claude-code/claude-code.ts

import Anthropic from "@anthropic-ai/sdk";
import { execSync } from "node:child_process";
import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import { dirname } from "node:path";
import { mkdirSync } from "node:fs";
import { createInterface } from "node:readline/promises";

// ─── Types ───

/** Plain-object version of SDK content blocks, safe for JSON serialization. */
type SerializedBlock =
  | { type: "text"; text: string }
  | { type: "tool_use"; id: string; name: string; input: unknown };

// ─── Tools ───

/**
 * Tool definitions sent to Claude via the API.
 * Each tool describes a capability the agent can invoke:
 * - read_file:    read a file's contents from disk
 * - write_file:   create or overwrite a file
 * - list_files:   list directory contents
 * - run_command:  execute an arbitrary shell command
 */
const TOOLS = [
  {
    name: "read_file",
    description: "Read the contents of a file",
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
    description: "Write content to a file (creates or overwrites)",
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
    name: "list_files",
    description: "List files in a directory",
    input_schema: {
      type: "object" as const,
      properties: {
        path: {
          type: "string" as const,
          description: "Directory path (default: current directory)",
        },
      },
    },
  },
  {
    name: "run_command",
    description: "Run a shell command",
    input_schema: {
      type: "object" as const,
      properties: {
        command: {
          type: "string" as const,
          description: "The command to run",
        },
      },
      required: ["command"],
    },
  },
] satisfies Anthropic.Tool[];

// ─── Type Guards ───

/** Narrow unknown values to a plain object. */
function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
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

/**
 * Shell patterns that trigger a confirmation prompt before execution.
 * These are substrings — if any appear in a command, the user must approve it.
 */
const DANGEROUS_PATTERNS = ["rm ", "sudo ", "chmod ", "mv ", "cp ", "> ", ">>"];

/**
 * Prompt the user for confirmation before executing a dangerous action.
 * Returns true if the action is safe or the user approves.
 */
async function checkPermission(
  toolName: string,
  toolInput: Record<string, unknown>,
  rl: ReturnType<typeof createInterface>,
): Promise<boolean> {
  if (toolName === "run_command") {
    const cmd = typeof toolInput.command === "string" ? toolInput.command : "";
    if (DANGEROUS_PATTERNS.some((p) => cmd.includes(p))) {
      const answer = await rl.question(
        `\n  ⚠️  Potentially dangerous command: ${cmd}\n  Allow? (y/n): `,
      );
      return answer.trim().toLowerCase() === "y";
    }
  } else if (toolName === "write_file") {
    const path = typeof toolInput.path === "string" ? toolInput.path : "";
    const answer = await rl.question(
      `\n  📝 Will write to: ${path}\n  Allow? (y/n): `,
    );
    return answer.trim().toLowerCase() === "y";
  }
  return true;
}

// ─── Tool Execution ───

/** Execute a tool by name, dispatching to the appropriate handler. */
function executeTool(name: string, toolInput: unknown): string {
  if (!isRecord(toolInput)) return "Error: invalid tool input";

  switch (name) {
    case "read_file": {
      const path = getString(toolInput, "path");
      try {
        return `Contents of ${path}:\n${readFileSync(path, "utf-8")}`;
      } catch (err: unknown) {
        return `Error reading file: ${err instanceof Error ? err.message : String(err)}`;
      }
    }

    case "write_file": {
      const path = getString(toolInput, "path");
      const content = getString(toolInput, "content");
      try {
        mkdirSync(dirname(path), { recursive: true });
        writeFileSync(path, content);
        return `✅ Successfully wrote to ${path}`;
      } catch (err: unknown) {
        return `Error writing file: ${err instanceof Error ? err.message : String(err)}`;
      }
    }

    case "list_files": {
      const path =
        typeof toolInput.path === "string" ? toolInput.path : ".";
      try {
        const files = readdirSync(path);
        return (
          `Files in ${path}:\n` +
          files
            .sort()
            .map((f) => `  ${f}`)
            .join("\n")
        );
      } catch (err: unknown) {
        return `Error listing files: ${err instanceof Error ? err.message : String(err)}`;
      }
    }

    case "run_command": {
      const cmd = getString(toolInput, "command");
      try {
        const output = execSync(cmd, {
          encoding: "utf-8",
          timeout: 30_000,
          stdio: ["pipe", "pipe", "pipe"],
        });
        return output ? `$ ${cmd}\n${output}` : `$ ${cmd}\n(no output)`;
      } catch (err: unknown) {
        if (isRecord(err) && typeof err.status === "number") {
          const stdout = typeof err.stdout === "string" ? err.stdout : "";
          const stderr = typeof err.stderr === "string" ? err.stderr : "";
          return stdout + stderr || `Command exited with code ${err.status}`;
        }
        return `Error: ${err instanceof Error ? err.message : String(err)}`;
      }
    }

    default:
      return `Unknown tool: ${name}`;
  }
}

// ─── Content Serialization ───

/** Convert SDK response content blocks to plain JSON-serializable objects. */
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

/**
 * The core agentic loop. Sends the conversation to Claude, processes any tool
 * calls it makes, feeds results back, and repeats until Claude produces a final
 * text response (stop_reason === "end_turn") or hits the iteration cap.
 *
 * Flow:
 *   1. Append the user's message to the conversation history.
 *   2. Call the Claude API with the full history + tool definitions.
 *   3. If Claude responds with text only → print it and return.
 *   4. If Claude responds with tool_use blocks:
 *      a. Check permissions (prompt user for dangerous ops).
 *      b. Execute each tool and collect results.
 *      c. Append tool results as a "user" message and loop back to step 2.
 */
async function agentLoop(
  userMessage: string,
  conversationHistory: Anthropic.MessageParam[],
  rl: ReturnType<typeof createInterface>,
): Promise<Anthropic.MessageParam[]> {
  conversationHistory.push({ role: "user", content: userMessage });

  // Cap iterations to prevent runaway tool-use loops
  for (let i = 0; i < 20; i++) {
    // Call Claude with the full conversation + tool definitions
    const response = await client.messages.create({
      model: "claude-sonnet-4-5-20250929",
      max_tokens: 4096,
      system: `You are a helpful coding assistant. Working directory: ${process.cwd()}`,
      tools: TOOLS,
      messages: conversationHistory,
    });

    // Serialize and append the assistant's response
    const content = serializeContent(response.content);
    conversationHistory.push({ role: "assistant", content });

    // If Claude is done (no tool calls), print the text and exit the loop
    if (response.stop_reason === "end_turn") {
      for (const block of response.content) {
        if (block.type === "text") {
          console.log(`\n🤖 ${block.text}`);
        }
      }
      break;
    }

    // Process tool calls: check permissions, execute, collect results
    const toolResults: Anthropic.ToolResultBlockParam[] = [];
    for (const block of response.content) {
      if (block.type !== "tool_use") continue;

      console.log(
        `\n🔧 ${block.name}: ${JSON.stringify(block.input).slice(0, 100)}`,
      );

      if (
        !isRecord(block.input) ||
        !(await checkPermission(block.name, block.input, rl))
      ) {
        const result = "Permission denied by user";
        console.log(`   🚫 ${result}`);
        toolResults.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: result,
        });
        continue;
      }

      const result = executeTool(block.name, block.input);
      const display =
        result.length > 200 ? result.slice(0, 200) + "..." : result;
      console.log(`   → ${display}`);
      toolResults.push({
        type: "tool_result",
        tool_use_id: block.id,
        content: result,
      });
    }

    // Feed tool results back as a "user" message for the next iteration
    conversationHistory.push({ role: "user", content: toolResults });
  }

  return conversationHistory;
}

// ─── Client & REPL ───

const client = new Anthropic();

async function main(): Promise<void> {
  const rl = createInterface({ input: process.stdin, output: process.stdout });

  console.log("Mini Claude Code");
  console.log("  Type your requests, or 'quit' to exit.\n");

  let conversationHistory: Anthropic.MessageParam[] = [];

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
    if (["quit", "exit", "q"].includes(userInput.toLowerCase())) {
      console.log("Goodbye!");
      break;
    }

    conversationHistory = await agentLoop(userInput, conversationHistory, rl);
  }

  rl.close();
  process.exit(0);
}

main();
