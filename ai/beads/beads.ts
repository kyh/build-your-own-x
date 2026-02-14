#!/usr/bin/env npx tsx
// beads.ts - A minimal reproduction of Beads (https://github.com/steveyegge/beads)
// A git-backed, dependency-aware issue tracker designed for AI coding agents.
// Run: npx tsx ai/beads/beads.ts <command>

import { Command } from "commander";
import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";

// ─── Types ───

/**
 * The three states an issue can be in.
 * - open: not started
 * - in-progress: actively being worked on
 * - closed: done
 */
type IssueStatus = "open" | "in-progress" | "closed";

/**
 * An issue in the tracker. This is the core data model.
 *
 * In the real Beads, issues have many more fields (assignee, threads, subtask
 * hierarchies). We keep it minimal: just enough to demonstrate the dependency
 * graph and ready queue concepts.
 */
type Issue = {
  id: string;
  title: string;
  description: string;
  status: IssueStatus;
  tags: string[];
  /** IDs of issues that this issue blocks. "A blocks B" means B can't start until A is done. */
  blocks: string[];
  createdAt: string;
  updatedAt: string;
};

// ─── Configuration ───

/** Issues are stored in a `.beads` directory in the current working directory, making them git-trackable. */
const BEADS_DIR = join(process.cwd(), ".beads");
const ISSUES_FILE = join(BEADS_DIR, "issues.jsonl");

// ─── ID Generation ───

/**
 * Generate a short, deterministic, hash-based ID for an issue.
 *
 * Real Beads uses hash-based IDs (like `bd-a3f8`) to avoid merge conflicts.
 * When two agents create issues independently on different branches, their IDs
 * won't collide because each ID is derived from the issue's content + timestamp.
 * This is a key insight: sequential IDs (1, 2, 3) would conflict constantly in
 * multi-agent workflows.
 */
function generateId(title: string, timestamp: string): string {
  const hash = createHash("sha256")
    .update(title + timestamp + Math.random().toString())
    .digest("hex")
    .slice(0, 4);
  return `bd-${hash}`;
}

// ─── Storage ───

/**
 * Load all issues from the JSONL file.
 *
 * JSONL (JSON Lines) stores one JSON object per line. This format is ideal for
 * git because adding/modifying an issue only changes one line, producing clean
 * diffs. It's also append-friendly, though we do full rewrites for simplicity.
 */
function loadIssues(): Issue[] {
  if (!existsSync(ISSUES_FILE)) return [];
  const lines = readFileSync(ISSUES_FILE, "utf-8").split("\n");
  const issues: Issue[] = [];
  for (const line of lines) {
    if (!line.trim()) continue;
    try {
      issues.push(JSON.parse(line));
    } catch {
      // skip malformed lines
    }
  }
  return issues;
}

/**
 * Write all issues back to the JSONL file.
 *
 * We do a full rewrite rather than append-only because updates and deletes
 * modify existing entries. The real Beads uses Dolt (a version-controlled SQL
 * database) for more sophisticated storage, but JSONL + git gives us the same
 * version control benefits with zero dependencies.
 */
function saveIssues(issues: Issue[]): void {
  mkdirSync(BEADS_DIR, { recursive: true });
  const content = issues.map((i) => JSON.stringify(i)).join("\n") + "\n";
  writeFileSync(ISSUES_FILE, content);
}

/**
 * Find an issue by ID, or by a prefix of the ID.
 *
 * Prefix matching is a UX convenience — you can type `bd-a3` instead of
 * `bd-a3f8` as long as the prefix is unambiguous.
 */
function findIssue(issues: Issue[], idOrPrefix: string): Issue | undefined {
  return (
    issues.find((i) => i.id === idOrPrefix) ??
    issues.find((i) => i.id.startsWith(idOrPrefix))
  );
}

// ─── Dependency Graph ───

/**
 * Compute the set of all issues that transitively block a given issue.
 *
 * This is the core graph algorithm. If A blocks B and B blocks C, then A
 * transitively blocks C. An issue is only "ready" if it has zero transitive
 * blockers that are still open.
 *
 * We use BFS to walk the reverse dependency graph. The real Beads does this
 * same computation in Go — the key insight is that this should be computed
 * deterministically by code, not by an LLM. Asking an LLM to analyze a
 * dependency graph burns tokens and is error-prone.
 */
function getTransitiveBlockers(
  issues: Issue[],
  issueId: string,
): Set<string> {
  // Build reverse map: for each issue, which issues block it?
  const blockedBy = new Map<string, string[]>();
  for (const issue of issues) {
    for (const blockedId of issue.blocks) {
      const existing = blockedBy.get(blockedId) ?? [];
      existing.push(issue.id);
      blockedBy.set(blockedId, existing);
    }
  }

  // BFS from the target issue, walking up the "blocked by" edges
  const visited = new Set<string>();
  const queue = blockedBy.get(issueId) ?? [];
  for (const id of queue) {
    if (visited.has(id)) continue;
    visited.add(id);
    for (const parentId of blockedBy.get(id) ?? []) {
      queue.push(parentId);
    }
  }

  return visited;
}

/**
 * Get all open issues that are ready to work on — i.e., not blocked by any
 * open issue.
 *
 * This is the most important function for AI agents. Instead of asking the LLM
 * "what should I work on next?", the agent calls `bd ready` and gets a
 * deterministic, correctly-ordered list. No token burn, no hallucinated
 * priorities, no missed dependencies.
 *
 * Issues are sorted by creation date (oldest first) for deterministic ordering.
 */
function getReadyIssues(issues: Issue[]): Issue[] {
  const openIssues = issues.filter((i) => i.status !== "closed");

  return openIssues.filter((issue) => {
    const blockers = getTransitiveBlockers(issues, issue.id);
    // An issue is ready if none of its blockers are still open
    return [...blockers].every((blockerId) => {
      const blocker = issues.find((i) => i.id === blockerId);
      return !blocker || blocker.status === "closed";
    });
  }).sort((a, b) => a.createdAt.localeCompare(b.createdAt));
}

// ─── Display ───

/** Format an issue as a single line for list views. */
function formatIssueLine(issue: Issue): string {
  const status =
    issue.status === "closed"
      ? "[x]"
      : issue.status === "in-progress"
        ? "[~]"
        : "[ ]";
  const tags = issue.tags.length > 0 ? ` (${issue.tags.join(", ")})` : "";
  return `  ${status} ${issue.id}  ${issue.title}${tags}`;
}

/** Format an issue with full details for the show command. */
function formatIssueDetail(issue: Issue, issues: Issue[]): string {
  const lines = [
    `  ID:          ${issue.id}`,
    `  Title:       ${issue.title}`,
    `  Status:      ${issue.status}`,
    `  Created:     ${issue.createdAt}`,
    `  Updated:     ${issue.updatedAt}`,
  ];

  if (issue.description) {
    lines.push(`  Description: ${issue.description}`);
  }
  if (issue.tags.length > 0) {
    lines.push(`  Tags:        ${issue.tags.join(", ")}`);
  }
  if (issue.blocks.length > 0) {
    const blockNames = issue.blocks.map((id) => {
      const blocked = findIssue(issues, id);
      return blocked ? `${id} (${blocked.title})` : id;
    });
    lines.push(`  Blocks:      ${blockNames.join(", ")}`);
  }

  // Show what blocks this issue
  const blockedBy = issues.filter((i) => i.blocks.includes(issue.id));
  if (blockedBy.length > 0) {
    const blockerNames = blockedBy.map((i) => `${i.id} (${i.title})`);
    lines.push(`  Blocked by:  ${blockerNames.join(", ")}`);
  }

  // Show transitive blockers
  const transitive = getTransitiveBlockers(issues, issue.id);
  const openTransitive = [...transitive].filter((id) => {
    const i = issues.find((x) => x.id === id);
    return i && i.status !== "closed";
  });
  if (openTransitive.length > 0) {
    lines.push(`  Transitively blocked by: ${openTransitive.join(", ")}`);
  }

  return lines.join("\n");
}

// ─── CLI ───

const program = new Command();

program
  .name("bd")
  .description("A minimal, git-backed issue tracker for AI coding agents")
  .version("0.0.1");

/**
 * Create a new issue.
 *
 * This is how agents (or humans) add work items. The hash-based ID means
 * two agents can create issues independently without ID conflicts.
 */
program
  .command("create")
  .description("Create a new issue")
  .argument("<title>", "Issue title")
  .option("-d, --description <text>", "Issue description", "")
  .option("-t, --tag <tags...>", "Tags for the issue")
  .action((title: string, opts: { description: string; tag?: string[] }) => {
    const issues = loadIssues();
    const now = new Date().toISOString();
    const issue: Issue = {
      id: generateId(title, now),
      title,
      description: opts.description,
      status: "open",
      tags: opts.tag ?? [],
      blocks: [],
      createdAt: now,
      updatedAt: now,
    };
    issues.push(issue);
    saveIssues(issues);
    console.log(`Created ${issue.id}: ${issue.title}`);
  });

/** List issues, optionally filtered by status. */
program
  .command("list")
  .description("List issues")
  .option("-a, --all", "Include closed issues")
  .option("-s, --status <status>", "Filter by status (open, in-progress, closed)")
  .action((opts: { all?: boolean; status?: string }) => {
    const issues = loadIssues();
    let filtered = issues;
    if (opts.status) {
      filtered = issues.filter((i) => i.status === opts.status);
    } else if (!opts.all) {
      filtered = issues.filter((i) => i.status !== "closed");
    }
    if (filtered.length === 0) {
      console.log("  No issues found.");
      return;
    }
    for (const issue of filtered) {
      console.log(formatIssueLine(issue));
    }
  });

/** Show detailed information about a single issue, including dependency info. */
program
  .command("show")
  .description("Show issue details")
  .argument("<id>", "Issue ID or prefix")
  .action((id: string) => {
    const issues = loadIssues();
    const issue = findIssue(issues, id);
    if (!issue) {
      console.error(`  Issue not found: ${id}`);
      process.exitCode = 1;
      return;
    }
    console.log(formatIssueDetail(issue, issues));
  });

/**
 * Update an issue's fields.
 *
 * Agents use this to change status (e.g., marking an issue as in-progress
 * when they start working on it) or to refine titles/descriptions as they
 * learn more about the task.
 */
program
  .command("update")
  .description("Update an issue")
  .argument("<id>", "Issue ID or prefix")
  .option("-t, --title <title>", "New title")
  .option("-d, --description <text>", "New description")
  .option("-s, --status <status>", "New status (open, in-progress, closed)")
  .option("--tag <tags...>", "Replace tags")
  .action(
    (
      id: string,
      opts: {
        title?: string;
        description?: string;
        status?: string;
        tag?: string[];
      },
    ) => {
      const issues = loadIssues();
      const issue = findIssue(issues, id);
      if (!issue) {
        console.error(`  Issue not found: ${id}`);
        process.exitCode = 1;
        return;
      }
      if (opts.title) issue.title = opts.title;
      if (opts.description) issue.description = opts.description;
      if (opts.status) {
        if (!["open", "in-progress", "closed"].includes(opts.status)) {
          console.error(`  Invalid status: ${opts.status}`);
          process.exitCode = 1;
          return;
        }
        issue.status = opts.status as IssueStatus;
      }
      if (opts.tag) issue.tags = opts.tag;
      issue.updatedAt = new Date().toISOString();
      saveIssues(issues);
      console.log(`Updated ${issue.id}: ${issue.title}`);
    },
  );

/** Close an issue. Shorthand for `update <id> --status closed`. */
program
  .command("close")
  .description("Close an issue")
  .argument("<id>", "Issue ID or prefix")
  .action((id: string) => {
    const issues = loadIssues();
    const issue = findIssue(issues, id);
    if (!issue) {
      console.error(`  Issue not found: ${id}`);
      process.exitCode = 1;
      return;
    }
    issue.status = "closed";
    issue.updatedAt = new Date().toISOString();
    saveIssues(issues);
    console.log(`Closed ${issue.id}: ${issue.title}`);
  });

/**
 * Add a "blocks" dependency between two issues.
 *
 * `bd link A B` means "A blocks B" — B cannot be worked on until A is closed.
 * This is how you express task ordering. The dependency graph is what makes
 * `bd ready` work: it can deterministically compute which issues have no
 * open blockers.
 */
program
  .command("link")
  .description("Add a dependency: <from> blocks <to>")
  .argument("<from>", "Issue ID that blocks")
  .argument("<to>", "Issue ID that is blocked")
  .action((fromId: string, toId: string) => {
    const issues = loadIssues();
    const from = findIssue(issues, fromId);
    const to = findIssue(issues, toId);
    if (!from) {
      console.error(`  Issue not found: ${fromId}`);
      process.exitCode = 1;
      return;
    }
    if (!to) {
      console.error(`  Issue not found: ${toId}`);
      process.exitCode = 1;
      return;
    }
    if (from.blocks.includes(to.id)) {
      console.log(`  ${from.id} already blocks ${to.id}`);
      return;
    }
    from.blocks.push(to.id);
    from.updatedAt = new Date().toISOString();
    saveIssues(issues);
    console.log(`Linked: ${from.id} (${from.title}) blocks ${to.id} (${to.title})`);
  });

/** Remove a "blocks" dependency between two issues. */
program
  .command("unlink")
  .description("Remove a dependency: <from> no longer blocks <to>")
  .argument("<from>", "Issue ID that blocks")
  .argument("<to>", "Issue ID that is blocked")
  .action((fromId: string, toId: string) => {
    const issues = loadIssues();
    const from = findIssue(issues, fromId);
    const to = findIssue(issues, toId);
    if (!from) {
      console.error(`  Issue not found: ${fromId}`);
      process.exitCode = 1;
      return;
    }
    if (!to) {
      console.error(`  Issue not found: ${toId}`);
      process.exitCode = 1;
      return;
    }
    const idx = from.blocks.indexOf(to.id);
    if (idx === -1) {
      console.log(`  ${from.id} does not block ${to.id}`);
      return;
    }
    from.blocks.splice(idx, 1);
    from.updatedAt = new Date().toISOString();
    saveIssues(issues);
    console.log(`Unlinked: ${from.id} no longer blocks ${to.id}`);
  });

/**
 * Show all issues that are ready to work on.
 *
 * This is the most important command for AI agents. It answers "what should I
 * do next?" by computing the dependency graph and returning only issues with
 * no open blockers. The agent doesn't need to understand the graph — it just
 * picks the first item from this list.
 *
 * Output is JSON when piped (for programmatic agent use) and human-readable
 * when interactive.
 */
program
  .command("ready")
  .description("Show issues ready for work (no open blockers)")
  .option("-j, --json", "Output as JSON (for agent consumption)")
  .action((opts: { json?: boolean }) => {
    const issues = loadIssues();
    const ready = getReadyIssues(issues);
    if (opts.json) {
      console.log(JSON.stringify(ready, null, 2));
      return;
    }
    if (ready.length === 0) {
      console.log("  No issues ready. All issues are either blocked or closed.");
      return;
    }
    console.log(`  ${ready.length} issue(s) ready:`);
    for (const issue of ready) {
      console.log(formatIssueLine(issue));
    }
  });

program.parse();
