# Stripe Minions: Autonomous Coding Agents — Architectural Insights & Patterns

## Section 1: Quick Reference — Key Insights & Learnings

### Architecture Principles

1. **LLM as Component, Not Orchestrator**
   The LLM should be one node in a deterministic state machine, not the decision-maker controlling flow. Stripe uses blueprints where the orchestrator is code; the LLM is invoked only when needed.

2. **Hybrid Orchestration Beats Pure Agentic Loops**
   Deterministic steps (context fetching, validation, formatting) + agentic steps (creative reasoning) = reliability. Pure loops ("let the agent loop until it's done") fail ~10% of the time due to hallucination or infinite loops.

3. **Deterministic vs. Agentic Step Split**
   Use deterministic code for anything reliably automatable (linting, imports, type checking, file navigation). Use LLM only for novel problems (refactoring logic, writing new algorithms, debugging).

4. **Isolation Enables Trust**
   Devboxes (isolated EC2 instances) let agents run `git push` without fear. No devbox = agents stay cautious; devbox = agents go fast.

5. **Pre-Hydrate Context Before LLM Starts**
   Deterministically fetch relevant docs, code snippets, ticket details BEFORE sending the prompt. Saves tokens and improves reasoning quality.

6. **"What's Good for Humans is Good for Agents"**
   If fast feedback helps engineers (pre-push linting), it helps agents. If isolation helps teams (separate VMs), it helps agents. If scoped tools reduce noise (curated MCP), it helps agents.

### Design Patterns

1. **Blueprint Pattern: State Machine with Typed Nodes**
   Define workflow as a directed acyclic graph: input → [deterministic] → [agentic] → [deterministic] → output. Each node is typed (rectangles = code, clouds = LLM). Stripe's most reliable automation pattern.

2. **Shift-Left Feedback: Catch Early, Pay Less**
   Kill bugs at the cheapest layer: lint daemon (free) → pre-push hook (seconds) → deterministic node (seconds) → CI (minutes) → human review (hours). Each rightward step costs exponentially more.

3. **Smaller Box Principle: Curated Tools Over All Tools**
   500 tools × 200 tokens/tool = half your context window before you write code. Give agents a curated subset (20–50 relevant tools) instead. Better decision-making, lower token cost.

4. **Bounded Iteration: Structural Caps, Not Prompts**
   Don't prompt "keep trying until you pass tests." Set a hard cap: 2 CI rounds, then stop, hand to human. Prevents token waste and forces focus on reliable automation.

5. **Scoped Rules Over Global Rules**
   Store rules in `.claude/rules.md` subdirectories (e.g., `backend/.claude/rules.md`, `frontend/.claude/rules.md`). Agents inherit only rules relevant to their working directory—less noise, more precision.

### Operational Insights

1. **2 CI Rounds Maximum**
   Round 1 catches 80% of issues. Round 2 catches most of the rest. Round 3+ hits diminishing returns—LLM is stuck in a loop, burning tokens. Set a hard cap and escalate to human.

2. **Network Effect of Centralized Tooling**
   One MCP Toolshed = all 200+ agents (minions, code review, incident response) can use its 500 tools. Every new tool added benefits the entire fleet. Build once, leverage everywhere.

3. **Cattle, Not Pets: Devbox Lifecycle**
   Treat execution environments as disposable: pool → allocate on demand → use for 5–10 min → destroy. Reproducible, scalable, cheap. Don't name them or keep state in them.

4. **Dual-Layer Security: Tool-Level + Environment-Level**
   Level 1: restrict what tools can do (`get_ticket` = read-only, `delete_ticket` = blocked). Level 2: restrict what environment can access (QA only, no prod data, no external network). Both must pass.

### Applicable to Your Claude Code Workflow

1. **Git Worktrees as Personal Devboxes**
   `claude --worktree approach-1` gives you an isolated copy. Run 3 agents in parallel on different approaches, then `git merge` the winner. Same isolation principle, zero cost.

2. **Rule Files as Scoped Context**
   Write `.claude/rules.md` (global) and `backend/.claude/rules.md` (scoped). Claude Code respects these. Replaces 10,000 tokens of repetitive context with precise, filesystem-scoped rules.

3. **MCP Servers as Personal Toolshed**
   You already have Supabase, Playwright, Context7 MCP servers. This IS your Toolshed. Keep adding relevant tools, curate which ones you enable for each task. Scales your agent's capabilities.

4. **Headless Mode + Worktrees for Parallel Execution**
   `claude -p "task" --dangerously-skip-permissions` in multiple worktrees = unattended agents. Orchestrate with a Python script. You now have a personal "Minions" setup.

---

## Section 2: Source & Overview

### Articles & Context

This knowledge base captures insights from two Stripe engineering blog posts on their autonomous coding agent system called "Minions":

- **Part 1:** Describes the core architecture, orchestration patterns, and how minions work end-to-end
- **Part 2:** Deep dive into feedback loops, context management, MCP tooling, and scaling patterns

These articles document how Stripe deploys AI agents to write code, review PRs, fix bugs, and run migrations—autonomously, at scale, with high reliability.

### What Are Minions?

Minions are fully autonomous AI agents that:
- **Receive a task** (e.g., "Fix the bug in payment-processor: …")
- **Run fully unattended** (no human in the loop; no confirmation prompts)
- **Execute in isolation** (dedicated EC2 devbox, not shared environment)
- **Push code to a branch** (`git push` with full permissions)
- **Create a PR** (for human review before merge)

They're not just Claude in a terminal with `--dangerously-skip-permissions`. They're orchestrated via deterministic blueprints, sandboxed in ephemeral VMs, guarded by shift-left feedback loops, and backed by a network of 500 MCP tools.

### Key Metrics

- **Scale:** 1000+ PRs per week generated by minions
- **Success rate:** ~95% (automated CI passes first time)
- **Tooling:** ~500 MCP tools in central Toolshed, ~50 per agent curated subset
- **Testing:** 3 million+ tests run in CI, selectively triggered per PR
- **Payment volume protected:** $1 trillion+ annually (context for why reliability matters)
- **Feedback loop layers:** 5 levels of automated checking before human review
- **Iteration cap:** Hard limit of 2 CI rounds per task (no infinite loops)

## Section 3: Agent Harness — Supervised vs. Unattended Operation

### What "Fully Unattended" Means

A fully unattended agent:
- Receives a task description (text or structured)
- Executes autonomously from start to finish
- Makes decisions without human approval at each step
- Pushes code changes to a git branch
- Creates a PR with full context
- Never prompts the human mid-execution
- Never waits for feedback until the PR is done

This is fundamentally different from Claude Code (supervised), where you're in the loop—reading output, giving feedback, redirecting the agent.

### The Minion Stack: Goose Fork

Stripe doesn't use off-the-shelf Claude Code. They fork and heavily customize the Goose agent framework:
- **Goose (open-source):** General-purpose agent framework
- **Stripe's fork:** Custom blueprint orchestration, MCP integration, security layers, devbox integration

Key additions:
- Blueprint state machine (deterministic + agentic nodes)
- MCP Toolshed integration (500 tools, curation layer)
- Devbox lifecycle management (EC2 spin-up, teardown)
- Pre-hydration pipeline (fetch context before agent starts)
- Bounded retry logic (2 CI rounds max, hard cap)

### Supervised (Claude Code) vs. Unattended (Minion): Comparison

| Aspect | Claude Code (You) | Minion (Stripe) |
|--------|-------------------|-----------------|
| **Initiation** | You type a prompt or use UI | Task pushed to fleet; agent picked automatically |
| **Execution** | Synchronous; you watch live | Asynchronous; runs in background |
| **Feedback Loop** | You read output, say "keep going" or "fix this" | Pre-defined blueprint; no human feedback until PR |
| **Environment** | Your local machine or a cloud VM you control | Ephemeral EC2 devbox, destroyed after task |
| **Tool Access** | MCP servers you configure | 500 tools in Toolshed; curated ~50-100 per task |
| **Context** | You provide it (documents, screenshots, code) | Pre-hydrated deterministically from ticket, git, MCP |
| **Decision Making** | LLM decides flow (no state machine) | State machine decides flow; LLM invoked at specific nodes |
| **Retry Logic** | You decide ("try again" vs. "give up") | Hard cap: 2 CI rounds, then escalate |
| **Output** | Repo state + conversation history | PR + all commit history + CI results |
| **Trust Model** | You approve before push | Devbox isolation + dual security + mandatory PR review |

### Typical Minion Workflow (Full Lifecycle)

```
1. TASK ASSIGNMENT
   Ticket created in Jira: "Fix race condition in payment processing"
   Tags: #backend #high-priority
   Minion scheduler picks an idle minion

2. PRE-HYDRATION (deterministic, no LLM)
   - Fetch Jira ticket details + linked PRs
   - Pull recent git history for affected files
   - Run sourcegraph search for similar patterns
   - Inject into context (all before LLM starts)

3. DEVBOX ALLOCATION
   - Provision fresh EC2 instance (10 sec spin-up)
   - Clone repo, checkout develop branch
   - Install deps, run quick smoke tests

4. BLUEPRINT EXECUTION
   - Node 1 (deterministic): Parse task, identify affected files
   - Node 2 (agentic): Write fix, explain reasoning
   - Node 3 (deterministic): Run linters locally
   - Node 4 (agentic): Write tests
   - Node 5 (deterministic): Format, validate imports
   - Node 6 (deterministic): Push to branch, create PR

5. CI FEEDBACK LOOP
   - First push → CI runs (tests, linters, type checks)
   - If pass → PR created, awaiting human review ✓
   - If fail → Autofixes applied, minion notified
     - Minion writes fix (round 1 of 2 allowed)
     - Second push → CI runs again
     - If pass → PR created ✓
     - If fail → PR created anyway, tagged "needs human review"

6. DEVBOX DESTRUCTION
   - Task complete (regardless of PR status)
   - EC2 instance terminated
   - All state lost (cattle, not pets)

7. HUMAN REVIEW
   - Engineer reviews PR
   - Comments or approves
   - If approved: merge to main
```

### Why NOT Just "Claude Code + `--dangerously-skip-permissions` in EC2"?

This is the key clarification. You might think: "Claude Code headless on a remote machine = minion." It's not. Here's why:

| Problem | Just Claude Code | Minion + Safeguards |
|---------|------------------|---------------------|
| **LLM hallucination** | Agent gets stuck in loop, burns tokens, messes up repo | Blueprint caps retry at 2 rounds; escalates |
| **Context explosion** | Agent reads all 500 tools, burns 50K tokens on descriptions | Curated subset (50 tools), pre-hydrated context only |
| **Broken CI feedback** | Agent doesn't understand CI output, keeps trying same thing | Deterministic lint node validates before pushing |
| **Bad decisions** | Agent makes choices about flow, retry count, when to push | State machine enforces decisions structurally |
| **No isolation** | One bad command could affect repo, or your system | Ephemeral devbox = every mistake is sandboxed |
| **No early wins** | Agent waits for CI to tell it about formatting issues | Pre-push lint hook catches 99% before CI |
| **Token waste** | Agent re-reads context on each retry | Pre-hydration + blueprint = context once, reused |

### The Triple Safety Net

```
LAYER 1: ISOLATION (Devbox)
  ├─ Fresh EC2 instance per task
  ├─ Full permissions (can git push, npm install, etc.)
  ├─ But: isolated from prod data, internal systems, other agents
  └─ Any mistake destroyed in 30 seconds when task done

LAYER 2: BOUNDED ITERATION (Blueprint)
  ├─ Hard cap: 2 CI rounds
  ├─ If still failing after round 2: stop, create PR with "needs review" label
  └─ Prevents infinite loops, runaway token spend

LAYER 3: MANDATORY HUMAN REVIEW (PR + Gatekeeping)
  ├─ All minion PRs must be reviewed + approved before merge
  ├─ Merge permission requires explicit human approval
  └─ Devbox destroyed even if agent failed (no lingering state)
```

### Mental Models

**Supervised (Claude Code):**
Think of it like pair programming. You and Claude sit at one keyboard. You see every move, redirect constantly, make collaborative decisions.

**Unattended (Minion):**
Think of it like assigning a task to a junior developer. You write a clear ticket ("Fix the X bug, here's what we've tried, here's the context file"). They work independently. They push a PR when done. You review. If good, merge. If not, give feedback (they're not looping back automatically—they wait for next assignment).

The key difference: a junior dev doesn't loop indefinitely on the same task if they're stuck. They escalate. Minions do the same.

## Section 4: Blueprints — Hybrid State Machine Orchestration

### The Core Problem Blueprints Solve

You have two conflicting needs:
1. **Reliability:** Agents must follow a consistent, predictable process
2. **Flexibility:** Agents must make intelligent decisions on novel problems

If you write: *"Follow these steps, then use your judgment,"* the agent hallucinates, skips steps, or loops indefinitely (~10% failure rate).

If you write: *"Do exactly X, then Y, then Z,"* the agent can't adapt to unexpected situations (~5% failure rate, but brittle).

**Blueprints solve this:** Deterministic structure (X → Y → Z) with agentic nodes at specific points (creative reasoning happens in bounded box, not everywhere).

### What Is a Blueprint?

A blueprint is a **directed acyclic graph** where:
- **Nodes** are typed: deterministic (code/rectangles) or agentic (LLM/clouds)
- **Edges** are transitions (if success → next node; if error → retry or escalate)
- **State** is passed between nodes (output of Node 1 = input to Node 2)
- **Context** is pre-hydrated once, reused through all nodes
- **Termination** is explicit (success → PR; failure after 2 rounds → escalate)

### Example: "Fix a Bug" Blueprint

```
INPUT: Jira ticket with bug description + stack trace

                 ┌─────────────────────────────┐
                 │ DETERMINISTIC: Parse Task   │
                 │ - Extract affected files    │
                 │ - Fetch error logs          │
                 │ - Identify test file        │
                 └──────────────┬──────────────┘
                                │
                 ┌──────────────▼──────────────┐
                 │ AGENTIC: Write Fix          │
                 │ - Analyze bug root cause    │
                 │ - Write code fix            │
                 │ - Explain reasoning         │
                 └──────────────┬──────────────┘
                                │
                 ┌──────────────▼──────────────┐
                 │ DETERMINISTIC: Lint & Fmt  │
                 │ - Run prettier/eslint       │
                 │ - Fix type errors           │
                 │ - Validate imports          │
                 └──────────────┬──────────────┘
                                │
                 ┌──────────────▼──────────────┐
                 │ AGENTIC: Write Tests       │
                 │ - Design test cases        │
                 │ - Write test code          │
                 │ - Verify coverage          │
                 └──────────────┬──────────────┘
                                │
                 ┌──────────────▼──────────────┐
                 │ DETERMINISTIC: Final Check │
                 │ - Run all linters           │
                 │ - Type check               │
                 │ - Format                    │
                 └──────────────┬──────────────┘
                                │
                 ┌──────────────▼──────────────┐
                 │ Push & Create PR            │
                 │ → HUMAN REVIEW              │
                 └────────────────────────────┘

SUCCESS RATE:
- Pure agentic loop: 90% (hallucinations, infinite loops)
- Deterministic only: 95% (brittle, can't adapt)
- Hybrid blueprint: 98%+ (structured flexibility)
```

### Why Blueprints Are Better Than "Prompt the Agent to Follow Steps"

| Approach | Reliability | Token Cost | Why |
|----------|-------------|-----------|-----|
| Prompt: "Follow these 10 steps" | ~85% | High | Agent skips steps, loops, misinterprets |
| Code: Hard-coded sequence | ~90% | Medium | Works but can't adapt to variants |
| Blueprint: Typed state machine | ~98% | Low | Structure enforced; LLM only invoked when needed |

**Why low token cost?**
If your blueprint has 6 nodes and only 2 need LLM (write fix, write tests), then only 2 of 6 invoke Claude. The other 4 (parsing, linting, formatting, pushing) are deterministic code. This saves 60–80% of tokens vs. letting the LLM orchestrate.

### LangGraph Equivalence

Blueprints are conceptually identical to LangGraph:
- **Blueprint "node"** = LangGraph `StateGraph.add_node()`
- **Blueprint "deterministic node"** = LangGraph node with pure Python
- **Blueprint "agentic node"** = LangGraph node with LLM call
- **Blueprint "edge"** = LangGraph conditional routing

Here's a conceptual LangGraph version of the bug-fix blueprint:

```python
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-opus-4-6")

# State: what flows through the graph
state_schema = {
    "task": str,
    "affected_files": list,
    "error_logs": str,
    "fix_code": str,
    "test_code": str,
}

# Deterministic node: parse task
def parse_task(state):
    return {
        "affected_files": extract_files(state["task"]),
        "error_logs": fetch_logs(state["task"]),
    }

# Agentic node: write fix
def write_fix(state):
    prompt = f"Fix this bug:\n{state['error_logs']}\nAffected files: {state['affected_files']}"
    fix = llm.invoke([{"role": "user", "content": prompt}])
    return {"fix_code": fix.content}

# Deterministic node: lint
def lint_and_format(state):
    formatted = subprocess.run(["prettier", "--write", "..."], capture_output=True)
    return {"fix_code": formatted.stdout}

# Agentic node: write tests
def write_tests(state):
    prompt = f"Write tests for this fix:\n{state['fix_code']}"
    tests = llm.invoke([{"role": "user", "content": prompt}])
    return {"test_code": tests.content}

# Deterministic node: final check
def final_check(state):
    lint_result = subprocess.run(["eslint", "..."], capture_output=True)
    if lint_result.returncode != 0:
        return {"error": "Lint failed"}
    return {"success": True}

# Build graph
graph = StateGraph(state_schema)
graph.add_node("parse", parse_task)
graph.add_node("write_fix", write_fix)
graph.add_node("lint", lint_and_format)
graph.add_node("write_tests", write_tests)
graph.add_node("final_check", final_check)

# Connect edges
graph.add_edge("parse", "write_fix")
graph.add_edge("write_fix", "lint")
graph.add_edge("lint", "write_tests")
graph.add_edge("write_tests", "final_check")
graph.add_conditional_edges(
    "final_check",
    lambda x: "success" if x.get("success") else "error",
    {"success": END, "error": "write_fix"}  # Retry write_fix if lint fails
)

graph.set_entry_point("parse")
app = graph.compile()
```

### Token Economics: Why Only 2 of 6 Nodes Need LLM

```
BLUEPRINT NODES:           LLM COST?
1. Parse task              No (regex, file reads, subprocess)
2. Write fix               Yes (~4K tokens)
3. Lint & format           No (deterministic: prettier, eslint)
4. Write tests             Yes (~3K tokens)
5. Final check             No (deterministic: type check, linters)
6. Push & create PR        No (deterministic: git commands)

TOTAL BLUEPRINT: 7K tokens per task
PURE AGENTIC (read, reason, loop): 20K+ tokens (reads context 3x, tries 2–3 times)

SAVINGS: 60–70% token reduction
```

### Custom Blueprints Per Team

Stripe doesn't have one monolithic blueprint. Different teams author their own:
- **Payments team:** Bug fix → unit test → integration test → smoke test on staging → deploy to canary
- **Infra team:** Deploy → health check → metrics validation → rollback if needed
- **Data team:** Schema migration → backfill → validation → notify stakeholders

All use the same framework (deterministic + agentic nodes), but the DAG is team-specific.

### Key Takeaway

*The LLM is a component, not the orchestrator.* Code orchestrates flow. LLM is invoked at specific, bounded nodes for creative reasoning. This is why Stripe's agents are reliable (95%+) and efficient (60–80% fewer tokens).

## Section 5: Context Management — Three-Tier Funnel

### The Problem: Context Window Overflow

At scale, you have:
- 1000+ internal docs
- 3 million+ tests (which ones are relevant?)
- 500 MCP tools (which ones should the agent use?)
- Unlimited GitHub context (every commit, issue, PR)

Dump all of it into context = agent gets confused, wastes tokens deciding what to read, can't reason clearly.

**Solution:** Three-tier funnel. Each tier filters ruthlessly.

### The Three-Tier Funnel

```
TIER 1: RULE FILES (Scoped Context)
┌─ Global rules: .claude/rules.md
│  └─ What: project-wide conventions, security policies, team standards
│  └─ Who reads: every agent, every task
│  └─ Example: "Always use error boundaries in React components"
│
├─ Scoped rules: backend/.claude/rules.md, frontend/.claude/rules.md
│  └─ What: team-specific conventions (only if agent is in that directory)
│  └─ Who reads: agent working in that subdirectory
│  └─ Example: backend/.claude/rules.md = "Always use prepared statements"
│
└─ Cost: Free (deterministic filesystem scoping)


TIER 2: MCP TOOLSHED (Curated Tools)
┌─ Full toolshed: 500 MCP tools (Jira, Slack, Sourcegraph, Stripe APIs, etc.)
│  └─ Cost in tokens: 500 × 200 = 100K tokens just for tool descriptions
│  └─ Decision quality: agent gets confused with too many options
│
├─ Curated subset: 30–100 tools per agent type
│  └─ Example for bug-fix minion: Jira, GitHub, Sourcegraph, Stripe API (read-only)
│  └─ Example for deploy minion: CI/CD API, monitoring, alerting, GitHub
│  └─ Cost in tokens: 50 × 200 = 10K tokens
│
└─ Selection: determined by agent type + task tags (e.g., #backend #payments)


TIER 3: PRE-HYDRATION (Deterministic Context Fetching)
┌─ Before LLM even starts, fetch:
│  ├─ Jira ticket + linked PRs
│  ├─ Recent git history for affected files
│  ├─ Sourcegraph search results for similar code patterns
│  ├─ CI failures on this branch (if any)
│  ├─ Relevant test files
│  └─ Recent architecture decisions (from docs or comments)
│
├─ Inject all of this into context
│
└─ Why: agent doesn't waste time "thinking" about what context to gather;
   it's already there, prioritized, and relevant.
```

### Tier 1: Rule Files in Detail

**Global Rules: `.claude/rules.md`**
```markdown
# Project Rules

## Code Style
- Use TypeScript (no any)
- Max line length: 100
- Use single quotes

## Security
- Always validate user input
- Never log sensitive data
- Use prepared statements for SQL

## Testing
- Minimum 80% coverage
- Every API endpoint needs tests
```

**Scoped Rules: `backend/.claude/rules.md`**
```markdown
# Backend-Specific Rules

## Database
- Use Prisma ORM
- Migrations only via knex
- Always include rollback

## API Design
- REST endpoints only
- Error responses: {code, message, details}
- Use HTTP status codes correctly
```

When agent works in `backend/`, it inherits BOTH:
- `.claude/rules.md` (global)
- `backend/.claude/rules.md` (scoped)

When agent works in `frontend/`, it gets:
- `.claude/rules.md` (global)
- `frontend/.claude/rules.md` (if exists, scoped)

**Benefit:** No need to paste "follow these conventions" 100 times in prompts. Rules are deterministically scoped by filesystem. Saves thousands of tokens.

### Tier 2: MCP Toolshed Curation

**The Smaller Box Principle:**
Don't give all 500 tools to every agent. Instead:

| Agent Type | Task Type | Curated Tools |
|-----------|-----------|--------------|
| Bug-fix minion | Backend bug in payments | `jira_get_ticket`, `github_search_code`, `sourcegraph_find_similar`, `stripe_api_read` (no write) |
| Deploy minion | Deploy new service | `github_get_branch`, `ci_trigger_job`, `monitoring_get_metrics`, `slack_notify` |
| Code-review agent | Review PR | `github_pr_files`, `sourcegraph_get_context`, `linter_rules`, `test_results` |
| Migration agent | Data migration | `database_query`, `bigquery_read`, `monitoring_check_performance` (production read-only) |

**Benefit:** Agent makes better decisions with fewer options. Token cost drops from 100K to 10K.

### Tier 3: Pre-Hydration in Practice

**Example Task:** "Fix race condition in payment processing"

**Without pre-hydration:**
1. Agent receives task
2. Agent thinks "I need to understand the bug"
3. Agent asks for Jira ticket → MCP call → wait → inject
4. Agent thinks "I need recent context"
5. Agent searches GitHub → MCP call → wait → inject
6. Agent thinks "Are there similar issues?"
7. Agent searches Sourcegraph → MCP call → wait → inject
8. LLM has burned tokens just gathering context

**With pre-hydration:**
0. Before agent starts, deterministic code:
   - Regex finds "JIRA-12345" in task → fetch ticket → inject
   - Parse ticket, find "affected file: `payment-processor.js`" → fetch recent commits → inject
   - Sourcegraph search for "payment processing race condition" → inject
   - Find test failures on this issue → inject
1. Agent receives task + all relevant context already in window
2. Agent starts reasoning immediately, no "gathering" phase
3. Token cost of reasoning is the only cost

**Benefit:** Faster execution (no MCP round-trips), better reasoning (all context upfront), lower token cost.

### Tier 3: Cross-Tool Context Sync

In Claude Code (your setup):
- `.claude/rules.md` in repo
- Cursor respects same rules format
- Claude Code respects same rules format
- Minions respect same rules format (at Stripe)

**Same syntax, same semantics, same filesystem scoping.**

Why this matters: You write the rules once, every tool respects them. You don't have to configure tool-specific "prompts to follow the rules."

### Three-Tier Funnel Comparison: Stripe vs. Your Claude Code Setup

| Tier | Stripe Minion | Your Claude Code | Equivalent? |
|------|---------------|-----------------|-------------|
| **Tier 1: Rules** | `.claude/rules.md` + scoped subdirs | Same | Yes ✓ |
| **Tier 2: Tools** | 500 in Toolshed, ~50 curated per task | Supabase, Playwright, Context7 (~10 total) | Scaled version |
| **Tier 3: Pre-hydration** | Deterministic MCP fetching before agent | You manually provide context in prompts | Manual version |

**Insight:** You already have Tier 1. Tier 2 (your MCP servers) is your personal Toolshed. Tier 3 is where you manually inject docs/code snippets before asking Claude—same idea, just less automated.

---

## Section 6: Devboxes — Isolated Execution Environments

### What Are Devboxes?

Devboxes are **ephemeral AWS EC2 instances** allocated on-demand for each minion task:
- Fresh Linux VM (Ubuntu, 2–4 CPU cores, 8–16 GB RAM)
- Repo cloned, dependencies installed
- Task executed (agent runs for 5–20 minutes)
- Instance destroyed (all state lost)

Why not containers (Docker)? At Stripe's scale:
- EC2 allows pre-warming pools (faster spin-up)
- Better resource isolation (one agent per VM vs. shared kernel)
- Simpler permission model (VMs get full sudo if needed)
- Easier to restrict network access (VPC isolation)

### Pre-Warming Pool

Stripe maintains a **pool of pre-initialized EC2 instances:**
- 50–200 idle VMs at any time (configurable)
- All have repo cloned, deps installed, ready to execute
- When minion needs devbox: grab from pool (instant allocation, ~0.5 sec)
- After minion done: return to pool (reset, reuse)
- If pool runs low: auto-provision more

**Benefit:** 10-second spin-up becomes instant. Tasks start faster.

### Cattle, Not Pets

**Pet philosophy (old):**
- Name your server: "prod-db-1", "stripe-payment-worker-3"
- Keep it running forever
- Configure it by hand (SSH, make changes, commit to memory)
- Disaster if it dies

**Cattle philosophy (Stripe's devboxes):**
- No names; instances are fungible
- Allocate, use, destroy
- Every instance configured identically (from template)
- If one fails, grab another (no loss)
- Makes scale possible

**Why this matters for agents:** Agents can run destructive commands (`rm -rf /, sudo halt, git push --force`) and it doesn't matter. Devbox dies anyway in 10 minutes. Next task gets a fresh one.

### QA Isolation: What Agents Can & Cannot Access

Devbox is sandboxed to prevent accidents:

```
✓ CAN ACCESS:
├─ Repo clone (read/write)
├─ npm/pip/cargo (install packages)
├─ Local development tools (linters, tests)
├─ QA databases (read + write)
├─ QA Stripe API (test keys, sandbox)
├─ Slack (send messages to #dev-chat)
├─ GitHub (push to branches, create PRs)
└─ Internal docs (read-only MCP access)

✗ CANNOT ACCESS:
├─ Production databases
├─ Production Stripe API (live keys)
├─ Prod user data
├─ External internet (no curl to random websites)
├─ Other agents' environments
├─ AWS console / credentials
└─ Sensitive internal systems
```

Implementation: Security group + VPC rules + IAM role.

### Why Devboxes Over Worktrees or Docker at Stripe's Scale

| Approach | Pros | Cons | At Stripe Scale |
|----------|------|------|-----------------|
| **Git Worktrees** | Free, instant | Shared machine (one bad command affects all) | Doesn't scale; agents interfere |
| **Docker Containers** | Good isolation, fast | Container escape risks | Works, but less isolated than VMs |
| **Devboxes (EC2)** | True VM isolation, pre-warmed | Costs $, infrastructure | Worth it; isolation is essential |

**Personal projects:** Worktrees are fine (you trust the agent).
**Stripe scale:** Devboxes essential (1000+ PRs/week, need perfect isolation).

### Parallelism Multiplier

Without devboxes:
```
1 agent on 1 machine = 1 task processed
3 agents on 1 machine = 1 task processed (context switching, contention)
```

With devboxes:
```
1 devbox = 1 task processed
2 devboxes = 2 tasks processed (in parallel)
100 devboxes = 100 tasks processed (in parallel)
```

This is why Stripe can do 1000+ PRs per week with ~200 agents. Parallelism.

### Devbox Lifecycle

```
1. POOL MONITORING
   └─ System maintains 50–200 idle pre-warmed instances

2. TASK QUEUED
   └─ Engineer or scheduler creates ticket → task enters queue

3. DEVBOX ALLOCATED
   └─ System picks idle instance from pool
   └─ Checkpoint: reset to clean state (git clean -fd, etc.)

4. MINION RUNS
   └─ Clone repo (already done)
   └─ Execute blueprint
   └─ Push branch + create PR

5. CLEANUP
   └─ Git reset to clean state
   └─ Return instance to idle pool
   └─ Ready for next task

6. LIFECYCLE
   └─ Each instance reused ~200 times
   └─ Every 100 tasks: terminate + rebuild (refresh OS, deps)
```

### "What's Good for Humans is Good for Agents"

This is Stripe's operating philosophy applied to infrastructure.

**For humans:**
- Fast feedback (pre-push linting, quick CI)
- Isolation (branches don't interfere)
- Trust (no fear of breaking prod)
- Simplicity (no manual setup)

**For agents:**
- Same benefits via blueprints, devboxes, shift-left feedback
- Isolated execution (agent can't break other agents)
- Bounded feedback (2-round cap, clear escalation)
- Deterministic context (pre-hydrated, not guessed)

---

## Section 7: Feedback Loops & CI — The 5-Layer Error Funnel

### The Shift-Left Principle

Cost spectrum (left to right = more expensive):
```
COST:    $0         $1         $10       $100      $1000
SPEED:   <1s        <5s        1m        10m       1hr
         ─────────────────────────────────────────────►

Layer 1: Lint daemon         (free, instant)
Layer 2: Pre-push hook       (free, <5s)
Layer 3: Deterministic node  (free, <5s)
Layer 4: CI run #1           (costs $, takes time)
Layer 5: CI run #2 + human   (costs $, takes longer)
```

**Philosophy:** Kill bugs as far left as possible. Every layer you skip rightward costs exponentially more.

### The 5-Layer Funnel Explained

#### Layer 1: Background Lint Daemon (Sub-second)

**What it does:**
- Runs in background on developer's machine (or in devbox)
- Pre-computes lint rules on changed files
- Caches results
- Applies auto-fixes automatically

**Example:**
```bash
# While agent writes code...
daemon: (in background)
  detect file change → run prettier → cache result
  detect file change → run eslint → cache result
  detect import error → auto-fix → cache result

# When agent does `git commit` or `git push`...
pre-push hook:
  reads cache (which is already correct) → instant ✓
  push succeeds
```

**Why it's fast:**
- Not running linters in real time (that would be slow)
- Reading pre-computed cached results
- Auto-fixes already applied

**Cost:** Free (deterministic, no LLM)

#### Layer 2: Pre-Push Hook (<5 seconds)

**What it does:**
- Runs on `git push` (before push actually happens)
- Uses heuristics to select relevant lints
- Blocks push if failures found
- Agent sees immediate feedback

**Example:**
```bash
$ git push
  Running pre-push hook...
  Changed files: payment-processor.js, utils.js

  Running selective lints:
  └─ payment-processor.js: TypeScript strict mode...PASS
  └─ utils.js: Import order...FAIL
     └─ Auto-fixing: reordered imports...OK
  └─ All files: prettier...OK

  Push allowed ✓
```

**Cost:** Free (deterministic, no LLM), takes <5 seconds

#### Layer 3: Deterministic Lint Node (< 5 seconds, Blueprint)

**What it does:**
- Part of the blueprint state machine
- Runs linters locally (not in CI)
- Loops until lint passes
- Only then proceeds to CI

**Why separate from pre-push hook:**
- Pre-push hook is optimized (fast, heuristic-based)
- Lint node is comprehensive (all rules, no skips)
- Runs deterministically within blueprint

**Example:**
```
Blueprint node 3: "Format & Lint"
  ├─ Run prettier → auto-apply fixes
  ├─ Run eslint → auto-fix what's fixable
  ├─ Run tsc --noEmit → check types
  ├─ If passes → continue to CI
  ├─ If fails → (loop within this node)
  │   ├─ Try autofix again
  │   ├─ If still fails → agent writes manual fix
  │   ├─ Re-run lint
  │   └─ Loop until passes OR manual fix fails
  └─ Exit node when lint clean (ready for CI push)
```

**Cost:** Free (code + deterministic tools), <5 seconds typically

#### Layer 4: First CI Run (Minutes)

**What it does:**
- Real CI execution: compile, unit tests, integration tests
- Selective: Stripe runs only 1–5% of their 3M+ tests (heuristic selection)
- Auto-fixes applied (if tests have autofixes)
- If all pass → PR created ✓

**Example:**
```
$ git push (from devbox)

CI triggered:
├─ Compile step...OK
├─ Run tests (selected subset)...2 failures
│  ├─ Test failure: "fixture mismatch"
│  ├─ Autofix available: update fixture file
│  └─ Autofix applied...OK
│  ├─ Test failure: "type mismatch"
│  ├─ No autofix available
│  └─ Agent notified
└─ If failures remain without fixes → go to Layer 5
   Else → PR created ✓
```

**Cost:** $5–50 in compute/tokens, takes 5–30 minutes

#### Layer 5: Second CI Run (Minutes, Final)

**What it does:**
- Agent gets ONE more chance to fix failing tests
- Agent sees exact CI output, writes fix locally
- Deterministic lint node runs again (layer 3)
- Push happens

**Mechanics:**
```
Round 1 failed:
  CI says: "Test 'payment-processing' failed:
            expected result.total to equal 100, got 99"

Agent (round 1 of 2 allowed):
  Reads error → analyzes code → writes fix
  → Deterministic node validates
  → Push second time

CI runs again:
  ├─ If passes → PR created ✓✓ (human review)
  └─ If fails → PR created with "needs human review" label (agent is done)

No third round. Hard stop.
```

**Cost:** $5–50 in compute/tokens, takes 5–30 minutes

**Rationale for hard cap:**
```
Round 1: Catches 80% of issues (autofixes, common problems)
Round 2: Catches 15% (agent writes fix, validates locally)
Round 3+: Catches <5%, costs exponentially more

Diminishing returns curve: Beyond round 2, LLM is likely stuck
on a deep logic bug. Token waste accelerates. Better to escalate.
```

### Autofix Mechanisms

**What are autofixes?**
Deterministic fixes for common failures:
- Type changed → update mocks/fixtures
- API response shape changed → update test expectations
- Deprecated function called → apply migration automatically
- Import path changed → update imports
- Missing semicolon → add it

**How they're applied:**
1. Test runs → fails with "fixable error" flag
2. Test framework identifies fix type
3. Deterministic code applies fix (no LLM)
4. Rerun test

**Benefits:**
- No token cost (deterministic)
- Instant (no LLM latency)
- Reliable (same fix applied same way every time)

### Cost Analysis: Problem Type → Fix Method → Cost

| Problem | Fix With | Cost | Time |
|---------|----------|------|------|
| Formatting (prettier) | Lint daemon (Layer 1) | $0 | <1s |
| Import order (eslint) | Pre-push hook (Layer 2) | $0 | 2s |
| Type errors (tsc) | Deterministic node (Layer 3) | $0 | 5s |
| Fixture mismatch | CI autofix (Layer 4) | ~$5 | 10m |
| Test expectation wrong | Agent fix (Layer 4/5) | ~$20 | 15m |
| Logic bug | Agent fix (Layer 5) | ~$30 | 20m |
| Deep architectural issue | Human review (Layer 5+) | ~$100 | 1hr |

### 5-Layer Funnel Visual

```
INPUT: Agent writes code + pushes

       LAYER 1: Background Lint Daemon
       ├─ Pre-computed results cached
       ├─ Auto-fixes applied
       └─ $0 | <1s ────────┐
                            │
       LAYER 2: Pre-Push Hook
       ├─ Heuristic lint selection
       ├─ Auto-fixes applied
       └─ $0 | <5s ────────┐
                            │
       LAYER 3: Deterministic Node (Blueprint)
       ├─ Comprehensive lint
       ├─ Type check
       ├─ Manual agent fix if needed
       └─ $0 | <5s ────────┐
                            │
       LAYER 4: First CI Run
       ├─ Compile + selective tests
       ├─ Auto-fixes applied
       └─ $5–50 | 10–30m ─┐
                           │
       LAYER 5: Second CI Run
       ├─ Agent fixes (round 1 of 2)
       ├─ Deterministic validation
       └─ $5–50 | 10–30m ─┐
                           │
       ESCALATE: Human Review
       ├─ Engineer reviews PR
       ├─ Feedback or approve
       └─ $100+ | 1hr+ ────────────→ Human Decision

Output: PR ready for human review (or needs human attention)
```

---

## Section 8: MCP Toolshed — Centralized Tool Ecosystem

### What Is MCP and Why It Matters

**MCP** = Model Context Protocol. Think of it as a **universal plug standard for AI tools**:
- Just like USB lets any device connect to any computer
- MCP lets any AI agent connect to any tool server
- Common language for function calling across tools

**At Stripe:**
- Before MCP: each agent had custom integrations (Jira, Slack, GitHub, …)
- After MCP: all agents speak MCP; all tools are MCP servers

### Toolshed: 500 Tools in One Place

Stripe built a central MCP server called **Toolshed** hosting ~500 tools:

| Category | Tools | Examples |
|----------|-------|----------|
| **Issue Tracking** | ~30 | get_jira_ticket, create_jira_issue, update_status |
| **Code** | ~50 | github_search_code, github_create_pr, sourcegraph_find_references |
| **Build & CI** | ~40 | trigger_build, get_ci_logs, list_test_results |
| **Monitoring & Alerts** | ~50 | get_metrics, check_latency, get_recent_errors |
| **Payment APIs** | ~100 | stripe_get_charge, stripe_get_customer, list_transactions |
| **Internal Services** | ~150+ | internal-api calls to PaymentProcessor, Ledger, Billing, … |
| **Communications** | ~20 | send_slack_message, send_email, notify_team |
| **Data & Analytics** | ~30 | query_bigquery, get_data_pipeline_status |

### The "Smaller Box" Principle

**Problem:** 500 tools × 200 tokens per tool description = 100,000 tokens

```
Context window: 200K tokens
Tool descriptions: 100K tokens
Code to read: 50K tokens
Agent thinking: 20K tokens (if lucky)
Space left: 30K tokens
```

This is wasteful. Agent hasn't written a line of code yet and 50% of context is tool metadata.

**Solution:** Give agent a curated subset (~50 tools) not the full 500.

| Agent Type | Curated Tools | Why |
|-----------|------|-----|
| Bug-fix minion | Jira, GitHub, Sourcegraph, Stripe API (read), Slack | Read task, search code, get context, report back |
| Deploy minion | GitHub, CI, monitoring, Slack, Stripe API | Trigger builds, check health, notify |
| Code-review agent | GitHub, linters, test results, Sourcegraph | Review code, run checks, suggest improvements |
| Migration agent | Database, BigQuery, monitoring (read), Slack | Run migrations safely, monitor side effects |

**Benefit:** Agent makes better decisions (fewer options), lower token cost (50 tools vs 500).

### Dual-Layer Security

```
LAYER 1: TOOL-LEVEL SECURITY
┌─ Each tool has read/write restrictions
│
├─ ✓ get_jira_ticket(id) ────────────────► READ-ONLY
├─ ✓ github_search_code(query) ──────────► READ-ONLY
├─ ✓ ci_trigger_build(config) ──────────► WRITE (allowed)
│
├─ ✗ delete_jira_ticket(id) ────────────► BLOCKED
├─ ✗ stripe_charge_refund_all() ────────► BLOCKED
├─ ✗ database_drop_table(name) ─────────► BLOCKED
│
└─ Implementation: MCP tool definitions include permission flags

LAYER 2: ENVIRONMENT-LEVEL SECURITY
┌─ Devbox has network restrictions
│
├─ ✓ Can access QA databases
├─ ✓ Can access QA Stripe API
├─ ✓ Can access internal MCP server
├─ ✓ Can access GitHub
│
├─ ✗ Cannot access production databases
├─ ✗ Cannot access production Stripe API
├─ ✗ Cannot access arbitrary external internet
├─ ✗ Cannot see other agents' environments
│
└─ Implementation: VPC, security groups, IAM roles
```

Both layers must pass. If either is violated, action is blocked.

### Network Effect & Flywheel

```
BEFORE:
  New service "PaymentRouter" is built
  ↓
  "Agents can't interact with it" ← friction
  ↓
  Service islanded

AFTER MCP:
  New service "PaymentRouter" is built
  ↓
  Engineer writes 1 MCP tool: get_payment_router_status(id)
  ↓
  Tool added to Toolshed
  ↓
  ALL 200+ agents (Minions, code review, incident response, …)
  can NOW interact with PaymentRouter (if in their curated subset)
  ↓
  Service immediately useful to entire fleet
  ↓
  Every new tool benefits the entire fleet (flywheel)
```

### Pre-Hydration via MCP

Pre-hydration = deterministically fetch context BEFORE agent starts.

**Example:** Task = "Fix the bug described in JIRA-12345"

**Without pre-hydration (agent fetches on-demand):**
```
1. Agent receives task
2. Agent thinks "I need the ticket details"
3. Agent calls get_jira_ticket("JIRA-12345") via MCP
4. Toolshed responds (2–5 second latency)
5. Agent reads response, injects into context
... (repeat for each data fetch, wastes tokens + time)
```

**With pre-hydration (deterministic script before agent starts):**
```
0. Before agent even starts:
   ├─ Regex finds "JIRA-12345" in task → call get_jira_ticket
   ├─ Ticket mentions file "payments/checkout.rb" → call sourcegraph_get_file
   ├─ Check recent CI on payments/ → call ci_get_logs
   ├─ Find related PRs → call github_search_prs
   └─ All results injected into context

1. Agent receives task + full context + ready to reason
2. Agent never calls Jira, GitHub, Sourcegraph (already in context)
3. Tokens saved, latency eliminated
```

### Comparison: Stripe Toolshed vs. Your Claude Code Setup

| Aspect | Stripe | Your Setup | Gap |
|--------|--------|----------|-----|
| **Protocol** | MCP | MCP (same) | ✓ Same |
| **Number of tools** | ~500 | 10–30 (Supabase, Playwright, Context7, …) | Scaled version |
| **Tool sources** | Central Toolshed | Distributed MCP servers | Different architecture |
| **Curation** | Per agent type | All tools available | You don't curate |
| **Security** | Tool-level + environment | Permission prompts | Less strict |
| **Pre-hydration** | Deterministic before agent | Manual (you paste context) | Manual version |
| **Cost** | Token-efficient | No cost, but higher token use | Less optimized |

**Key insight:** Same protocol (MCP), different scale. Stripe's 500 tools = your 10 tools curated to your projects. Stripe's Toolshed = your collection of MCP servers.

---

## Section 9: Building Your Own — Personal Devbox Options

### Option 1: Git Worktrees (Free, Already Built Into Claude Code)

**What it is:**
- Isolated copy of your repo on local machine
- Separate git branch per worktree
- Built-in to Claude Code via `claude --worktree`

**Setup:**
```bash
# Project structure (automatic when you use --worktree)
YOUR PROJECT REPO
├── main branch (your code)
├── .claude/worktrees/
│   ├── approach-1/  ← Full copy, branch: worktree-approach-1
│   ├── approach-2/  ← Full copy, branch: worktree-approach-2
│   └── approach-3/  ← Full copy, branch: worktree-approach-3
```

**Usage: Run 3 agents in parallel, pick the best**
```bash
# Terminal 1
claude --worktree approach-1 \
  -p "Implement auth system using JWT tokens" \
  --dangerously-skip-permissions &

# Terminal 2
claude --worktree approach-2 \
  -p "Implement auth system using session cookies" \
  --dangerously-skip-permissions &

# Terminal 3
claude --worktree approach-3 \
  -p "Implement auth system using OAuth2" \
  --dangerously-skip-permissions &

# Wait for all to finish
wait

# Review all three approaches
git diff main..worktree-approach-1
git diff main..worktree-approach-2
git diff main..worktree-approach-3

# Merge the one you like
git merge worktree-approach-2
```

**Pros:** Free, instant, no setup
**Cons:** All on your local machine (shared CPU/RAM), not true isolation

**Best for:** Small to medium projects where you trust the agent

---

### Option 2: Docker Containers (Free, True Isolation)

**Setup: Reusable Dockerfile**
```dockerfile
FROM node:20-slim

# Install Claude Code
RUN npm install -g @anthropic-ai/claude-code

# Install common dev tools
RUN apt-get update && apt-get install -y \
    git curl python3 pip make jq

# Set working directory
WORKDIR /workspace

# Health check
HEALTHCHECK --interval=10s CMD ps aux | grep claude || exit 1
```

**Usage: Run 3 agents in parallel**
```bash
# Build once
docker build -t my-devbox .

# Run 3 containers in parallel
for i in 1 2 3; do
  docker run -d \
    --name agent-$i \
    -v $(pwd):/source:ro \
    -v /tmp/agent-$i:/workspace \
    -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
    my-devbox \
    bash -c "
      cp -r /source /workspace/project && \
      cd /workspace/project && \
      git checkout -b approach-$i && \
      claude -p 'Implement feature X using approach $i' \
        --dangerously-skip-permissions \
        --output-format json > /workspace/result-$i.json
    "
done

# Wait for all to finish
docker wait agent-1 agent-2 agent-3

# Copy results back
for i in 1 2 3; do
  docker cp agent-$i:/workspace/project ./results/approach-$i
done

# Review and merge
for i in 1 2 3; do
  echo "=== Approach $i ==="
  cat ./results/approach-$i/.git/HEAD
done
```

**Pros:** True isolation (filesystem, network, processes), bad commands can't affect host
**Cons:** Slightly more setup than worktrees, need Docker installed

**Best for:** When you want real isolation but don't want to pay for cloud

---

### Option 3: GitHub Codespaces (Cheapest Cloud)

**Cost:** 3 codespaces × 1 hour = ~$0.54. Free tier: 120 core-hours/month included.

**Setup: devcontainer.json**
```jsonc
// .devcontainer/devcontainer.json in your repo
{
  "image": "mcr.microsoft.com/devcontainers/javascript-node:20",
  "features": {
    "ghcr.io/devcontainers/features/git:latest": {}
  },
  "postCreateCommand": "npm install -g @anthropic-ai/claude-code",
  "secrets": {
    "ANTHROPIC_API_KEY": { "description": "API key for Claude" }
  },
  "remoteUser": "codespace"
}
```

**Usage: Run 3 codespaces in parallel**
```bash
# Create 3 codespaces from the same repo
for i in 1 2 3; do
  gh codespace create \
    --repo your-username/your-project \
    --branch main \
    --machine basicLinux32gb \
    --display-name "agent-$i"
done

# Wait for them to start
for i in 1 2 3; do
  gh codespace code-owner agent-$i
done

# Run Claude in each
for i in 1 2 3; do
  gh codespace ssh -c "agent-$i" -- \
    "cd /workspaces/your-project && \
     git checkout -b approach-$i && \
     claude -p 'Implement feature X, approach $i' \
       --dangerously-skip-permissions" &
done

# Wait for all to finish
wait

# Review branches
gh codespace ssh -c "agent-1" -- "git diff main..approach-1"
gh codespace ssh -c "agent-2" -- "git diff main..approach-2"
gh codespace ssh -c "agent-3" -- "git diff main..approach-3"

# Merge the best one
git merge origin/approach-2

# Clean up codespaces
for i in 1 2 3; do
  gh codespace delete -c "agent-$i"
done
```

**Pros:** True cloud VMs, GitHub free tier covers most hobby use, easy to destroy
**Cons:** Costs money (but has free tier), slightly slower startup

**Best for:** When you want cloud VMs without managing AWS yourself

---

### Option 4: Claude Code Agent SDK (Most Stripe-Like)

**What it is:** Orchestration script that spawns multiple isolated agents

**Setup: Python orchestration script**
```python
import asyncio
import subprocess
import tempfile
import shutil
import json
import os

async def run_agent(task_description, approach_name, project_dir):
    """Run one Claude agent in an isolated directory"""

    # Create isolated copy (like a mini devbox)
    work_dir = tempfile.mkdtemp(prefix=f"agent-{approach_name}-")
    project_copy = f"{work_dir}/project"
    shutil.copytree(project_dir, project_copy)

    try:
        # Run Claude Code headless in the isolated copy
        result = subprocess.run(
            [
                "claude", "-p", task_description,
                "--dangerously-skip-permissions",
                "--output-format", "json"
            ],
            cwd=project_copy,
            capture_output=True,
            text=True,
            env={**os.environ, "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY")}
        )

        return {
            "approach": approach_name,
            "work_dir": project_copy,
            "success": result.returncode == 0,
            "output": json.loads(result.stdout) if result.stdout else {},
            "stderr": result.stderr
        }

    except Exception as e:
        return {
            "approach": approach_name,
            "work_dir": project_copy,
            "success": False,
            "error": str(e)
        }

async def main():
    task = "Add user authentication with login/logout endpoints"
    project = "/path/to/your/project"

    # Run 3 agents in parallel — like Stripe's minion fleet
    print("Spawning 3 agents in parallel...")
    results = await asyncio.gather(
        run_agent(f"{task} using JWT tokens", "jwt", project),
        run_agent(f"{task} using session cookies", "sessions", project),
        run_agent(f"{task} using OAuth2 with Google", "oauth", project),
    )

    # Show results for human review
    for r in results:
        print(f"\n{'='*60}")
        print(f"Approach: {r['approach']}")
        print(f"Status: {'✓ SUCCESS' if r['success'] else '✗ FAILED'}")
        print(f"Code location: {r['work_dir']}")
        if r['success']:
            print(f"Output preview: {str(r['output'])[:200]}...")
        else:
            print(f"Error: {r.get('stderr', r.get('error', 'Unknown'))[:200]}")

        # Show git diff if successful
        if r['success']:
            diff_result = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=r['work_dir'],
                capture_output=True,
                text=True
            )
            print(f"\nGit diff (first 500 chars):")
            print(diff_result.stdout[:500])

    # Prompt user to pick best approach
    print(f"\n{'='*60}")
    print("Review the above. Which approach do you prefer?")
    choice = input("Enter: jwt, sessions, or oauth: ").strip().lower()

    # Copy best approach back to main repo
    if choice in ["jwt", "sessions", "oauth"]:
        best = next((r for r in results if r['approach'] == choice), None)
        if best:
            print(f"Copying {choice} approach to main repo...")
            subprocess.run(["cp", "-r", f"{best['work_dir']}/.", project])
            print("Done! Review and commit the changes.")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    asyncio.run(main())
```

**Usage:**
```bash
python orchestrate_agents.py
```

**Pros:** Most Stripe-like (programmatic), full control over workflow, can add blueprints/pre-hydration
**Cons:** Requires scripting, all on your local machine, limited isolation

**Best for:** When you want orchestration without cloud infrastructure

---

### Recommendation Matrix

```
                        COST          ISOLATION       SIMPLICITY
                        |             |               |
FREE ◄─────────────────┤             │               │
                        │             │               │
CHEAP ◄──────────────────┤            │               │
                        │ $$$         │               │
                        │             │               │
                        │             ▼               │
                        │        LOW ◄────────► HIGH  │
                        │                             │
                        │       EASY ◄────────► COMPLEX
                        │                             │
Option 1 (Worktrees)    ████░░░░░░  ████░░░░░░  ████░░░░░░
Option 2 (Docker)       ██████░░░░  ████████░░  ██████░░░░
Option 3 (Codespaces)   ████████░░  ████████░░  ██████░░░░
Option 4 (Agent SDK)    ████░░░░░░  ████░░░░░░  ████████░░
```

**Recommended approach:**
- **Start with:** Option 1 (Worktrees) — Free, instant, good enough for most projects
- **Scale to:** Option 2 or 3 (Docker or Codespaces) — When you need real isolation
- **Optimize with:** Option 4 (Agent SDK) — If you want full orchestration

---

## Section 10: Architectural Reference Diagram

### Full System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STRIPE MINION ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────────────┘

                              HUMAN LAYER
                         ┌──────────────────┐
                         │ Engineer writes  │
                         │ Jira ticket      │
                         └────────┬─────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │  TASK QUEUE (Scheduler)    │
                    │  - Prioritize by urgency   │
                    │  - Load balance            │
                    └─────────────┬──────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
         ▼                        ▼                        ▼
    ┌─────────┐          ┌──────────────┐          ┌─────────┐
    │ Minion  │          │ Minion       │          │ Minion  │
    │ Instance│          │ Instance     │          │ Instance│
    │ #1      │          │ #2 (busy)    │          │ #3      │
    └────┬────┘          └──────────────┘          └────┬────┘
         │                                               │
         ▼                                               ▼
    ┌──────────────────────────────────────────────────────┐
    │          PRE-HYDRATION (Deterministic)              │
    ├──────────────────────────────────────────────────────┤
    │ 1. Regex parse task → extract ticket ID/file paths  │
    │ 2. MCP: Fetch Jira ticket + linked PRs              │
    │ 3. MCP: Sourcegraph search for similar code         │
    │ 4. Git: Recent commits on affected files            │
    │ 5. Inject all into context (before LLM starts)      │
    └──────────────┬───────────────────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │  Devbox Allocation │
         │  (from pool)       │
         │  ┌──────────────┐  │
         │  │ EC2 instance │  │
         │  │ Pre-warmed   │  │
         │  │ Ready to go  │  │
         │  └──────────────┘  │
         └────────┬───────────┘
                  │
    ┌─────────────▼──────────────────────────────────────────┐
    │            BLUEPRINT STATE MACHINE                     │
    ├─────────────▼──────────────────────────────────────────┤
    │                                                        │
    │ ┌─────────────────────┐                               │
    │ │  Node 1: PARSE TASK │ (Deterministic)               │
    │ │  - Extract files    │                               │
    │ │  - Set priorities   │                               │
    │ └────────────┬────────┘                               │
    │              │                                         │
    │ ┌────────────▼────────────┐                           │
    │ │  Node 2: WRITE FIX      │ (Agentic: LLM call)      │
    │ │  - Analyze bug/feature  │ ~5K tokens               │
    │ │  - Generate code        │                           │
    │ └────────────┬────────────┘                           │
    │              │                                         │
    │ ┌────────────▼──────────────────┐                     │
    │ │  Node 3: DETERMINISTIC LINT   │ (Deterministic)    │
    │ │  - Run prettier, eslint, tsc  │                     │
    │ │  - Auto-apply fixes           │                     │
    │ └────────────┬──────────────────┘                     │
    │              │                                         │
    │ ┌────────────▼──────────────┐                         │
    │ │  Node 4: WRITE TESTS      │ (Agentic: LLM call)    │
    │ │  - Generate test cases    │ ~3K tokens             │
    │ │  - Write test code        │                         │
    │ └────────────┬──────────────┘                         │
    │              │                                         │
    │ ┌────────────▼──────────────┐                         │
    │ │  Node 5: FINAL VALIDATE   │ (Deterministic)        │
    │ │  - Run all linters        │                         │
    │ │  - Type check complete    │                         │
    │ │  - Format code            │                         │
    │ └────────────┬──────────────┘                         │
    │              │                                         │
    │ ┌────────────▼──────────────┐                         │
    │ │  Node 6: GIT PUSH         │ (Deterministic)        │
    │ │  - Commit changes         │                         │
    │ │  - Create PR              │                         │
    │ └────────────┬──────────────┘                         │
    │              │                                         │
    └──────────────┬───────────────────────────────────────┘
                   │
         ┌─────────▼────────────────────────────┐
         │   CI FEEDBACK LOOP (Shift-Left)      │
         ├─────────────────────────────────────┤
         │                                      │
         │ LAYER 1: Lint Daemon                │
         │ ├─ Pre-computed results             │
         │ ├─ $0 | <1s                         │
         │ └─ Auto-fixes applied               │
         │                                      │
         │ LAYER 2: Pre-Push Hook              │
         │ ├─ Heuristic lint selection         │
         │ ├─ $0 | <5s                         │
         │ └─ Block if failures                │
         │                                      │
         │ LAYER 3: Deterministic Node         │
         │ ├─ Comprehensive validation         │
         │ ├─ $0 | <5s                         │
         │ └─ Loop until lint passes           │
         │                                      │
         │ LAYER 4: CI Run #1                  │
         │ ├─ Compile + selective tests        │
         │ ├─ $5–50 | 10–30m                   │
         │ ├─ Auto-fixes applied               │
         │ └─ If pass → PR created ✓           │
         │           → Go to HUMAN REVIEW      │
         │ └─ If fail → Go to LAYER 5          │
         │                                      │
         │ LAYER 5: CI Run #2 (Final)          │
         │ ├─ Agent gets 1 more chance         │
         │ ├─ $5–50 | 10–30m                   │
         │ ├─ Re-run deterministic node        │
         │ ├─ If pass → PR created ✓           │
         │ └─ If fail → PR created anyway      │
         │            with "needs review" tag  │
         │                                      │
         └──────────┬───────────────────────────┘
                    │
          ┌─────────▼──────────────────┐
          │   DEVBOX DESTRUCTION       │
          │ ├─ EC2 terminated          │
          │ ├─ State reset (cattle)    │
          │ └─ Return to pool (reuse)  │
          └──────────┬─────────────────┘
                     │
         ┌───────────▼──────────────┐
         │   HUMAN REVIEW LAYER     │
         ├───────────────────────────┤
         │ ┌──────────────────────┐  │
         │ │  PR in GitHub        │  │
         │ │  - Code              │  │
         │ │  - Tests             │  │
         │ │  - CI results        │  │
         │ │  - Commit history    │  │
         │ └──────────────────────┘  │
         │          │                 │
         │  ┌──────────────┬────────┐ │
         │  ▼              ▼        ▼ │
         │ APPROVE     COMMENT    REJECT
         │  │              │        │  │
         │  └──────┬───────┴────────┘  │
         │         ▼                    │
         │   MERGE to main              │
         │   (if approved)              │
         │                              │
         └──────────────────────────────┘
```

### Context Flow Diagram

```
┌────────────────────────────────────────────────────────────┐
│                  CONTEXT MANAGEMENT                        │
│                   (3-Tier Funnel)                          │
└────────────────────────────────────────────────────────────┘

TIER 1: RULE FILES (Filesystem-Scoped)
┌─ .claude/rules.md (Global)
│  ├─ Project-wide conventions
│  ├─ Security policies
│  └─ Team standards
│
├─ backend/.claude/rules.md (Scoped)
│  ├─ Database rules
│  ├─ API design rules
│  └─ Only if agent works in backend/
│
├─ frontend/.claude/rules.md (Scoped)
│  ├─ UI/component rules
│  ├─ Style guide
│  └─ Only if agent works in frontend/
│
└─ Cost: FREE (deterministic filesystem scoping)


TIER 2: MCP TOOLSHED (Curated Tools)
┌─ Full Toolshed: ~500 tools
│  ├─ Jira, GitHub, Sourcegraph, Stripe APIs, …
│  └─ Cost: 500 tools × 200 tokens = 100K tokens (wasteful)
│
├─ Curated Subset: 30–100 tools per agent
│  ├─ Example (bug fix): Jira, GitHub, Sourcegraph
│  ├─ Example (deploy): CI/CD, monitoring, Slack
│  └─ Cost: 50 tools × 200 tokens = 10K tokens (efficient)
│
└─ Selection: By agent type + task tags


TIER 3: PRE-HYDRATION (Deterministic Fetching)
┌─ Before LLM starts:
│  ├─ Parse task regex → extract ticket IDs, file paths
│  ├─ Fetch Jira tickets (via MCP tool)
│  ├─ Fetch recent git history
│  ├─ Sourcegraph code search
│  ├─ CI failures (if any)
│  └─ Inject all into context
│
└─ Result: Agent starts with full context, no "thinking" phase


┌────────────────────────────────────────────┐
│          MCP TOOLSHED SECURITY             │
├────────────────────────────────────────────┤
│                                            │
│ LAYER 1: TOOL-LEVEL RESTRICTIONS           │
│ ├─ get_jira_ticket() → READ-ONLY ✓         │
│ ├─ delete_jira_ticket() → BLOCKED ✗        │
│ ├─ trigger_ci_build() → WRITE ✓            │
│ └─ stripe_refund_all() → BLOCKED ✗         │
│                                            │
│ LAYER 2: ENVIRONMENT-LEVEL RESTRICTIONS    │
│ ├─ Devbox: QA databases only (no prod)     │
│ ├─ Devbox: No external internet            │
│ ├─ Devbox: VPC isolation                   │
│ └─ Devbox: IAM role restrictions           │
│                                            │
│ RESULT: Both layers must pass               │
│                                            │
└────────────────────────────────────────────┘
```

### Parallelism & Scale

```
WITHOUT DEVBOXES:
┌─ 1 machine
│  └─ 1 agent running
│     └─ 1 task processed
│        └─ Capacity: 1 PR/minute
│           (serial, no parallelism)

WITH DEVBOXES & PRE-WARMING POOL:
┌─ 200 machines (devbox pool)
│  ├─ Agent 1 → Devbox A → Task 1 ────→ PR 1
│  ├─ Agent 2 → Devbox B → Task 2 ────→ PR 2
│  ├─ Agent 3 → Devbox C → Task 3 ────→ PR 3
│  └─ ...
│  └─ Agent 200 → Devbox Z → Task 200 → PR 200
│
└─ Capacity: 200 PRs / 10 min = 1000+ PRs/week

PARALLELISM MULTIPLIER: 200x ✓
```

---

## Appendix: Key Metrics & Takeaways

### Stripe Minion Fleet Statistics
- **PRs per week:** 1000+
- **Success rate (CI passes on first try):** ~95%
- **Token efficiency:** 60–80% savings vs. pure agentic loops (blueprint-based)
- **Iteration cap:** Hard limit of 2 CI rounds per task
- **Devbox spin-up:** 10 seconds (from pool)
- **MCP tools in Toolshed:** ~500
- **Tools per curated agent:** 30–100
- **Context window efficiency:** Pre-hydration saves 50% of context vs. on-demand fetching
- **Supported payment volume:** $1 trillion+ annually

### Principles That Matter
1. **Hybrid > Pure Agentic:** Structure (deterministic) + creativity (agentic LLM) = reliable
2. **Shift Left:** Kill bugs cheaply before expensive CI
3. **Smaller Box:** Curated tools make better decisions than all tools
4. **Isolation Enables Trust:** Ephemeral devboxes let agents run fearlessly
5. **Pre-Hydrate:** Fetch context deterministically, not reactively
6. **Bounded Iteration:** Hard caps prevent runaway costs
7. **Cattle, Not Pets:** Treat environments as disposable
8. **What's Good for Humans is Good for Agents:** Apply successful patterns from human workflows

---
