# Eval Engineering — Learning Journal

This directory is the lab bench for learning how to evaluate AI systems — skills, agents, and pipelines. This file captures core learnings accumulated over the course of that journey.

## How to Update This File

- Capture the **essence of the insight**, not the source. Do not attribute learnings to specific articles, authors, or libraries unless the library/framework is directly needed to understand how to implement something.
- No article links, no author names, no "LangChain says..." or "the article mentions...". Just the distilled knowledge.
- Only mention specific tools or frameworks when they are the practical answer to "how do I actually do this?" — e.g. which library to use for a specific eval pattern.
- Keep entries conceptual and durable. Specific product names (LangChain, LangSmith, etc.) age poorly. Principles don't.
- When adding new learnings, check if an existing section already covers the concept and update it rather than creating a duplicate.

---

## What Eval Engineering Is

Eval engineering is the discipline of measuring whether an AI system (skill, agent, or pipeline) does what it's supposed to do — reliably, correctly, and consistently. It is the same discipline as software testing applied to nondeterministic systems.

The core loop is always:
1. Define what "correct" looks like (assertions / ground truth)
2. Run the system on test cases
3. Measure pass rate
4. Find the gap
5. Fix the system
6. Re-run and measure the delta

---

## Skill Evals — What We Learned

### The Three Things a Skill Eval Measures
1. **Trigger accuracy** — does the skill fire when it should, and not fire when it shouldn't?
2. **Workflow correctness** — does it follow the defined steps?
3. **Output quality** — is the output consistent with what the skill document specified?

### Assertion Types
- **Deterministic checks** — file exists, section present, word count, keyword preserved, path correct. Fast, cheap, always reproducible.
- **LLM-as-judge** — qualitative checks that can't be regex'd: voice preservation, hook quality, no clichés, emotional register. Use sparingly — only when structural checks can't cover the ground.

### The Baseline Comparison
Always run with-skill AND without-skill in parallel. The delta between them is what the skill actually contributes. Without a baseline, you don't know if your assertions are measuring skill value or just model capability.

From our learning-insights experiments:
- Without skill: "save this insight" → wrote to MEMORY.md or produced a .txt with no structure
- With skill: same prompt → structured 5-section markdown card in the correct directory with metadata

The skill's value was entirely structural enforcement — not content generation.

### Test Isolation Is Non-Negotiable
Skills that write to production paths will contaminate real data during test runs. Every eval run must write to a sandboxed path inside the iteration folder. In iteration 1 we learned this the hard way — test files ended up in the real `~/Documents/projects/learning-insights/` directory.

**Rule:** Executor agents must be given an explicit sandbox override path. The skill's production path must never be used during evals.

### Multi-Turn Context Matters
In iteration 1, we gave executor agents a single bundled prompt with a fake "conversation" embedded as text. This doesn't reflect how the skill is actually used — the skill reads prior conversation turns, not a synthetic block.

From iteration 2 onwards: executor agents receive a realistic multi-turn conversation history as prior context, and the skill invocation is the final turn. This is the correct way to simulate real usage.

### Eval Quality Is As Important As Skill Quality
Weak assertions give false confidence. Key failure modes we encountered:
- **Assertions that always pass for baseline runs** — e.g. "skill triggered" fails by definition in without-skill runs. Separate skill invocation checks from content output checks.
- **Assertions that accept opposite behaviors** — e.g. "either clarification OR capture is fine" is not a test.
- **Contaminated grading** — without-skill run "passed" because the with-skill run had already created the file in the shared directory. Fix: isolate storage per run.
- **Overly literal string matching** — "28%." catches one pattern but "28% jump." on its own line is the same cliché. Pattern-based assertions beat literal string matching.

### No-Intent Eval Design
A `/capture-insight` with no intent given over a single-topic conversation is not ambiguous — the skill correctly captures it without asking. A genuine no-intent test requires a multi-topic conversation (3+ unrelated subjects) where the skill genuinely cannot infer which one to focus on. Only then does "ask for clarification" become the meaningful correct behavior to test.

### Progress Across Three Iterations (learning-insights skill)
| Iteration | With Skill | Without Skill | Delta | Key Change |
|-----------|-----------|---------------|-------|-----------|
| 1 | 94.6% | 73.4% | +21% | Baseline — contaminated evals, single-turn prompts |
| 2 | 94.2% | 57.0% | +37% | Multi-turn context, sandboxed storage |
| 3 | 97.5% | 50.9% | +47% | Hook rule tightened, no-intent eval redesigned |

The delta growing from +21% to +47% was mostly the evals getting more accurate — not the skill getting dramatically better. Better evals expose the real gap.

---

## Skill Eval vs Agent Eval

| Dimension | Skill Eval | Agent Eval |
|-----------|-----------|-----------|
| Scope | One specific behavior | End-to-end task completion |
| Assertions | Structural + voice/quality | Structural + correctness + reasoning |
| Baseline | Without skill = plain Claude | Without agent = nothing (or simpler agent) |
| Key failure modes | Trigger mismatch, schema violations | Tool misuse, wrong planning, hallucination, cascading errors |
| Nondeterminism | Low | High — variance compounds across steps |
| Runs needed | 1-3 per eval | 5-10+ per eval |
| Ground truth needed | Rarely | Almost always |

### Ground Truth for Agent Evals
Agent evals require a reference answer to compare the agent's output against. You can't just check structure — you have to check correctness. Ground truth needs to be authored by humans (or a trusted oracle), which makes agent evals significantly more expensive to build than skill evals.

Ground truth layers:
- **Answer ground truth** — what is the correct final answer?
- **Trajectory ground truth** — what is the correct sequence of steps/tools?
- **Reasoning ground truth** — is the justification sound?

---

## The Eval Harness Architecture (for Skills)

```
evals.json                    ← test definitions (prompts + assertions)
iteration-N/
├── sandbox/                  ← isolated storage for this iteration's runs
├── eval-<name>/
│   ├── eval_metadata.json    ← per-eval prompt + assertions
│   ├── with_skill/
│   │   ├── outputs/          ← what the skill produced
│   │   ├── timing.json       ← tokens + duration
│   │   └── run-1/
│   │       └── grading.json  ← pass/fail per assertion with evidence
│   └── without_skill/
│       └── ...               ← same structure, baseline run
├── benchmark.json            ← aggregated pass rates, delta
└── benchmark.md              ← human-readable summary
```

The filesystem is the communication layer. Executor agents write to `outputs/`. Grader agents read from `outputs/`. The benchmark script reads all `grading.json` files. Nothing talks to anything else directly.

---

## Agentic Product Evals

### "Agent eval" is a misleading term

What teams building agentic products are actually doing is **product capability testing**. They define what their product claims to do, then verify those claims hold under controlled conditions. The LLM underneath, the subagents spawned, the parallel execution — these are implementation details. The eval lives above all of that and only asks: did the product do what it promised?

The term "agent eval" conflates two distinct things:

| Level | What it tests | Analogy |
|-------|-------------|---------|
| Component level | One agent's decision in one step (tool selection, recovery) | Unit test |
| Product level | Full product capability end-to-end | Integration test |

Most "agent eval" articles mean product level but slip into component level without distinguishing the two.

### There is no universal scoring function

For a skill or LLM call, correctness has one shape — does the output match the expected structure or answer. You can write one assertion pattern and reuse it.

For an agentic product, correctness looks different per capability:

| Test case | What correct means |
|-----------|-------------------|
| Tool selection | Did it call the right tool with the right arguments? |
| Memory update | Did state change correctly after this turn? |
| File artifact | Did the output file have the right structure? |
| Error recovery | Did it retry correctly after the tool failed? |

No single scoring function covers all of these. Each capability needs its own assertion logic. This is what "bespoke test logic per datapoint" means — not that every user's use case needs a test, but that every capability type has a different definition of correct.

### What LangChain is actually testing

They are not testing every possible user workflow. They are verifying that the agent's internal capabilities are reliable:
- Does it select the right tool for a given state?
- Does it maintain context across turns?
- Does it produce correct artifacts?
- Does it recover from failures?

These are capability tests, not use-case tests. The distinction matters — you don't need to anticipate every user. You need to define and verify your product's capability contract.

### Three layers of what to evaluate in a full agent run
- **Trajectory** — did it call the right tools in the right sequence?
- **Final output** — is the response correct and high quality?
- **Artifacts** — did it produce the right files, data, or side effects?

### Single-step evals are underrated
You don't always need to run the full agent. Isolate one decision point and test it in isolation — cheaper, faster, and catches regressions without running the full pipeline. LangChain used single-step evals for about half their test cases.

### Environment isolation applies here too
Agentic products interact with real systems — filesystems, APIs, databases. Tests that share state produce flaky results. Every test needs a clean, reproducible environment: fresh directories, mocked API calls, reset state. Same principle as sandbox isolation in skill evals, at a higher complexity level.

---

## Key Principle: Test Trigger Accuracy Before Output Quality

For skills, most failures are trigger-related — the skill doesn't fire when it should, or fires when it shouldn't. Fixing the trigger description fixes more failures than improving the skill body.

**Test that the system activates correctly first. A system that doesn't fire reliably has zero value regardless of how good its output is.**

---

## Directory Structure

```
eval_engineering/
├── CLAUDE.md               ← this file
├── skills/                 ← skill eval projects
│   └── learning-insights-eval-setup/
│       ├── iteration-1/
│       ├── iteration-2/
│       └── iteration-3/
└── agents/                 ← agent eval projects (future)
```
