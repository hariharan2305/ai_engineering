# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

A structured learning repository for a Senior ML Engineer building **production GenAI application backends**. The goal is shipping GenAI products (not becoming a backend specialist). All examples, exercises, and concepts are tailored to LLM application patterns (streaming, multi-provider integration, token tracking, RAG, agents).

## Repository Structure

Two independent learning tracks:

- **FastAPI/** — Primary track: FastAPI for GenAI backends (6-week fast-track roadmap)
- **Prompting/** — Secondary track: DSPy framework and meta-prompting

### FastAPI Track Layout

- `FastAPI/FastAPI_GenAI_Builder_Roadmap.md` — **Master roadmap** tracking all progress, phases, topics, and next actions. Always read this first to understand current state.
- `FastAPI/concepts/NN_TopicName.md` — Concept documents (theory, diagrams, examples)
- `FastAPI/concepts/NN_TopicName_Practice.md` — Hands-on practice documents (step-by-step exercises with full code)
- `FastAPI/projects/fastapi_concepts_hands_on/` — Implementation files the user writes themselves

### Naming Conventions

- Concept docs: `NN_TopicName.md` (e.g., `03_Error_Handling.md`)
- Practice docs: `NN_TopicName_Practice.md` (e.g., `03_Error_Handling_Practice.md`)
- Implementation files: `NN_description.py` or `NN_M_description.py` for sub-exercises (e.g., `06_2_working_with_db_session_dependency.py`)
- File numbering is sequential across topics (Topics 1-2 use files 01-06_4, Topic 3 uses 07_1-07_5, Topic 4 uses 08_1-08_4)

## Common Commands

```bash
# Run a FastAPI exercise (from the hands-on directory)
cd FastAPI/projects/fastapi_concepts_hands_on
uv run uvicorn 01_hello_world:app --reload --port 8000

# Run a specific exercise by filename
uv run uvicorn 07_3_working_with_error_handling_global_exception_handlers:app --reload --port 8000

# Install dependencies
cd FastAPI/projects/fastapi_concepts_hands_on
uv sync
```

## Document Formatting Patterns

When creating new concept or practice documents, follow these established patterns exactly:

### Concept Documents (`NN_TopicName.md`)
1. H1 title with topic number
2. Context blockquote explaining why this matters for GenAI builders
3. Table of Contents with anchor links
4. Sections with ASCII diagrams for architecture/flow visualization
5. ✅/❌ code examples showing correct vs incorrect patterns
6. `★ Key Insight` boxes for important takeaways
7. Quick Reference table at the end
8. Next Steps section linking to the practice doc and next topic

### Practice Documents (`NN_TopicName_Practice.md`)
Each exercise follows: **Goal → Steps → Complete Code → Test It → What You Should See → Key Takeaway**

Exercises include time estimates and are progressive in complexity. All code uses GenAI-specific scenarios (LLM providers, chat endpoints, token budgets, rate limits).

## Key Conventions

- **Do NOT create implementation .py files** — the user writes those themselves as part of learning
- **Only create concept and practice .md files** when advancing to new topics
- Always update `FastAPI_GenAI_Builder_Roadmap.md` when marking topics complete or starting new ones
- Use GenAI-specific examples throughout (React chat UIs, LLM API calls, streaming responses, multi-provider patterns)
- The user's context: Senior ML Engineer familiar with Python, distributed systems (Spark), and AWS — draw parallels to these when explaining concepts

## Dependencies

FastAPI stack defined in `FastAPI/projects/fastapi_concepts_hands_on/pyproject.toml`:
- FastAPI, Uvicorn, Pydantic v2+, pydantic-settings, SQLAlchemy 2.0+, aiosqlite
- Python 3.12+, managed with `uv`
