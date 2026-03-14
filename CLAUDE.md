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

---

## RAG Track

A third learning track focused on mastering RAG system building from beginner to advanced.

### Goal

Understand every RAG component deeply by building a lab bench where components are swapped one at a time and quality is measured via RAGAS. FastAPI is the final capstone only — not used during the learning phase.

### Lab Bench Location

`RAG/rag_lab/` — run all experiments from this directory with `uv run python experiments/<exp_file>.py`

### Key Files

- `RAG/rag_lab/components/` — reusable component modules (ingestion, chunking, embeddings, vectordb, retrieval, reranking, generation, evaluation)
- `RAG/rag_lab/configs/rag_config.py` — `BASELINE_CONFIG` and `RAGConfig` pydantic model; all experiments reference this
- `RAG/rag_lab/corpus/` — knowledge documents and test question sets:
  - `*.txt` — 3 prose documents (RAG theory, vector DBs, LLM evaluation)
  - `rag_advanced_techniques.md` — markdown-structured corpus for markdown chunking experiments
  - `embedding_models_guide.html` — HTML-structured corpus for HTML chunking experiments
  - `llm_pipeline_code.py` — production Python code corpus (LLMClient, EmbeddingCache, RAGPipeline) for code chunking experiments
  - `test_questions.json` — 8 QA pairs for prose corpus experiments
  - `test_questions_markdown.json`, `test_questions_html.json`, `test_questions_code.json` — domain-matched test sets for structured/code corpus experiments
- `RAG/rag_lab/experiments/` — one file per experiment (`exp_01_baseline.py`, `exp_02_chunking.py`, etc.)
- `RAG/rag_lab/results/` — JSON output per experiment run; compare against baseline JSON to measure delta

### Experiment Pattern (NEVER deviate from this)

- `exp_01_baseline.py` is the **zero point** — never modify it
- Each new experiment swaps **exactly one component**, everything else stays at baseline
- Always compare RAGAS scores against the baseline JSON in `results/`
- The delta tells you what that component actually contributed

### Curriculum Order (component-by-component)

1. **Chunking** — `exp_02_*` (complete): recursive, semantic, markdown, HTML (LangChain); SentenceSplitter, SemanticSplitter, TokenTextSplitter, CodeSplitter (LlamaIndex); contextual enrichment; late chunking concept file
2. **Embeddings** — MiniLM → larger/domain-specific models (`exp_03_*`)
3. **Retrieval** — dense-only → hybrid BM25+dense (`exp_04_*`)
4. **Reranking** — identity → cross-encoder reranker (`exp_05_*`)
5. **Generation** — prompt engineering, context window management (`exp_06_*`)
6. **Advanced retrieval** — HyDE, multi-query, query expansion (`exp_07_*`)
7. **FastAPI capstone** — wrap the best pipeline as a production API

### Evaluation Stack

- `faithfulness` — RAGAS LLM-judge: is the answer grounded in context?
- `answer_relevancy` — RAGAS LLM-judge: does the answer address the question?
- `answer_similarity` — **deterministic cosine sim** (sentence-transformers, no LLM); most trusted signal — if this moves, the improvement is real

### Models in Use

- Generator: `gpt-4o-mini` (OpenAI)
- RAGAS judge LLM: `gpt-4o-mini` (via LangchainLLMWrapper)
- RAGAS embeddings: `text-embedding-3-small` (via LangchainEmbeddingsWrapper)
- Local embedder: `all-MiniLM-L6-v2` (sentence-transformers, no API cost)

### RAG Component Stack Guide

`RAG/` contains deep-research cheat sheets for each component. These are the curriculum reference — read the relevant cheat sheet before building each experiment.

### Architecture Reference

`RAG/rag_lab/ARCHITECTURE.md` — explains the design rationale, full data flow diagram, and the role of every file and directory. Read this to understand why the lab is structured the way it is.

### Tech Stack Philosophy

Use well-tested library implementations where they exist — this builds hands-on familiarity with industry-standard tools. Add comments in experiment files explaining what the library does under the hood so the learning is preserved.

Preferred libraries per component:

| Component | Library |
|---|---|
| Chunking (recursive) | `langchain_text_splitters.RecursiveCharacterTextSplitter` |
| Chunking (semantic, LC) | `langchain_experimental.text_splitter.SemanticChunker` |
| Chunking (markdown) | `langchain_text_splitters.MarkdownHeaderTextSplitter` |
| Chunking (HTML) | `langchain_text_splitters.HTMLHeaderTextSplitter` |
| Chunking (LI sentence) | `llama_index.core.node_parser.SentenceSplitter` |
| Chunking (LI semantic) | `llama_index.core.node_parser.SemanticSplitterNodeParser` |
| Chunking (LI token) | `llama_index.core.node_parser.TokenTextSplitter` |
| Chunking (LI code/AST) | `llama_index.core.node_parser.CodeSplitter` (needs `tree-sitter-language-pack`) |
| Embeddings | `sentence-transformers` |
| Vector DB | `chromadb` directly |
| BM25 (hybrid retrieval) | `rank_bm25` |
| Cross-encoder reranking | `sentence-transformers` CrossEncoder |
| Generation | `openai` SDK directly |
| Evaluation | `ragas` |

All components must still conform to the `base.py` data types (`Document`, `Chunk`, etc.) — the pipeline interface does not change, only the internal implementation of each component.

### Import Verification Rule (MANDATORY)

Before writing any import statement for a third-party library, use Context7 MCP to verify the exact module path:
1. `resolve-library-id` — find the library
2. `query-docs` — confirm the exact class/function location

Never assume import paths. Example of a past mistake: `SemanticChunker` is in `langchain_experimental.text_splitter`, not `langchain_text_splitters`.
