# RAG Lab — Architecture Reference

## Design Principle

One variable changes per experiment. Everything else stays fixed. This is what makes score deltas meaningful.

---

## Directory Responsibilities

| Directory | What it owns | Changes between experiments? |
|---|---|---|
| `components/` | How each pipeline stage works | Only when swapping that component |
| `configs/` | What settings each stage uses | Yes — one value per experiment |
| `corpus/` | Test documents and questions | Never |
| `experiments/` | Orchestration scripts | New file per experiment |
| `results/` | Permanent score records | New JSON per run |

Each directory has exactly one reason to change. This separation is what makes comparisons valid.

---

## Data Flow

Data moves through the pipeline as typed objects defined in `components/base.py`. Each stage receives the previous stage's output type and produces the next.

```
corpus/*.txt
    |
    v
ingestion.py        raw files  -->  Document(id, text, metadata)
    |
    v
chunking.py         Document   -->  Chunk(id, text, doc_id, chunk_index)
    |
    v
embeddings.py       Chunk      -->  EmbeddedChunk(... + embedding: list[float])
    |
    v
vectordb.py         stores EmbeddedChunks, searches by vector similarity
    |
    |  <-- query from test_questions.json
    v
retrieval.py        query      -->  [RetrievedChunk(... + score: float)]
    |
    v
reranking.py        reorders   -->  [RetrievedChunk] (same type, different order)
    |
    v
generation.py       chunks + query  -->  answer (str)
    |
    v
evaluation.py       answer + ground_truth  -->  scores dict
    |
    v
results/*.json      config + scores + samples persisted to disk
```

---

## Role of Each File

### `components/base.py`
Defines the shared data types: `Document`, `Chunk`, `EmbeddedChunk`, `RetrievedChunk`, `RAGResult`.

These are the universal connectors. Every component agrees on these types, which is what allows any component to be swapped without breaking the rest of the pipeline.

### `configs/rag_config.py`
Defines `RAGConfig` — a Pydantic model with one sub-config per pipeline stage.

```
BASELINE_CONFIG = RAGConfig()   # all defaults, the zero point
```

Every component in an experiment is initialised from config values, not hardcoded constants. Changing one config field is how you isolate a variable for testing.

### `experiments/exp_01_baseline.py`
The orchestrator. Imports all components, reads `BASELINE_CONFIG`, runs the full pipeline end-to-end across all 8 test questions, then saves scores + config to `results/`.

This file is the zero point — never modified. All future experiments are new files that copy this structure and change one thing.

### `corpus/`
Three `.txt` knowledge documents and `test_questions.json` (8 questions with `ground_truth` answers). Fixed across all experiments. Changing this would contaminate comparisons.

### `results/`
One JSON file per run. Contains the config that produced the run, the scores, and the per-question answers. Used to compute delta against baseline.

---

## How an Experiment Differs from the Baseline

A new experiment (e.g. `exp_02_chunking.py`) is structurally identical to `exp_01_baseline.py` with one change:

```python
# exp_01_baseline.py
cfg = BASELINE_CONFIG                          # chunk_size=512

# exp_02_chunking.py
cfg = RAGConfig(chunking=ChunkingConfig(chunk_size=256))   # one change
```

Everything else — corpus, embedder, vectordb, retriever, generator, test questions — stays at baseline. The delta in `answer_similarity` between the two result JSONs is the measured contribution of that chunk size change.

---

## Evaluation Metrics

| Metric | Type | Signal |
|---|---|---|
| `faithfulness` | LLM judge | Is the answer supported by retrieved context? |
| `answer_relevancy` | LLM judge | Does the answer address the question? |
| `answer_similarity` | Deterministic cosine sim | How close is the answer to ground truth? |

`answer_similarity` is the most trusted signal. It uses local sentence-transformers with no LLM variance — if it moves, the component change caused it.
