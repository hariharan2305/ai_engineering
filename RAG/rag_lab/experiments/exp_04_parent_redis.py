"""
Experiment 04-ParentDoc-Redis — Parent Document Retrieval (ChromaDB + Redis)

What changed vs baseline:
  - Retrieval: DenseRetriever (single chunk size, embeds and returns same chunks)
             → RedisParentRetriever (children in ChromaDB, parents in Redis)

Everything else is identical to the baseline:
  - Chunking:    Two-level (parent=512 tokens, child=128 tokens)
  - Embeddings:  SentenceTransformerEmbedder (all-MiniLM-L6-v2)
  - Reranking:   IdentityReranker (none)
  - Generation:  OpenAIGenerator (gpt-4o-mini)
  - Evaluation:  RAGAS (faithfulness + answer_relevancy + answer_similarity)

Architecture — ChromaDB + Redis two-store:
  Index time:
    corpus → parent chunks (512 tokens) → stored in Redis as "parent:{id}" → text
    each parent → child chunks (128 tokens) → embedded → stored in ChromaDB
    ChromaDB metadata carries parent_id for the reverse lookup

  Query time:
    query → embed → ChromaDB search children (top_k * 4 candidates)
    → deduplicate by parent_id from ChromaDB metadata
    → Redis pipeline batch-fetch parent texts (single round-trip)
    → return parent texts as context to generator

Why two separate stores (vs Qdrant single-store):
  ChromaDB handles high-dimensional vector similarity search.
  Redis handles sub-millisecond key-value lookups at scale.
  Each store is independently scalable and replaceable:
    - Swap ChromaDB → Pinecone/Weaviate without touching Redis
    - Swap Redis → DynamoDB/Postgres without rebuilding the vector index
  This is the production-grade pattern used by most large-scale RAG systems.

Prerequisite — start Redis with Docker:
  docker run -p 6379:6379 redis

Run: uv run python experiments/exp_04_parent_redis.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from components import (
    load_directory,
    FixedSizeChunker, chunk_documents,
    SentenceTransformerEmbedder, embed_chunks,
    ChromaVectorDB,
    RedisParentRetriever,
    IdentityReranker,
    OpenAIGenerator,
    EvalSample, evaluate_pipeline,
)
from configs import BASELINE_CONFIG

console = Console()
cfg = BASELINE_CONFIG
EXPERIMENT_NAME = "exp_04_parent_redis"
COLLECTION_NAME = "rag_lab_parent_redis"

# Two-level chunk sizes
PARENT_CHUNK_SIZE = 512   # rich context for generator
CHILD_CHUNK_SIZE  = 128   # precise embedding for retrieval
CHILD_OVERLAP     = 20

BASELINE_SCORES = {
    "faithfulness":      0.641,
    "answer_relevancy":  0.583,
    "answer_similarity": 0.538,
}


# ── 1. INGEST ─────────────────────────────────────────────────────────────────
def ingest(retriever: RedisParentRetriever) -> None:
    """
    Two-level chunking:
      1. Chunk corpus into parent chunks (large) → store in Redis
      2. For each parent, chunk into children (small) → embed → store in ChromaDB
         with parent_id in metadata for reverse lookup at query time
    """
    corpus_dir = ROOT / "corpus"
    console.print(f"\n[bold cyan]Loading corpus from:[/] {corpus_dir}")
    docs = load_directory(corpus_dir, extensions=[".txt"])
    console.print(f"  {len(docs)} documents loaded")

    # Level 1 — parent chunks
    parent_chunker = FixedSizeChunker(chunk_size=PARENT_CHUNK_SIZE, overlap=50)
    parent_chunks = chunk_documents(docs, parent_chunker)
    console.print(f"  {len(parent_chunks)} parent chunks (size={PARENT_CHUNK_SIZE}) → Redis")

    # Level 2 — child chunks from each parent, tagged with parent_id
    child_chunker = FixedSizeChunker(chunk_size=CHILD_CHUNK_SIZE, overlap=CHILD_OVERLAP)
    all_children = []

    for parent in parent_chunks:
        from components.base import Document
        parent_as_doc = Document(
            id=parent.id,
            text=parent.text,
            metadata=parent.metadata,
        )
        children = chunk_documents([parent_as_doc], child_chunker)
        for child in children:
            # parent_id in metadata is the key that ChromaDB carries for Redis lookup
            child.metadata["parent_id"] = parent.id
        all_children.extend(children)

    console.print(f"  {len(all_children)} child chunks (size={CHILD_CHUNK_SIZE}) → ChromaDB")

    # Embed children — these go into ChromaDB
    embedder = retriever._embedder
    child_embedded = embed_chunks(all_children, embedder)

    total_indexed = retriever.index(parent_chunks, child_embedded)
    console.print(
        f"  {len(parent_chunks)} parent texts stored in Redis\n"
        f"  {total_indexed} child embeddings stored in ChromaDB"
    )


# ── 2. QUERY ──────────────────────────────────────────────────────────────────
def run_query(
    question: str,
    retriever: RedisParentRetriever,
    reranker: IdentityReranker,
    generator: OpenAIGenerator,
) -> tuple[str, list[str]]:
    chunks = retriever.retrieve(question, top_k=cfg.retrieval.top_k)
    chunks = reranker.rerank(question, chunks)
    answer = generator.generate(question, chunks)
    contexts = [c.text for c in chunks]
    return answer, contexts


# ── 3. EVALUATE ───────────────────────────────────────────────────────────────
def run_evaluation(samples: list[EvalSample], embedder) -> dict[str, float]:
    console.print("\n[bold cyan]Running RAGAS evaluation...[/]")
    return evaluate_pipeline(samples, embedder=embedder)


# ── 4. DISPLAY ────────────────────────────────────────────────────────────────
def display_results(samples: list[EvalSample], scores: dict[str, float]) -> None:
    qa_table = Table(title="Per-Question Results", show_lines=True)
    qa_table.add_column("Question", style="cyan", max_width=40)
    qa_table.add_column("Answer", style="white", max_width=60)
    qa_table.add_column("Chunks Retrieved", style="dim", justify="center")

    for s in samples:
        qa_table.add_row(s.question, s.answer, str(len(s.contexts)))

    console.print(qa_table)

    score_table = Table(title=f"RAGAS Scores — {EXPERIMENT_NAME} vs Baseline", show_header=True)
    score_table.add_column("Metric", style="bold")
    score_table.add_column("Baseline", justify="right")
    score_table.add_column("This Run", justify="right")
    score_table.add_column("Delta", justify="right")
    score_table.add_column("Interpretation")

    interpretations = {
        "faithfulness":      "[LLM judge]   Is the answer supported by the retrieved context?",
        "answer_relevancy":  "[LLM judge]   Does the answer address the question?",
        "answer_similarity": "[Deterministic] Cosine sim vs ground truth — most reliable signal",
    }

    for metric, score in scores.items():
        baseline = BASELINE_SCORES.get(metric, 0.0)
        delta = score - baseline
        color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
        delta_color = "green" if delta > 0.01 else "red" if delta < -0.01 else "white"
        score_table.add_row(
            metric,
            f"{baseline:.3f}",
            f"[{color}]{score:.3f}[/{color}]",
            f"[{delta_color}]{delta:+.3f}[/{delta_color}]",
            interpretations.get(metric, ""),
        )

    console.print(score_table)


# ── 5. SAVE ───────────────────────────────────────────────────────────────────
def save_results(samples: list[EvalSample], scores: dict[str, float]) -> Path:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": EXPERIMENT_NAME,
        "timestamp": timestamp,
        "config": cfg.model_dump(),
        "retrieval_strategy": f"ParentDoc/Redis — child={CHILD_CHUNK_SIZE} in ChromaDB, parent={PARENT_CHUNK_SIZE} in Redis",
        "baseline_scores": BASELINE_SCORES,
        "scores": scores,
        "delta": {k: round(scores[k] - BASELINE_SCORES[k], 4) for k in scores},
        "samples": [
            {
                "question": s.question,
                "answer": s.answer,
                "num_contexts": len(s.contexts),
                "ground_truth": s.ground_truth,
            }
            for s in samples
        ],
    }

    out_path = results_dir / f"{EXPERIMENT_NAME}_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2))
    return out_path


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    console.rule("[bold]Experiment 04-ParentDoc-Redis — ChromaDB children + Redis parents")
    console.print(
        f"Child chunks (indexed): [cyan]{CHILD_CHUNK_SIZE} tokens[/] → ChromaDB (vector search)\n"
        f"Parent chunks (returned): [green]{PARENT_CHUNK_SIZE} tokens[/] → Redis (doc store)\n"
        f"top_k={cfg.retrieval.top_k}, generator={cfg.generation.model}"
    )
    console.print("\n[dim]Prerequisite: docker run -p 6379:6379 redis[/dim]")

    embedder = SentenceTransformerEmbedder(model_name=cfg.embedding.model_name)
    vectordb = ChromaVectorDB(
        collection_name=COLLECTION_NAME,
        persist_dir=str(ROOT / cfg.vectordb.persist_dir),
    )
    vectordb.reset()

    retriever = RedisParentRetriever(
        vectordb=vectordb,
        embedder=embedder,
    )
    reranker = IdentityReranker()
    generator = OpenAIGenerator(model=cfg.generation.model, max_tokens=cfg.generation.max_tokens)

    ingest(retriever)

    test_path = ROOT / "corpus" / "test_questions.json"
    test_data = json.loads(test_path.read_text())
    console.print(f"\n[bold cyan]Running {len(test_data)} test questions...[/]")

    samples = []
    for item in test_data:
        answer, contexts = run_query(item["question"], retriever, reranker, generator)
        samples.append(EvalSample(
            question=item["question"],
            answer=answer,
            contexts=contexts,
            ground_truth=item.get("ground_truth", ""),
        ))
        console.print(f"  [green]✓[/] {item['question'][:60]}...")

    scores = run_evaluation(samples, embedder=embedder)
    display_results(samples, scores)

    out_path = save_results(samples, scores)
    console.print(f"\n[dim]Results saved to: {out_path}[/dim]")
    console.rule("[bold green]Experiment 04-ParentDoc-Redis complete — check delta vs baseline above")


if __name__ == "__main__":
    main()
