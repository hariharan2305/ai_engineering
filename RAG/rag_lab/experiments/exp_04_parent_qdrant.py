"""
Experiment 04-ParentDoc-Qdrant — Parent Document Retrieval (Qdrant single-store)

What changed vs baseline:
  - Retrieval: DenseRetriever (single chunk size, embeds and returns same chunks)
             → QdrantParentRetriever (child chunks indexed, parent chunks returned)

Everything else is identical to the baseline:
  - Chunking:    Two-level (parent=512 tokens, child=128 tokens)
  - Embeddings:  SentenceTransformerEmbedder (all-MiniLM-L6-v2)
  - Reranking:   IdentityReranker (none)
  - Generation:  OpenAIGenerator (gpt-4o-mini)
  - Evaluation:  RAGAS (faithfulness + answer_relevancy + answer_similarity)

Architecture — Qdrant single-store:
  Index time:
    corpus → parent chunks (512 tokens) → child chunks (128 tokens per parent)
    each child → embed → upsert into Qdrant with payload:
      { parent_id, parent_text, child_text, doc_id, chunk_index }
    No secondary store — parent text lives in Qdrant payload.

  Query time:
    query → embed → search Qdrant children (top_k * 4 candidates)
    → deduplicate by parent_id (keep highest-scoring child per parent)
    → return parent_text from payload as context to generator
    Single network call — payload fetch is free with the search result.

Why two chunk sizes:
  Small child (128 tokens): embedding captures one focused idea → high precision retrieval
  Large parent (512 tokens): full surrounding context → generator produces complete answers
  This resolves the precision-vs-context tradeoff inherent in any fixed chunk size.

Prerequisite — start Qdrant with Docker:
  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

Run: uv run python experiments/exp_04_parent_qdrant.py
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
    SentenceTransformerEmbedder,
    QdrantParentRetriever,
    IdentityReranker,
    OpenAIGenerator,
    EvalSample, evaluate_pipeline,
)
from configs import BASELINE_CONFIG

console = Console()
cfg = BASELINE_CONFIG
EXPERIMENT_NAME = "exp_04_parent_qdrant"
COLLECTION_NAME = "rag_lab_parent_qdrant"

# Two-level chunk sizes
PARENT_CHUNK_SIZE = 512   # same as baseline — rich context for generator
CHILD_CHUNK_SIZE  = 128   # small — precise embedding for retrieval
CHILD_OVERLAP     = 20

BASELINE_SCORES = {
    "faithfulness":      0.641,
    "answer_relevancy":  0.583,
    "answer_similarity": 0.538,
}


# ── 1. INGEST ─────────────────────────────────────────────────────────────────
def ingest(retriever: QdrantParentRetriever) -> None:
    """
    Two-level chunking:
      1. Chunk corpus into parent chunks (large)
      2. For each parent, chunk further into children (small)
      3. Index children into Qdrant with parent text in payload
    """
    corpus_dir = ROOT / "corpus"
    console.print(f"\n[bold cyan]Loading corpus from:[/] {corpus_dir}")
    docs = load_directory(corpus_dir, extensions=[".txt"])
    console.print(f"  {len(docs)} documents loaded")

    # Level 1 — parent chunks
    parent_chunker = FixedSizeChunker(chunk_size=PARENT_CHUNK_SIZE, overlap=50)
    parent_chunks = chunk_documents(docs, parent_chunker)
    console.print(f"  {len(parent_chunks)} parent chunks (size={PARENT_CHUNK_SIZE})")

    # Level 2 — child chunks per parent
    child_chunker = FixedSizeChunker(chunk_size=CHILD_CHUNK_SIZE, overlap=CHILD_OVERLAP)
    child_chunks_by_parent: dict[str, list] = {}

    for parent in parent_chunks:
        from components.base import Document
        parent_as_doc = Document(
            id=parent.id,
            text=parent.text,
            metadata=parent.metadata,
        )
        children = chunk_documents([parent_as_doc], child_chunker)
        # Tag each child with its parent_id for retrieval lookup
        for child in children:
            child.metadata["parent_id"] = parent.id
        child_chunks_by_parent[parent.id] = children

    total_children = sum(len(v) for v in child_chunks_by_parent.values())
    console.print(f"  {total_children} child chunks (size={CHILD_CHUNK_SIZE}) across all parents")

    total_indexed = retriever.index(parent_chunks, child_chunks_by_parent)
    console.print(f"  {total_indexed} child embeddings indexed in Qdrant (parent text in payload)")


# ── 2. QUERY ──────────────────────────────────────────────────────────────────
def run_query(
    question: str,
    retriever: QdrantParentRetriever,
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
        "retrieval_strategy": f"ParentDoc/Qdrant — child={CHILD_CHUNK_SIZE} indexed, parent={PARENT_CHUNK_SIZE} returned",
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
    console.rule("[bold]Experiment 04-ParentDoc-Qdrant — Index Small, Retrieve Large")
    console.print(
        f"Child chunks (indexed): [cyan]{CHILD_CHUNK_SIZE} tokens[/] — precise retrieval\n"
        f"Parent chunks (returned): [green]{PARENT_CHUNK_SIZE} tokens[/] — rich context for generator\n"
        f"Store: [bold]Qdrant[/] (single store — parent text in payload)\n"
        f"top_k={cfg.retrieval.top_k}, generator={cfg.generation.model}"
    )
    console.print("\n[dim]Prerequisite: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant[/dim]")

    embedder = SentenceTransformerEmbedder(model_name=cfg.embedding.model_name)
    retriever = QdrantParentRetriever(
        collection_name=COLLECTION_NAME,
        embedder=embedder,
        embedding_dim=384,   # all-MiniLM-L6-v2 output dim
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
    console.rule("[bold green]Experiment 04-ParentDoc-Qdrant complete — check delta vs baseline above")


if __name__ == "__main__":
    main()
