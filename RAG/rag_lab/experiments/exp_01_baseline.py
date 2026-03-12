"""
Experiment 01 — Naive Baseline RAG

Components used:
  - Ingestion:   load_directory (plain text files)
  - Chunking:    FixedSizeChunker (chunk_size=512, overlap=50)
  - Embeddings:  SentenceTransformerEmbedder (all-MiniLM-L6-v2)
  - VectorDB:    ChromaVectorDB (cosine similarity, local)
  - Retrieval:   DenseRetriever (top_k=5)
  - Reranking:   IdentityReranker (none)
  - Generation:  ClaudeGenerator (claude-haiku-4-5-20251001)
  - Evaluation:  RAGAS (faithfulness + answer_relevancy)

This is the baseline all future experiments measure delta against.
Run: uv run python experiments/exp_01_baseline.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import print as rprint

# ── project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from components import (
    load_directory,
    FixedSizeChunker, chunk_documents,
    SentenceTransformerEmbedder, embed_chunks,
    ChromaVectorDB,
    DenseRetriever,
    IdentityReranker,
    OpenAIGenerator,
    EvalSample, evaluate_pipeline,
)
from configs import BASELINE_CONFIG

console = Console()
cfg = BASELINE_CONFIG


# ── 1. INGEST ─────────────────────────────────────────────────────────────────
def ingest(vectordb: ChromaVectorDB, embedder: SentenceTransformerEmbedder) -> int:
    corpus_dir = ROOT / "corpus"
    console.print(f"\n[bold cyan]Loading corpus from:[/] {corpus_dir}")
    docs = load_directory(corpus_dir, extensions=[".txt"])
    console.print(f"  {len(docs)} documents loaded")

    chunker = FixedSizeChunker(
        chunk_size=cfg.chunking.chunk_size,
        overlap=cfg.chunking.overlap,
    )
    chunks = chunk_documents(docs, chunker)
    console.print(f"  {len(chunks)} chunks created (size={cfg.chunking.chunk_size}, overlap={cfg.chunking.overlap})")

    embedded = embed_chunks(chunks, embedder)
    vectordb.add_chunks(embedded)
    console.print(f"  {vectordb.count()} chunks indexed in ChromaDB")
    return len(chunks)


# ── 2. QUERY ──────────────────────────────────────────────────────────────────
def run_query(
    question: str,
    retriever: DenseRetriever,
    reranker: IdentityReranker,
    generator: OpenAIGenerator,
) -> tuple[str, list[str]]:
    chunks = retriever.retrieve(question, top_k=cfg.retrieval.top_k)
    chunks = reranker.rerank(question, chunks)
    answer = generator.generate(question, chunks)
    contexts = [c.text for c in chunks]
    return answer, contexts
           

# ── 3. EVALUATE ───────────────────────────────────────────────────────────────
def run_evaluation(samples: list[EvalSample], embedder=None) -> dict[str, float]:
    console.print("\n[bold cyan]Running RAGAS evaluation...[/]")
    scores = evaluate_pipeline(samples, embedder=embedder)
    return scores


# ── 4. DISPLAY ────────────────────────────────────────────────────────────────
def display_results(samples: list[EvalSample], scores: dict[str, float]) -> None:
    # Per-question table
    qa_table = Table(title="Per-Question Results", show_lines=True)
    qa_table.add_column("Question", style="cyan", max_width=40)
    qa_table.add_column("Answer", style="white", max_width=60)
    qa_table.add_column("Chunks Retrieved", style="dim", justify="center")

    for s in samples:
        qa_table.add_row(s.question, s.answer, str(len(s.contexts)))

    console.print(qa_table)

    # Score summary
    score_table = Table(title="RAGAS Scores (Baseline)", show_header=True)
    score_table.add_column("Metric", style="bold")
    score_table.add_column("Score", justify="right")
    score_table.add_column("Interpretation")

    interpretations = {
        "faithfulness":      "[LLM judge]   Is the answer supported by the retrieved context?",
        "answer_relevancy":  "[LLM judge]   Does the answer address the question?",
        "answer_similarity": "[Deterministic] Cosine sim vs ground truth — most reliable signal",
    }
    for metric, score in scores.items():
        color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
        score_table.add_row(
            metric,
            f"[{color}]{score:.3f}[/{color}]",
            interpretations.get(metric, ""),
        )

    console.print(score_table)


# ── 5. SAVE ───────────────────────────────────────────────────────────────────
def save_results(samples: list[EvalSample], scores: dict[str, float]) -> Path:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "exp_01_baseline",
        "timestamp": timestamp,
        "config": cfg.model_dump(),
        "scores": scores,
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

    out_path = results_dir / f"exp_01_baseline_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2))
    return out_path


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    console.rule("[bold]Experiment 01 — Naive Baseline RAG")
    console.print(f"Config: chunk_size={cfg.chunking.chunk_size}, "
                  f"model={cfg.embedding.model_name}, "
                  f"top_k={cfg.retrieval.top_k}, "
                  f"generator={cfg.generation.model}")

    # Init components
    embedder = SentenceTransformerEmbedder(model_name=cfg.embedding.model_name)
    vectordb = ChromaVectorDB(
        collection_name=cfg.vectordb.collection_name,
        persist_dir=str(ROOT / cfg.vectordb.persist_dir),
    )
    retriever = DenseRetriever(embedder=embedder, vectordb=vectordb)
    reranker = IdentityReranker()
    generator = OpenAIGenerator(model=cfg.generation.model, max_tokens=cfg.generation.max_tokens)

    # Fresh index for reproducibility
    console.print("\n[dim]Resetting vector index for clean run...[/dim]")
    vectordb.reset()
    ingest(vectordb, embedder)

    # Load test questions
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

    # Evaluate (pass embedder for deterministic answer_similarity)
    scores = run_evaluation(samples, embedder=embedder)

    # Display
    display_results(samples, scores)

    # Save
    out_path = save_results(samples, scores)
    console.print(f"\n[dim]Results saved to: {out_path}[/dim]")
    console.rule("[bold green]Baseline complete — this is your zero point")


if __name__ == "__main__":
    main()
