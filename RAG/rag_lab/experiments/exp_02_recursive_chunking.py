"""
Experiment 02 — Recursive Chunking

Variable changed vs baseline:
  - Chunking: FixedSizeChunker → RecursiveChunker (langchain RecursiveCharacterTextSplitter)

Everything else is identical to exp_01_baseline:
  - Same chunk_size (512) and overlap (50) — only the splitting strategy changes
  - Same embedder, vectordb, retriever, reranker, generator, test questions

What to look for in the results:
  - answer_similarity delta vs baseline → did respecting text boundaries help?
  - chunk count difference → recursive chunking often produces fewer, cleaner chunks
  - faithfulness delta → cleaner chunks = less noise in context = better grounding?

Compare results against: results/exp_01_baseline_*.json
Run: uv run python experiments/exp_02_recursive.py
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
    RecursiveChunker, chunk_documents,
    SentenceTransformerEmbedder, embed_chunks,
    ChromaVectorDB,
    DenseRetriever,
    IdentityReranker,
    OpenAIGenerator,
    EvalSample, evaluate_pipeline,
)
from configs import BASELINE_CONFIG, RAGConfig
from configs.rag_config import VectorDBConfig

# One variable changed: use a separate collection so baseline index is untouched
cfg = RAGConfig(
    vectordb=VectorDBConfig(collection_name="rag_lab_exp02")
)

console = Console()


# ── 1. INGEST ─────────────────────────────────────────────────────────────────
def ingest(vectordb: ChromaVectorDB, embedder: SentenceTransformerEmbedder) -> int:
    corpus_dir = ROOT / "corpus"
    console.print(f"\n[bold cyan]Loading corpus from:[/] {corpus_dir}")
    docs = load_directory(corpus_dir, extensions=[".txt"])
    console.print(f"  {len(docs)} documents loaded")

    # RecursiveChunker — same size/overlap as baseline, different splitting strategy
    chunker = RecursiveChunker(
        chunk_size=cfg.chunking.chunk_size,
        overlap=cfg.chunking.overlap,
    )
    chunks = chunk_documents(docs, chunker)
    console.print(f"  {len(chunks)} chunks created (strategy=recursive, size={cfg.chunking.chunk_size}, overlap={cfg.chunking.overlap})")

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
def load_baseline_scores() -> dict[str, float] | None:
    """Load the most recent baseline result for delta comparison."""
    results_dir = ROOT / "results"
    baseline_files = sorted(results_dir.glob("exp_01_baseline_*.json"), reverse=True)
    if not baseline_files:
        return None
    data = json.loads(baseline_files[0].read_text())
    return data.get("scores")


def display_results(samples: list[EvalSample], scores: dict[str, float]) -> None:
    qa_table = Table(title="Per-Question Results", show_lines=True)
    qa_table.add_column("Question", style="cyan", max_width=40)
    qa_table.add_column("Answer", style="white", max_width=60)
    qa_table.add_column("Chunks Retrieved", style="dim", justify="center")

    for s in samples:
        qa_table.add_row(s.question, s.answer, str(len(s.contexts)))
    console.print(qa_table)

    baseline_scores = load_baseline_scores()

    score_table = Table(title="RAGAS Scores — Exp 02 vs Baseline", show_header=True)
    score_table.add_column("Metric", style="bold")
    score_table.add_column("Baseline", justify="right")
    score_table.add_column("Recursive", justify="right")
    score_table.add_column("Delta", justify="right")
    score_table.add_column("Interpretation")

    interpretations = {
        "faithfulness":      "[LLM judge]   Is the answer supported by retrieved context?",
        "answer_relevancy":  "[LLM judge]   Does the answer address the question?",
        "answer_similarity": "[Deterministic] Cosine sim vs ground truth — most reliable",
    }

    for metric, score in scores.items():
        color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
        baseline_val = baseline_scores.get(metric) if baseline_scores else None
        baseline_str = f"{baseline_val:.3f}" if baseline_val is not None else "n/a"

        if baseline_val is not None:
            delta = score - baseline_val
            delta_color = "green" if delta > 0.01 else "red" if delta < -0.01 else "dim"
            delta_str = f"[{delta_color}]{delta:+.3f}[/{delta_color}]"
        else:
            delta_str = "n/a"

        score_table.add_row(
            metric,
            baseline_str,
            f"[{color}]{score:.3f}[/{color}]",
            delta_str,
            interpretations.get(metric, ""),
        )

    console.print(score_table)


# ── 5. SAVE ───────────────────────────────────────────────────────────────────
def save_results(samples: list[EvalSample], scores: dict[str, float]) -> Path:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "exp_02_recursive",
        "timestamp": timestamp,
        "config": cfg.model_dump(),
        "variable_changed": "chunking_strategy: fixed → recursive",
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

    out_path = results_dir / f"exp_02_recursive_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2))
    return out_path


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    console.rule("[bold]Experiment 02 — Recursive Chunking")
    console.print(f"Variable changed: [yellow]chunking strategy[/] fixed → recursive")
    console.print(f"Config: chunk_size={cfg.chunking.chunk_size}, "
                  f"overlap={cfg.chunking.overlap}, "
                  f"top_k={cfg.retrieval.top_k}, "
                  f"generator={cfg.generation.model}")

    embedder = SentenceTransformerEmbedder(model_name=cfg.embedding.model_name)
    vectordb = ChromaVectorDB(
        collection_name=cfg.vectordb.collection_name,
        persist_dir=str(ROOT / "chroma_db"),
    )
    retriever = DenseRetriever(embedder=embedder, vectordb=vectordb)
    reranker = IdentityReranker()
    generator = OpenAIGenerator(model=cfg.generation.model, max_tokens=cfg.generation.max_tokens)

    console.print("\n[dim]Resetting vector index for clean run...[/dim]")
    vectordb.reset()
    ingest(vectordb, embedder)

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
    console.rule("[bold green]Exp 02 complete — compare delta vs baseline above")


if __name__ == "__main__":
    main()
