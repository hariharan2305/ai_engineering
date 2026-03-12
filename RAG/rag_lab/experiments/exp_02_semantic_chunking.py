"""
Experiment 02b — Semantic Chunking

Variable changed vs baseline:
  - Chunking: FixedSizeChunker → SemanticChunker (LangChain SemanticChunker + local embeddings)

Everything else is identical to exp_01_baseline:
  - Same embedder (all-MiniLM-L6-v2), vectordb, retriever, reranker, generator, test questions

Key difference from recursive chunking:
  - Recursive: respects text structure (paragraphs, sentences) but still size-bounded
  - Semantic: ignores size entirely — splits where topic/meaning shifts, detected via
    cosine similarity between adjacent sentence embeddings
  - Real cost implications:
    - If using a local model (your case): CPU/GPU time, no dollar cost. Still slow.
    - If using an API embedder (e.g. text-embedding-3-small): you pay for every sentence embedding AND every chunk embedding. Effectively 2-3x the token cost of recursive chunking.
    - Re-ingestion cost: if your corpus changes frequently, you re-run this every time.
    ┌──────────────────────────────────┬───────────────────────────────────┬──────────────────────────────────┐
  │                                  │             Recursive             │             Semantic             │
  ├──────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────┤
  │ Embedding passes                 │ 1                                 │ 2                                │
  ├──────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────┤
  │ Relative ingestion cost          │ 1x                                │ 2-3x                             │
  ├──────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────┤
  │ Re-ingestion frequency tolerance │ High                              │ Low                              │
  ├──────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────┤
  │ Best for                         │ General purpose, frequent updates │ Heterogeneous docs, rare updates │
  ├──────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────┤
  │ Industry adoption                │ Default in most systems           │ Selective, high-value use cases  │
  └──────────────────────────────────┴───────────────────────────────────┴──────────────────────────────────┘

What to look for in the results:
  - answer_similarity delta vs baseline AND vs exp_02_recursive
  - chunk count: semantic typically produces fewer but larger, variable-size chunks
  - faithfulness: topic-coherent chunks = less irrelevant context noise in generation

Compare results against:
  - results/exp_01_baseline_*.json    (zero point)
  - results/exp_02_recursive_*.json   (previous chunking experiment)

Run: uv run python experiments/exp_02_semantic.py
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
    SemanticChunker, chunk_documents,
    SentenceTransformerEmbedder, embed_chunks,
    ChromaVectorDB,
    DenseRetriever,
    IdentityReranker,
    OpenAIGenerator,
    EvalSample, evaluate_pipeline,
)
from configs import BASELINE_CONFIG, RAGConfig
from configs.rag_config import VectorDBConfig

# Separate collection so baseline and recursive indexes are untouched
cfg = RAGConfig(
    vectordb=VectorDBConfig(collection_name="rag_lab_exp02_semantic")
)

console = Console()


# ── 1. INGEST ─────────────────────────────────────────────────────────────────
def ingest(vectordb: ChromaVectorDB, embedder: SentenceTransformerEmbedder) -> int:
    corpus_dir = ROOT / "corpus"
    console.print(f"\n[bold cyan]Loading corpus from:[/] {corpus_dir}")
    docs = load_directory(corpus_dir, extensions=[".txt"])
    console.print(f"  {len(docs)} documents loaded")

    # SemanticChunker — no chunk_size; boundaries are driven by meaning shifts
    chunker = SemanticChunker(
        model_name=cfg.embedding.model_name,
        breakpoint_threshold_type="percentile",  # splits at top-N% most dissimilar sentence pairs
    )
    chunks = chunk_documents(docs, chunker)
    console.print(f"  {len(chunks)} chunks created (strategy=semantic, threshold=percentile)")

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
    return evaluate_pipeline(samples, embedder=embedder)


# ── 4. DISPLAY ────────────────────────────────────────────────────────────────
def load_prior_scores() -> dict[str, dict[str, float]]:
    """Load baseline and recursive scores for 3-way comparison."""
    results_dir = ROOT / "results"
    prior = {}
    for exp_name in ("exp_01_baseline", "exp_02_recursive"):
        files = sorted(results_dir.glob(f"{exp_name}_*.json"), reverse=True)
        if files:
            prior[exp_name] = json.loads(files[0].read_text()).get("scores", {})
    return prior


def display_results(samples: list[EvalSample], scores: dict[str, float]) -> None:
    qa_table = Table(title="Per-Question Results", show_lines=True)
    qa_table.add_column("Question", style="cyan", max_width=40)
    qa_table.add_column("Answer", style="white", max_width=60)
    qa_table.add_column("Chunks Retrieved", style="dim", justify="center")
    for s in samples:
        qa_table.add_row(s.question, s.answer, str(len(s.contexts)))
    console.print(qa_table)

    prior = load_prior_scores()
    baseline = prior.get("exp_01_baseline", {})
    recursive = prior.get("exp_02_recursive", {})

    score_table = Table(title="RAGAS Scores — Chunking Strategy Comparison", show_header=True)
    score_table.add_column("Metric", style="bold")
    score_table.add_column("Baseline\n(fixed)", justify="right")
    score_table.add_column("Recursive", justify="right")
    score_table.add_column("Semantic", justify="right")
    score_table.add_column("Delta vs\nBaseline", justify="right")

    interpretations = {
        "faithfulness":      "[LLM judge]   Grounded in context?",
        "answer_relevancy":  "[LLM judge]   Addresses the question?",
        "answer_similarity": "[Deterministic] Cosine sim vs ground truth",
    }

    for metric, score in scores.items():
        color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
        base_val = baseline.get(metric)
        rec_val = recursive.get(metric)

        base_str = f"{base_val:.3f}" if base_val is not None else "n/a"
        rec_str = f"{rec_val:.3f}" if rec_val is not None else "n/a"

        if base_val is not None:
            delta = score - base_val
            delta_color = "green" if delta > 0.01 else "red" if delta < -0.01 else "dim"
            delta_str = f"[{delta_color}]{delta:+.3f}[/{delta_color}]"
        else:
            delta_str = "n/a"

        score_table.add_row(
            metric,
            base_str,
            rec_str,
            f"[{color}]{score:.3f}[/{color}]",
            delta_str,
        )

    console.print(score_table)


# ── 5. SAVE ───────────────────────────────────────────────────────────────────
def save_results(samples: list[EvalSample], scores: dict[str, float]) -> Path:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "exp_02_semantic",
        "timestamp": timestamp,
        "config": cfg.model_dump(),
        "variable_changed": "chunking_strategy: fixed → semantic",
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

    out_path = results_dir / f"exp_02_semantic_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2))
    return out_path


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    console.rule("[bold]Experiment 02b — Semantic Chunking")
    console.print(f"Variable changed: [yellow]chunking strategy[/] fixed → semantic")
    console.print(f"Config: model={cfg.embedding.model_name}, "
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
    console.rule("[bold green]Exp 02b complete — compare all 3 chunking strategies above")


if __name__ == "__main__":
    main()
