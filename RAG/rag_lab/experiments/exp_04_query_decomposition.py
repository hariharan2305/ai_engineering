"""
Experiment 04-QueryDecomposition — Parallel Query Decomposition with RRF

What changed vs baseline:
  - Retrieval: DenseRetriever (single query, cosine similarity)
             → QueryDecompositionRetriever (LLM decomposes into sub-questions,
               parallel retrieval per sub-question, RRF fusion)

Everything else is identical to the baseline:
  - Chunking:    FixedSizeChunker (chunk_size=512, overlap=50)
  - Embeddings:  SentenceTransformerEmbedder (all-MiniLM-L6-v2)
  - VectorDB:    ChromaVectorDB (cosine similarity, local)
  - Reranking:   IdentityReranker (none)
  - Generation:  OpenAIGenerator (gpt-4o-mini)
  - Evaluation:  RAGAS (faithfulness + answer_relevancy + answer_similarity)

How QueryDecompositionRetriever works (under the hood):
  1. LLM (gpt-4o-mini) breaks the original query into up to 3 independent sub-questions,
     each targeting a distinct piece of information needed to answer the original.
  2. All sub-questions are sent to DenseRetriever in parallel (ThreadPoolExecutor).
     Each retrieves top_k * 2 candidates — a larger pool for better RRF coverage.
  3. All result lists are fused with RRF (k=60):
         rrf_score(chunk) = Σ  1 / (60 + rank_i)   across all sub-question result lists
     Chunks appearing across multiple sub-question results score higher.
  4. Top-k from the fused list passed to the generator — which now has complete
     context for each part of the original question.

Why this is different from multi-query:
  Multi-query   → rephrases the same question (covers vocabulary gaps)
  Decomposition → breaks a complex question into parts (covers knowledge gaps)

  A single retrieval call for a multi-part question retrieves context that partially
  addresses each sub-question but fully addresses none. Decomposition ensures each
  distinct information need gets its own targeted retrieval pass.

Hypothesis:
  - Questions that are genuinely multi-part will show the biggest gains.
  - Our test set (8 QA pairs on a technical corpus) may have limited multi-part
    questions, so gains may be modest — but the mechanism will be validated.
  - answer_similarity is the most reliable signal to watch.

Cost per query:
  1 LLM call (decompose) + up to 3 parallel retrieval calls.
  Same cost structure as multi-query.

Run: uv run python experiments/exp_04_query_decomposition.py
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
    DenseRetriever,
    QueryDecompositionRetriever,
    IdentityReranker,
    OpenAIGenerator,
    EvalSample, evaluate_pipeline,
)
from configs import BASELINE_CONFIG

console = Console()
cfg = BASELINE_CONFIG
EXPERIMENT_NAME = "exp_04_query_decomposition"
COLLECTION_NAME = "rag_lab_exp04_decomposition"

BASELINE_SCORES = {
    "faithfulness":      0.641,
    "answer_relevancy":  0.583,
    "answer_similarity": 0.538,
}


# ── 1. INGEST ─────────────────────────────────────────────────────────────────
def ingest(vectordb: ChromaVectorDB, embedder: SentenceTransformerEmbedder) -> None:
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


# ── 2. QUERY ──────────────────────────────────────────────────────────────────
def run_query(
    question: str,
    retriever: QueryDecompositionRetriever,
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
        "retrieval_strategy": "QueryDecomposition — LLM breaks query into sub-questions, parallel retrieval, RRF fusion",
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
    console.rule("[bold]Experiment 04-QueryDecomposition — Parallel Sub-Question Retrieval")
    console.print(
        "Swapping: [red]DenseRetriever (single query)[/] → "
        "[green]QueryDecompositionRetriever (LLM decomposes → parallel retrieval → RRF)[/]\n"
        f"Everything else: chunk_size={cfg.chunking.chunk_size}, "
        f"top_k={cfg.retrieval.top_k}, generator={cfg.generation.model}"
    )
    console.print("[dim]Each query: 1 LLM call (decompose) + up to 3 parallel retrieval calls[/dim]")

    embedder = SentenceTransformerEmbedder(model_name=cfg.embedding.model_name)
    vectordb = ChromaVectorDB(
        collection_name=COLLECTION_NAME,
        persist_dir=str(ROOT / cfg.vectordb.persist_dir),
    )
    dense_retriever = DenseRetriever(embedder=embedder, vectordb=vectordb)
    retriever = QueryDecompositionRetriever(
        base_retriever=dense_retriever,
        max_sub_queries=3,
        model=cfg.generation.model,
    )
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
    console.rule("[bold green]Experiment 04-QueryDecomposition complete — check delta vs baseline above")


if __name__ == "__main__":
    main()
