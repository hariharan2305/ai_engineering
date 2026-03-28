"""
Experiment 04-MultiQuery — Query-Side Retrieval: Multi-Query with RRF

What changed vs baseline:
  - Retrieval: DenseRetriever (single query, cosine similarity)
             → MultiQueryRetriever (3 LLM rephrasings + original, parallel retrieval, RRF fusion)

Everything else is identical to the baseline:
  - Chunking:    FixedSizeChunker (chunk_size=512, overlap=50)
  - Embeddings:  SentenceTransformerEmbedder (all-MiniLM-L6-v2)
  - VectorDB:    ChromaVectorDB (cosine similarity, local)
  - Reranking:   IdentityReranker (none)
  - Generation:  OpenAIGenerator (gpt-4o-mini)
  - Evaluation:  RAGAS (faithfulness + answer_relevancy + answer_similarity)

How MultiQueryRetriever works (under the hood):
  1. An LLM (gpt-4o-mini) generates 3 alternative phrasings of the original query.
     Each rephrasing uses different vocabulary to express the same intent, covering
     different regions of the embedding space.
  2. All 4 queries (original + 3 rephrasings) are sent to DenseRetriever in parallel
     using ThreadPoolExecutor — no serial latency penalty.
  3. The 4 ranked result lists are fused with RRF (k=60):
         rrf_score(chunk) = Σ  1 / (60 + rank_i)   across all 4 lists
     Chunks appearing in multiple query results (cross-query consensus) score higher.
  4. Top-k from the fused list are returned.

Why this is a query-side fix, not an index-side fix:
  The index and retriever don't change — only the query does. This isolates the effect
  of vocabulary mismatch between how users phrase questions and how the corpus is written.
  If the corpus uses "factual grounding" and the user asks about "hallucination", a single
  dense query may miss the relevant chunk. A rephrasing like "reduce LLM factual errors"
  will find it.

Hypothesis:
  - answer_similarity should improve over baseline dense (vocabulary gap is bridged).
  - Improvement will be smaller than BM25 on this corpus (corpus already rewards keywords).
  - Questions with paraphrase/synonym gaps will show the biggest per-question gains.

Note on cost:
  Each question now incurs 1 LLM call (rephrase) + 4 retrieval calls vs 1 in baseline.
  For 8 test questions: 8 extra LLM calls at gpt-4o-mini pricing — negligible for eval.

Run: uv run python experiments/exp_04_multi_query.py
"""

import json
import sys
import argparse
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
    MultiQueryRetriever,
    LangChainMultiQueryRetriever,
    LlamaIndexMultiQueryRetriever,
    IdentityReranker,
    OpenAIGenerator,
    EvalSample, evaluate_pipeline,
)
from configs import BASELINE_CONFIG

console = Console()
cfg = BASELINE_CONFIG
EXPERIMENT_NAME = "exp_04_multi_query"
COLLECTION_NAME = "rag_lab_exp04_multiquery"

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
    retriever: MultiQueryRetriever,
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
def save_results(samples: list[EvalSample], scores: dict[str, float], retriever_name: str) -> Path:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    strategy_labels = {
        "custom":     "Custom MultiQueryRetriever — DenseRetriever + 3 rephrasings + RRF",
        "langchain":  "LangChain MultiQueryRetriever — DenseRetriever + union dedup",
        "llamaindex": "LlamaIndex QueryFusionRetriever — DenseRetriever + RECIPROCAL_RANK",
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": f"{EXPERIMENT_NAME}_{retriever_name}",
        "timestamp": timestamp,
        "config": cfg.model_dump(),
        "retrieval_strategy": strategy_labels[retriever_name],
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

    out_path = results_dir / f"{EXPERIMENT_NAME}_{retriever_name}_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2))
    return out_path


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 04 — Multi-Query Retrieval")
    parser.add_argument(
        "--retriever",
        choices=["custom", "langchain", "llamaindex"],
        default="custom",
        help=(
            "custom     — our own MultiQueryRetriever (RRF fusion, parallel calls)\n"
            "langchain  — langchain_classic.retrievers.MultiQueryRetriever (union dedup)\n"
            "llamaindex — llama_index QueryFusionRetriever (RECIPROCAL_RANK fusion)"
        ),
    )
    args = parser.parse_args()
    retriever_name = args.retriever

    retriever_labels = {
        "custom":     "Custom (RRF, parallel)",
        "langchain":  "LangChain MultiQueryRetriever (union dedup)",
        "llamaindex": "LlamaIndex QueryFusionRetriever (RECIPROCAL_RANK)",
    }

    console.rule(f"[bold]Experiment 04-MultiQuery — {retriever_labels[retriever_name]}")
    console.print(
        f"Retriever: [green]{retriever_labels[retriever_name]}[/]\n"
        f"Base: DenseRetriever, num_queries=3, chunk_size={cfg.chunking.chunk_size}, "
        f"top_k={cfg.retrieval.top_k}, generator={cfg.generation.model}"
    )

    embedder = SentenceTransformerEmbedder(model_name=cfg.embedding.model_name)
    vectordb = ChromaVectorDB(
        collection_name=COLLECTION_NAME,
        persist_dir=str(ROOT / cfg.vectordb.persist_dir),
    )
    dense_retriever = DenseRetriever(embedder=embedder, vectordb=vectordb)

    if retriever_name == "custom":
        retriever = MultiQueryRetriever(
            base_retriever=dense_retriever,
            num_queries=3,
            model=cfg.generation.model,
        )
    elif retriever_name == "langchain":
        retriever = LangChainMultiQueryRetriever(
            base_retriever=dense_retriever,
            num_queries=3,
            model=cfg.generation.model,
        )
    else:  # llamaindex
        retriever = LlamaIndexMultiQueryRetriever(
            base_retriever=dense_retriever,
            num_queries=3,
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

    out_path = save_results(samples, scores, retriever_name)
    console.print(f"\n[dim]Results saved to: {out_path}[/dim]")
    console.rule(f"[bold green]Experiment 04-MultiQuery ({retriever_name}) complete")


if __name__ == "__main__":
    main()
