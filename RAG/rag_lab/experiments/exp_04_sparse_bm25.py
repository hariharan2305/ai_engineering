"""
Experiment 04 — Sparse Retrieval: BM25Okapi

What changed vs baseline:
  - Retrieval: DenseRetriever (cosine similarity on embeddings)
             → BM25Retriever (keyword frequency scoring, no embeddings at query time)

Everything else is identical to the baseline:
  - Chunking:    FixedSizeChunker (chunk_size=512, overlap=50)
  - Embeddings:  SentenceTransformerEmbedder (all-MiniLM-L6-v2) — used only for answer_similarity eval
  - VectorDB:    NOT used — BM25 indexes chunks directly in memory
  - Reranking:   IdentityReranker (none)
  - Generation:  OpenAIGenerator (gpt-4o-mini)
  - Evaluation:  RAGAS (faithfulness + answer_relevancy + answer_similarity)

How BM25 works (under the hood):
  BM25Okapi scores each chunk against a query using three signals:
    1. Term Frequency (TF): how often the query token appears in the chunk,
       with saturation (doubling occurrences doesn't double the score; k1=1.5).
    2. Inverse Document Frequency (IDF): tokens rare across all chunks score higher.
       "embeddings" beats "the" because IDF("embeddings") >> IDF("the").
    3. Length normalization: long chunks are penalized to prevent them winning
       purely by size (b=0.75 controls this).
  Index is built from tokenized chunk text (lowercase, punctuation stripped).
  Query is tokenized the same way. No neural network — pure counting.

Why this experiment is useful:
  - Identifies where keyword matching outperforms semantic similarity
    (exact terms, rare technical tokens, product names, version numbers).
  - Sets the floor for hybrid: if BM25 > dense on certain questions,
    hybrid fusion will capture those wins.

Hypothesis:
  - Overall RAGAS scores will likely be lower than baseline (BM25 misses synonyms/paraphrases).
  - Some specific questions with exact technical keywords may score higher.
  - answer_similarity is the signal to watch — it directly reflects retrieval quality.

Run: uv run python experiments/exp_04_sparse_bm25.py
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
    BM25Retriever,
    IdentityReranker,
    OpenAIGenerator,
    EvalSample, evaluate_pipeline,
)
from configs import BASELINE_CONFIG

console = Console()
cfg = BASELINE_CONFIG
EXPERIMENT_NAME = "exp_04_sparse_bm25"


# ── 1. INGEST ─────────────────────────────────────────────────────────────────
def ingest() -> tuple[list, BM25Retriever]:
    """
    Chunk the corpus and build a BM25 index in memory.
    No embeddings generated, no ChromaDB involved.
    """
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

    retriever = BM25Retriever()
    retriever.index(chunks)
    console.print(f"  BM25 index built over {len(chunks)} chunks (no vector DB)")
    return chunks, retriever


# ── 2. QUERY ──────────────────────────────────────────────────────────────────
def run_query(
    question: str,
    retriever: BM25Retriever,
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
BASELINE_SCORES = {
    "faithfulness":      0.641,
    "answer_relevancy":  0.583,
    "answer_similarity": 0.538,
}


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
        "retrieval_strategy": "BM25Okapi (sparse keyword)",
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
    console.rule(f"[bold]Experiment 04 — Sparse Retrieval (BM25)")
    console.print(f"Config: chunk_size={cfg.chunking.chunk_size}, "
                  f"top_k={cfg.retrieval.top_k}, "
                  f"generator={cfg.generation.model}")
    console.print("[dim]Note: no embeddings at retrieval time — BM25 indexes chunk text directly[/dim]")

    # Embedder only needed for answer_similarity evaluation (deterministic metric)
    embedder = SentenceTransformerEmbedder(model_name=cfg.embedding.model_name)

    _, retriever = ingest()
    reranker = IdentityReranker()
    generator = OpenAIGenerator(model=cfg.generation.model, max_tokens=cfg.generation.max_tokens)

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
    console.rule("[bold green]Experiment 04 complete — compare against results/exp_01_baseline_*.json")


if __name__ == "__main__":
    main()
