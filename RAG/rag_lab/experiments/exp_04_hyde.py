"""
Experiment 04-HyDE — Hypothetical Document Embeddings

What changed vs baseline:
  - Retrieval: DenseRetriever (embeds the question, searches by question vector)
             → HyDERetriever (LLM generates a hypothetical answer, embeds that instead)

Everything else is identical to the baseline:
  - Chunking:    FixedSizeChunker (chunk_size=512, overlap=50)
  - Embeddings:  SentenceTransformerEmbedder (all-MiniLM-L6-v2)
  - VectorDB:    ChromaVectorDB (cosine similarity, local)
  - Reranking:   IdentityReranker (none)
  - Generation:  OpenAIGenerator (gpt-4o-mini)
  - Evaluation:  RAGAS (faithfulness + answer_relevancy + answer_similarity)

The representation alignment problem HyDE solves:
  Standard dense retrieval embeds the question and finds chunks similar to it.
  But questions and answers are representationally asymmetric — they live in
  different regions of embedding space even when they're about the same concept.

  When we index documents, we are indexing ground-truth answers.
  When a user queries, they send a question — which by definition does not know
  the answer, only the keywords or intent behind it.

  HyDE's fix: generate an answer-shaped text first, embed that, search for real
  chunks similar to it. Answer-to-answer similarity is a much tighter match than
  question-to-answer similarity. The hypothetical answer doesn't need to be
  factually correct — it just needs the right vocabulary and shape to point the
  embedding vector into the correct neighborhood of the corpus.

Cost per query:
  1 LLM call (generate hypothetical answer) + 1 embedding + 1 vector search.
  Same total calls as baseline retrieval + 1 LLM call overhead.

Hypothesis:
  - answer_similarity should improve — the hypothetical answer bridges the
    question-to-document representation gap.
  - Improvement depends on how well gpt-4o-mini can generate plausible
    answer-shaped text for our RAG/vector-DB domain.
  - Questions where the user phrasing diverges most from corpus vocabulary
    will show the biggest per-question gains.

Run: uv run python experiments/exp_04_hyde.py
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
    HyDERetriever,
    IdentityReranker,
    OpenAIGenerator,
    EvalSample, evaluate_pipeline,
)
from configs import BASELINE_CONFIG

console = Console()
cfg = BASELINE_CONFIG
EXPERIMENT_NAME = "exp_04_hyde"
COLLECTION_NAME = "rag_lab_exp04_hyde"

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
    retriever: HyDERetriever,
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
        "retrieval_strategy": "HyDE — LLM generates hypothetical answer, embeds that instead of query",
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
    console.rule("[bold]Experiment 04-HyDE — Hypothetical Document Embeddings")
    console.print(
        "Swapping: [red]DenseRetriever (embeds question)[/] → "
        "[green]HyDERetriever (LLM generates hypothetical answer, embeds that)[/]\n"
        f"Everything else: chunk_size={cfg.chunking.chunk_size}, "
        f"top_k={cfg.retrieval.top_k}, generator={cfg.generation.model}"
    )
    console.print("[dim]Each query: 1 LLM call (hypothetical answer) + 1 embed + 1 vector search[/dim]")

    embedder = SentenceTransformerEmbedder(model_name=cfg.embedding.model_name)
    vectordb = ChromaVectorDB(
        collection_name=COLLECTION_NAME,
        persist_dir=str(ROOT / cfg.vectordb.persist_dir),
    )
    dense_retriever = DenseRetriever(embedder=embedder, vectordb=vectordb)
    hyde_retriever = HyDERetriever(
        dense_retriever=dense_retriever,
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
        answer, contexts = run_query(item["question"], hyde_retriever, reranker, generator)
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
    console.rule("[bold green]Experiment 04-HyDE complete — check delta vs baseline above")


if __name__ == "__main__":
    main()
