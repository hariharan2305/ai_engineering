"""
Experiment 03-2 — Embedding Model: intfloat/e5-large-v2 (asymmetric)

What changed vs baseline:
  - Embeddings: all-MiniLM-L6-v2 (384-dim, symmetric)
              → intfloat/e5-large-v2 (1024-dim, asymmetric with prefixes)

Everything else is identical to the baseline:
  - Chunking:    FixedSizeChunker (chunk_size=512, overlap=50)
  - VectorDB:    ChromaVectorDB (cosine similarity, local)
  - Retrieval:   DenseRetriever (top_k=5)
  - Reranking:   IdentityReranker (none)
  - Generation:  OpenAIGenerator (gpt-4o-mini)
  - Evaluation:  RAGAS (faithfulness + answer_relevancy + answer_similarity)

Why E5 (intfloat/e5-large-v2):
  - Asymmetric model — queries and passages are encoded differently
  - Passages are prefixed with "passage: " during indexing
  - Queries are prefixed with "query: " during retrieval
  - Omitting these prefixes silently degrades quality (model was trained expecting them)
  - 1024-dim vectors — larger representational space than MiniLM (384) or mpnet (768)
  - Strong MTEB performance, especially on retrieval benchmarks

Asymmetric encoding design:
  - E5Embedder.embed(texts)     → adds "passage: " prefix (used during chunk indexing)
  - E5Embedder.embed_as_query() → adds "query: "   prefix (used during retrieval)
  - E5DenseRetriever overrides retrieve() to call embed_as_query() instead of embed()
  - The shared components (embeddings.py, retrieval.py) are NOT modified

Hypothesis:
  - Asymmetric encoding + 1024-dim should further improve retrieval quality
  - answer_similarity should move vs 03-1 (mpnet) as the correct semantic distinction
    between "what a passage is about" vs "what a user is asking" is now encoded

Run: uv run python experiments/exp_03_2_e5.py
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
    IdentityReranker,
    OpenAIGenerator,
    EvalSample, evaluate_pipeline,
)
from components.base import RetrievedChunk
from components.vectordb import ChromaVectorDB
from configs import BASELINE_CONFIG

console = Console()
cfg = BASELINE_CONFIG

EXPERIMENT_MODEL = "intfloat/e5-large-v2"
EXPERIMENT_NAME  = "exp_03_2_e5"
COLLECTION_NAME  = "rag_lab_exp03_2"

BASELINE_SCORES = {
    "faithfulness":      0.641,
    "answer_relevancy":  0.583,
    "answer_similarity": 0.538,
}


# ── Custom asymmetric embedder ────────────────────────────────────────────────
class E5Embedder(SentenceTransformerEmbedder):
    """
    E5 requires separate prefixes for queries and passages.
    Under the hood it is still a SentenceTransformer model — the prefix
    is just a string prepended to the input text before calling .encode().

    - Passages (indexing): "passage: " + text
    - Queries (retrieval): "query: "   + text

    The base class embed() method is overridden to add the passage prefix.
    A separate embed_as_query() method handles the query prefix.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Passage encoding — used by embed_chunks() during indexing
        prefixed = ["passage: " + t for t in texts]
        return self.model.encode(prefixed, show_progress_bar=False).tolist()

    def embed_as_query(self, query: str) -> list[float]:
        # Query encoding — called by E5DenseRetriever during retrieval
        prefixed = "query: " + query
        return self.model.encode([prefixed], show_progress_bar=False).tolist()[0]


# ── Custom retriever that uses embed_as_query ─────────────────────────────────
class E5DenseRetriever:
    """
    Same logic as DenseRetriever but calls embed_as_query() so that the
    query gets the correct "query: " prefix — not the passage prefix.
    """

    def __init__(self, embedder: E5Embedder, vectordb: ChromaVectorDB):
        self.embedder = embedder
        self.vectordb = vectordb

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed_as_query(query)
        return self.vectordb.search(query_embedding, top_k=top_k)


# ── 1. INGEST ─────────────────────────────────────────────────────────────────
def ingest(vectordb: ChromaVectorDB, embedder: E5Embedder) -> int:
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

    embedded = embed_chunks(chunks, embedder)  # embed() called here → "passage: " prefix
    vectordb.add_chunks(embedded)
    console.print(f"  {vectordb.count()} chunks indexed in ChromaDB")
    return len(chunks)


# ── 2. QUERY ──────────────────────────────────────────────────────────────────
def run_query(
    question: str,
    retriever: E5DenseRetriever,
    reranker: IdentityReranker,
    generator: OpenAIGenerator,
) -> tuple[str, list[str]]:
    chunks = retriever.retrieve(question, top_k=cfg.retrieval.top_k)  # embed_as_query() called here
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
        "config": {
            **cfg.model_dump(),
            "embedding": {"model_name": EXPERIMENT_MODEL, "mode": "asymmetric"},
        },
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
    console.rule(f"[bold]Experiment 03-2 — Embedding: {EXPERIMENT_MODEL} (asymmetric)")
    console.print(
        f"Swapping: [red]all-MiniLM-L6-v2 (384-dim, symmetric)[/] → [green]{EXPERIMENT_MODEL} (1024-dim, asymmetric)[/]\n"
        f"Prefix:   passages = 'passage: <text>'  |  queries = 'query: <text>'\n"
        f"Fixed:    chunk_size={cfg.chunking.chunk_size}, top_k={cfg.retrieval.top_k}, generator={cfg.generation.model}"
    )

    embedder  = E5Embedder(model_name=EXPERIMENT_MODEL)
    vectordb  = ChromaVectorDB(
        collection_name=COLLECTION_NAME,
        persist_dir=str(ROOT / cfg.vectordb.persist_dir),
    )
    retriever = E5DenseRetriever(embedder=embedder, vectordb=vectordb)
    reranker  = IdentityReranker()
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
    console.rule(f"[bold green]exp_03_2 complete — check delta vs baseline above")


if __name__ == "__main__":
    main()
