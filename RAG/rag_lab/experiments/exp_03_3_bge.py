"""
Experiment 03-3 — Embedding Model: BAAI/bge-large-en-v1.5 (asymmetric, query instruction)

What changed vs baseline:
  - Embeddings: all-MiniLM-L6-v2 (384-dim, symmetric)
              → BAAI/bge-large-en-v1.5 (1024-dim, asymmetric with query instruction)

Everything else is identical to the baseline:
  - Chunking:    FixedSizeChunker (chunk_size=512, overlap=50)
  - VectorDB:    ChromaVectorDB (cosine similarity, local)
  - Retrieval:   DenseRetriever (top_k=5)
  - Reranking:   IdentityReranker (none)
  - Generation:  OpenAIGenerator (gpt-4o-mini)
  - Evaluation:  RAGAS (faithfulness + answer_relevancy + answer_similarity)

Why BGE-large-en-v1.5:
  - From BAAI (Beijing Academy of AI) — top performer on MTEB English leaderboard
  - 1024-dim vectors — same size as E5-large
  - Asymmetric but different from E5: only queries get a prefix, passages do NOT
  - Query instruction: "Represent this sentence for searching relevant passages: "
  - Without this query instruction, retrieval quality degrades (model trained expecting it)
  - Good baseline for comparing instruction-only-on-query vs E5's both-side prefixes

BGE vs E5 prefix comparison:
  - E5:  query="query: <text>"      passage="passage: <text>"  (both sides prefixed)
  - BGE: query="Represent this sentence for searching relevant passages: <text>"
         passage=<text>             (only query prefixed)

Asymmetric encoding design:
  - BGEEmbedder.embed(texts)      → no prefix (used during chunk indexing)
  - BGEEmbedder.embed_as_query()  → adds instruction prefix (used during retrieval)
  - BGEDenseRetriever overrides retrieve() to call embed_as_query()
  - The shared components (embeddings.py, retrieval.py) are NOT modified

Hypothesis:
  - BGE is a stronger general-purpose retrieval model than E5 on English text
  - Single-sided prefix (query only) is sufficient — passages don't need explicit labeling
  - Compare this result against exp_03_2 (E5) to see which prefix strategy wins

Run: uv run python experiments/exp_03_3_bge.py
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

EXPERIMENT_MODEL  = "BAAI/bge-large-en-v1.5"
EXPERIMENT_NAME   = "exp_03_3_bge"
COLLECTION_NAME   = "rag_lab_exp03_3"
BGE_QUERY_PREFIX  = "Represent this sentence for searching relevant passages: "

BASELINE_SCORES = {
    "faithfulness":      0.641,
    "answer_relevancy":  0.583,
    "answer_similarity": 0.538,
}


# ── Custom asymmetric embedder ────────────────────────────────────────────────
class BGEEmbedder(SentenceTransformerEmbedder):
    """
    BGE uses an instruction prefix for queries only — passages are encoded as-is.

    This is different from E5 where both sides get a prefix.

    - Passages (indexing): text (no prefix)
    - Queries (retrieval): "Represent this sentence for searching relevant passages: " + text

    embed() → passage encoding (no prefix), called by embed_chunks() during indexing
    embed_as_query() → query encoding (with prefix), called by BGEDenseRetriever
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Passage encoding — no prefix for BGE passages
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_as_query(self, query: str) -> list[float]:
        # Query encoding — prepend the instruction prefix
        prefixed = BGE_QUERY_PREFIX + query
        return self.model.encode([prefixed], show_progress_bar=False).tolist()[0]


# ── Custom retriever that uses embed_as_query ─────────────────────────────────
class BGEDenseRetriever:
    """
    Same logic as DenseRetriever but calls embed_as_query() so the query
    gets the BGE instruction prefix — not the plain passage encoding.
    """

    def __init__(self, embedder: BGEEmbedder, vectordb: ChromaVectorDB):
        self.embedder = embedder
        self.vectordb = vectordb

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed_as_query(query)
        return self.vectordb.search(query_embedding, top_k=top_k)


# ── 1. INGEST ─────────────────────────────────────────────────────────────────
def ingest(vectordb: ChromaVectorDB, embedder: BGEEmbedder) -> int:
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

    embedded = embed_chunks(chunks, embedder)  # embed() called here → no prefix
    vectordb.add_chunks(embedded)
    console.print(f"  {vectordb.count()} chunks indexed in ChromaDB")
    return len(chunks)


# ── 2. QUERY ──────────────────────────────────────────────────────────────────
def run_query(
    question: str,
    retriever: BGEDenseRetriever,
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
            "embedding": {
                "model_name": EXPERIMENT_MODEL,
                "mode": "asymmetric",
                "query_prefix": BGE_QUERY_PREFIX,
                "passage_prefix": None,
            },
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
    console.rule(f"[bold]Experiment 03-3 — Embedding: {EXPERIMENT_MODEL} (asymmetric, query instruction)")
    console.print(
        f"Swapping: [red]all-MiniLM-L6-v2 (384-dim, symmetric)[/] → [green]{EXPERIMENT_MODEL} (1024-dim, asymmetric)[/]\n"
        f"Prefix:   passages = none  |  queries = '{BGE_QUERY_PREFIX[:50]}...'\n"
        f"Fixed:    chunk_size={cfg.chunking.chunk_size}, top_k={cfg.retrieval.top_k}, generator={cfg.generation.model}"
    )

    embedder  = BGEEmbedder(model_name=EXPERIMENT_MODEL)
    vectordb  = ChromaVectorDB(
        collection_name=COLLECTION_NAME,
        persist_dir=str(ROOT / cfg.vectordb.persist_dir),
    )
    retriever = BGEDenseRetriever(embedder=embedder, vectordb=vectordb)
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
    console.rule(f"[bold green]exp_03_3 complete — check delta vs baseline above")


if __name__ == "__main__":
    main()
