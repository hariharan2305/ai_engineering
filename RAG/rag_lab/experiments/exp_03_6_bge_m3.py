"""
Experiment 03-6 — Embedding Model: BAAI/bge-m3 (dense-only)

What changed vs baseline:
  - Embeddings: all-MiniLM-L6-v2 (384-dim, English-only, symmetric)
              → BAAI/bge-m3 (1024-dim, multilingual, dense retrieval only)

Everything else is identical to the baseline:
  - Chunking:    FixedSizeChunker (chunk_size=512, overlap=50)
  - VectorDB:    ChromaVectorDB (cosine similarity, local)
  - Retrieval:   DenseRetriever (top_k=5)
  - Reranking:   IdentityReranker (none)
  - Generation:  OpenAIGenerator (gpt-4o-mini)
  - Evaluation:  RAGAS (faithfulness + answer_relevancy + answer_similarity)

Why BGE-M3 (dense only):
  - BGE-M3 is a multi-functional model — it can produce THREE types of vectors:
      1. dense_vecs    — standard dense embedding (1024-dim) — used in this experiment
      2. lexical_weights — SPLADE-style learned sparse vectors (for hybrid retrieval)
      3. colbert_vecs  — multi-vector (ColBERT) for late interaction scoring
  - In this experiment, we isolate the DENSE component only to measure its quality
  - The sparse+colbert capabilities will be explored in the Retrieval track (exp_04_*)
  - 1024-dim output, 8192-token context window, multilingual (100+ languages)
  - No instruction prefixes for dense retrieval — symmetric usage

Dense-only API (from official FlagEmbedding docs):
  - BGEM3FlagModel.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
  - Returns dict — access dense vectors via output['dense_vecs']
  - use_fp16=True speeds up inference with minimal quality loss

What's NOT being tested here (saved for Retrieval track):
  - Hybrid dense+sparse retrieval using both dense_vecs and lexical_weights
  - SPLADE-style learned sparse weights from BGE-M3
  - ColBERT multi-vector late interaction scoring

Run: uv run python experiments/exp_03_6_bge_m3.py
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

from FlagEmbedding import BGEM3FlagModel
from components import (
    load_directory,
    FixedSizeChunker, chunk_documents,
    ChromaVectorDB,
    IdentityReranker,
    OpenAIGenerator,
    EvalSample, evaluate_pipeline,
)
from components.base import Chunk, EmbeddedChunk, RetrievedChunk
from components.vectordb import ChromaVectorDB
from configs import BASELINE_CONFIG

console = Console()
cfg = BASELINE_CONFIG

EXPERIMENT_MODEL = "BAAI/bge-m3"
EXPERIMENT_NAME  = "exp_03_6_bge_m3"
COLLECTION_NAME  = "rag_lab_exp03_6"

BASELINE_SCORES = {
    "faithfulness":      0.641,
    "answer_relevancy":  0.583,
    "answer_similarity": 0.538,
}


# ── BGE-M3 dense embedder ─────────────────────────────────────────────────────
class BGEM3DenseEmbedder:
    """
    BGE-M3 in dense-only mode via FlagEmbedding.

    BGEM3FlagModel.encode() returns a dict with three keys:
      - 'dense_vecs'      — 1024-dim float array (what we use here)
      - 'lexical_weights' — sparse SPLADE-style weights (not used in this experiment)
      - 'colbert_vecs'    — multi-vector ColBERT representations (not used here)

    For dense retrieval, BGE-M3 is symmetric — no separate query instruction needed.
    The dense vectors work well without any prefix.

    use_fp16=True: halves memory usage with negligible quality impact on modern hardware.
    max_length=8192: BGE-M3 supports up to 8192 tokens per input.
    """

    def __init__(self, model_name: str = EXPERIMENT_MODEL):
        self.model_name = model_name
        console.print(f"  [dim]Loading {model_name}...[/dim]")
        self.model = BGEM3FlagModel(model_name, use_fp16=True)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Dense-only encoding — return_sparse=False, return_colbert_vecs=False
        output = self.model.encode(
            texts,
            batch_size=12,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        # output['dense_vecs'] is a numpy array of shape (N, 1024)
        return output['dense_vecs'].tolist()

    def embed_as_query(self, query: str) -> list[float]:
        # Symmetric for dense — same encoding path as passages
        return self.embed([query])[0]

    @property
    def dimension(self) -> int:
        return 1024


def embed_chunks_m3(chunks: list[Chunk], embedder: BGEM3DenseEmbedder) -> list[EmbeddedChunk]:
    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)
    return [
        EmbeddedChunk(
            id=chunk.id,
            text=chunk.text,
            doc_id=chunk.doc_id,
            chunk_index=chunk.chunk_index,
            embedding=emb,
            metadata=chunk.metadata,
        )
        for chunk, emb in zip(chunks, embeddings)
    ]


# ── Custom retriever ──────────────────────────────────────────────────────────
class BGEM3DenseRetriever:
    def __init__(self, embedder: BGEM3DenseEmbedder, vectordb: ChromaVectorDB):
        self.embedder = embedder
        self.vectordb = vectordb

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed_as_query(query)
        return self.vectordb.search(query_embedding, top_k=top_k)


# ── 1. INGEST ─────────────────────────────────────────────────────────────────
def ingest(vectordb: ChromaVectorDB, embedder: BGEM3DenseEmbedder) -> int:
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

    embedded = embed_chunks_m3(chunks, embedder)
    vectordb.add_chunks(embedded)
    console.print(f"  {vectordb.count()} chunks indexed in ChromaDB")
    return len(chunks)


# ── 2. QUERY ──────────────────────────────────────────────────────────────────
def run_query(
    question: str,
    retriever: BGEM3DenseRetriever,
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
                "mode": "dense_only",
                "note": "sparse and colbert vectors available but not used — saved for retrieval track",
                "dimensions": 1024,
                "context_tokens": 8192,
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
    console.rule(f"[bold]Experiment 03-6 — Embedding: {EXPERIMENT_MODEL} (dense only)")
    console.print(
        f"Swapping: [red]all-MiniLM-L6-v2 (384-dim, English-only)[/] → [green]{EXPERIMENT_MODEL} (1024-dim, multilingual)[/]\n"
        f"Mode:     dense vectors only — sparse+colbert capabilities intentionally unused\n"
        f"Note:     sparse+colbert will be unlocked in Retrieval track (exp_04_*)\n"
        f"Fixed:    chunk_size={cfg.chunking.chunk_size}, top_k={cfg.retrieval.top_k}, generator={cfg.generation.model}"
    )

    embedder  = BGEM3DenseEmbedder(model_name=EXPERIMENT_MODEL)
    vectordb  = ChromaVectorDB(
        collection_name=COLLECTION_NAME,
        persist_dir=str(ROOT / cfg.vectordb.persist_dir),
    )
    retriever = BGEM3DenseRetriever(embedder=embedder, vectordb=vectordb)
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
    console.rule(f"[bold green]exp_03_6 complete — check delta vs baseline above")


if __name__ == "__main__":
    main()
