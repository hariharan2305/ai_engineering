"""
Experiment 04-1 — Qdrant Backend (Dense + BM25 Sparse Storage, Dense-Only Retrieval)

Components used:
  - Ingestion:   load_directory (plain text files)             [baseline]
  - Chunking:    FixedSizeChunker (chunk_size=512, overlap=50) [baseline]
  - Embeddings:  SentenceTransformerEmbedder (all-MiniLM-L6-v2)[baseline]
  - VectorDB:    QdrantVectorDB (stores BOTH dense + sparse)   [CHANGED]
  - Retrieval:   DenseRetriever (top_k=5)                      [baseline — UNCHANGED]
  - Reranking:   IdentityReranker (none)                       [baseline]
  - Generation:  OpenAIGenerator (gpt-4o-mini)                 [baseline]
  - Evaluation:  RAGAS (faithfulness + answer_relevancy + answer_similarity)

What this experiment tests:
  Backend swap to Qdrant with both dense and BM25 sparse vectors stored per point.
  Retrieval is UNCHANGED — dense only. RAGAS scores should match baseline (delta ≈ 0).

  Unlike Weaviate (which builds BM25 automatically from TEXT properties),
  Qdrant requires you to:
    1. Compute BM25 sparse vectors yourself (using rank_bm25)
    2. Declare a sparse_vectors_config at collection creation
    3. Store them explicitly in each PointStruct alongside the dense vector

  This makes Qdrant's sparse support fully visible and hands-on.

Learning objective:
  - See how Qdrant stores named vectors: {"dense": VectorParams, "sparse": SparseVectorParams}
  - Understand BM25 sparse vector computation: TF × IDF per term, vocabulary indices
  - See what a sparse vector looks like: SparseVector(indices=[...], values=[...])
  - Contrast with Weaviate: same end result, opposite implementation responsibility

Run: uv run python experiments/exp_04_1_qdrant.py
"""

import json
import sys
import uuid
from collections import Counter
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from components import (
    load_directory,
    FixedSizeChunker, chunk_documents,
    SentenceTransformerEmbedder, embed_chunks,
    DenseRetriever,
    IdentityReranker,
    OpenAIGenerator,
    EvalSample, evaluate_pipeline,
    EmbeddedChunk, RetrievedChunk,
)
from configs import BASELINE_CONFIG

# Qdrant-specific imports — self-contained to this experiment file
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, PointStruct, SparseVector, SparseVectorParams, VectorParams,
)

console = Console()
cfg = BASELINE_CONFIG

# all-MiniLM-L6-v2 produces 384-dimensional vectors
EMBEDDING_DIM = 384


# ── BM25 Sparse Vector Computation ───────────────────────────────────────────
def compute_bm25_sparse_vectors(
    chunks: list[EmbeddedChunk],
) -> list[SparseVector]:
    """
    Compute BM25 sparse vectors for each chunk using rank_bm25.

    How it works:
      1. Fit BM25Okapi over the entire corpus — this computes:
           - IDF per term: log((N - df + 0.5) / (df + 0.5)) — rarer terms score higher
           - avgdl: average document length for length normalisation
      2. For each chunk, compute the BM25 weight for every term it contains:
           weight = IDF(t) × TF_norm(t, d)
           where TF_norm uses BM25's saturation formula to prevent long docs
           from dominating just because they repeat a term many times
      3. Build a SparseVector: only non-zero term weights are stored as
           (vocabulary_index, weight) pairs — this is what makes it "sparse"

    Contrast with Weaviate: Weaviate builds this inverted index automatically
    from TEXT properties. Qdrant gives you the infrastructure (SparseVectorParams)
    but the computation is entirely your responsibility.

    Returns a list of SparseVector objects — one per chunk, ready for Qdrant upsert.
    """
    tokenized = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    # Build a stable vocabulary index: term → integer position
    # bm25.idf contains every unique term fitted over the corpus
    vocab: dict[str, int] = {term: idx for idx, term in enumerate(bm25.idf)}

    sparse_vectors: list[SparseVector] = []
    for i, tokens in enumerate(tokenized):
        tf_counts = Counter(tokens)
        doc_len = len(tokens)

        indices: list[int] = []
        values: list[float] = []

        for term, tf in tf_counts.items():
            if term not in bm25.idf:
                continue
            # BM25 TF normalisation with k1 saturation and b length penalty
            tf_norm = (tf * (bm25.k1 + 1)) / (
                tf + bm25.k1 * (1 - bm25.b + bm25.b * doc_len / bm25.avgdl)
            )
            weight = float(bm25.idf[term] * tf_norm)
            if weight > 0:
                indices.append(vocab[term])
                values.append(weight)

        sparse_vectors.append(SparseVector(indices=indices, values=values))

    return sparse_vectors


# ── Qdrant Vector DB ──────────────────────────────────────────────────────────
class QdrantVectorDB:
    """
    In-memory Qdrant store with BOTH dense and BM25 sparse vectors per point.

    QdrantClient(":memory:") runs entirely in-process — no Docker required.

    Key difference from Weaviate:
      - Weaviate builds BM25 automatically from TEXT properties (zero config)
      - Qdrant requires explicit sparse vector config + manual computation:
          1. Declare sparse_vectors_config at collection creation
          2. Compute BM25 sparse vectors yourself (see compute_bm25_sparse_vectors)
          3. Store them explicitly in each PointStruct

    Named vector setup:
      - "dense"  → VectorParams(size=384, cosine) — stored in HNSW index
      - "sparse" → SparseVectorParams()           — stored in sparse inverted index

    Retrieval uses "dense" only (using="dense" in query_points).
    The sparse vector is stored but not queried in this experiment.
    """

    def __init__(self, collection_name: str = "rag_lab_qdrant", vector_size: int = 384):
        self.client = QdrantClient(":memory:")
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._create_collection()

    def _create_collection(self) -> None:
        self.client.create_collection(
            collection_name=self.collection_name,
            # Named dense vector — required when mixing with sparse
            vectors_config={
                "dense": VectorParams(size=self.vector_size, distance=Distance.COSINE),
            },
            # Sparse vector slot — Qdrant stores this in a separate inverted index
            # SparseVectorParams() has no dimension — sparse vectors are variable-length
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )

    def add_chunks(
        self,
        chunks: list[EmbeddedChunk],
        sparse_vectors: list[SparseVector],
    ) -> None:
        if not chunks:
            return
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, c.id)),
                # Both vectors stored under named keys on the same PointStruct
                vector={
                    "dense":  c.embedding,   # 384 floats — HNSW index
                    "sparse": sv,            # SparseVector(indices, values) — inverted index
                },
                payload={
                    "original_id": c.id,
                    "text":        c.text,
                    "doc_id":      c.doc_id,
                    "chunk_index": c.chunk_index,
                    **c.metadata,
                },
            )
            for c, sv in zip(chunks, sparse_vectors)
        ]
        self.client.upsert(collection_name=self.collection_name, wait=True, points=points)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            using="dense",   # explicitly target the dense named vector; sparse is stored but not queried
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        chunks = []
        for point in results.points:
            p = point.payload
            chunks.append(RetrievedChunk(
                id=p.get("original_id", str(point.id)),
                text=p["text"],
                doc_id=p.get("doc_id", ""),
                chunk_index=p.get("chunk_index", 0),
                score=point.score,  # cosine similarity (higher = more similar)
                metadata={k: v for k, v in p.items()
                          if k not in ("original_id", "text", "doc_id", "chunk_index")},
            ))
        return chunks

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name).count

    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self._create_collection()


# ── 1. INGEST ─────────────────────────────────────────────────────────────────
def ingest(vectordb: QdrantVectorDB, embedder: SentenceTransformerEmbedder) -> int:
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
    console.print(f"  {len(embedded)} dense vectors computed ({EMBEDDING_DIM}-dim)")

    # BM25 sparse vectors — computed manually over the whole corpus.
    # rank_bm25 fits IDF over all chunks first, then derives per-chunk term weights.
    sparse_vecs = compute_bm25_sparse_vectors(embedded)
    console.print(f"  {len(sparse_vecs)} BM25 sparse vectors computed")

    # Show what one sparse vector looks like
    sample = sparse_vecs[0]
    top5 = sorted(zip(sample.indices, sample.values), key=lambda x: x[1], reverse=True)[:5]
    console.print(f"  [dim]Sample sparse vector (chunk 0, top-5 terms by weight):[/dim]")
    console.print(f"  [dim]{[(i, round(v, 4)) for i, v in top5]}[/dim]")
    console.print(f"  [dim]→ {len(sample.indices)} non-zero terms out of full vocabulary[/dim]")

    vectordb.add_chunks(embedded, sparse_vecs)
    console.print(f"  {vectordb.count()} points indexed in Qdrant (dense + sparse per point)")
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
def display_results(
    samples: list[EvalSample],
    scores: dict[str, float],
    baseline_path: Path | None = None,
) -> None:
    # Load baseline scores for delta comparison
    baseline_scores = {}
    if baseline_path and baseline_path.exists():
        baseline_data = json.loads(baseline_path.read_text())
        baseline_scores = baseline_data.get("scores", {})

    qa_table = Table(title="Per-Question Results", show_lines=True)
    qa_table.add_column("Question", style="cyan", max_width=40)
    qa_table.add_column("Answer", style="white", max_width=60)
    qa_table.add_column("Chunks Retrieved", style="dim", justify="center")

    for s in samples:
        qa_table.add_row(s.question, s.answer, str(len(s.contexts)))
    console.print(qa_table)

    score_table = Table(title="RAGAS Scores — exp_04_1 vs Baseline", show_header=True)
    score_table.add_column("Metric", style="bold")
    score_table.add_column("Qdrant", justify="right")
    score_table.add_column("Baseline", justify="right")
    score_table.add_column("Delta", justify="right")
    score_table.add_column("Interpretation")

    interpretations = {
        "faithfulness":      "[LLM judge]   Is the answer supported by context?",
        "answer_relevancy":  "[LLM judge]   Does the answer address the question?",
        "answer_similarity": "[Deterministic] Cosine sim vs ground truth",
    }
    for metric, score in scores.items():
        baseline = baseline_scores.get(metric)
        delta_str = ""
        if baseline is not None:
            delta = score - baseline
            delta_color = "green" if delta >= 0 else "red"
            delta_str = f"[{delta_color}]{delta:+.3f}[/{delta_color}]"
        color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
        score_table.add_row(
            metric,
            f"[{color}]{score:.3f}[/{color}]",
            f"{baseline:.3f}" if baseline is not None else "n/a",
            delta_str,
            interpretations.get(metric, ""),
        )
    console.print(score_table)

    if baseline_scores:
        console.print("\n[dim]Expected: delta ≈ 0 — retrieval is dense-only (BM25 sparse stored but not queried).[/dim]")


# ── 5. SAVE ───────────────────────────────────────────────────────────────────
def save_results(samples: list[EvalSample], scores: dict[str, float]) -> Path:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "exp_04_1_qdrant",
        "timestamp": timestamp,
        "config": cfg.model_dump(),
        "vectordb":  "qdrant_in_memory",
        "storage":   "dense_hnsw + bm25_sparse_inverted_index",
        "retrieval": "dense_only",
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

    out_path = results_dir / f"exp_04_1_qdrant_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2))
    return out_path


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    console.rule("[bold]Experiment 04-1 — Qdrant Backend (Dense-Only, In-Memory)")
    console.print(
        f"Config: chunk_size={cfg.chunking.chunk_size}, "
        f"model={cfg.embedding.model_name}, "
        f"top_k={cfg.retrieval.top_k}, "
        f"vectordb=qdrant_in_memory"
    )
    console.print("[dim]Only the vector DB backend changes vs baseline. Expecting delta ≈ 0.[/dim]")

    # Init components — everything identical to baseline except the vector DB
    embedder = SentenceTransformerEmbedder(model_name=cfg.embedding.model_name)
    vectordb = QdrantVectorDB(
        collection_name="rag_lab_qdrant",
        vector_size=EMBEDDING_DIM,
    )
    retriever = DenseRetriever(embedder=embedder, vectordb=vectordb)
    reranker = IdentityReranker()
    generator = OpenAIGenerator(model=cfg.generation.model, max_tokens=cfg.generation.max_tokens)

    console.print("\n[dim]Resetting Qdrant collection for clean run...[/dim]")
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

    # Find the most recent baseline result for delta comparison
    results_dir = ROOT / "results"
    baseline_files = sorted(results_dir.glob("exp_01_baseline_*.json"), reverse=True)
    baseline_path = baseline_files[0] if baseline_files else None

    display_results(samples, scores, baseline_path=baseline_path)

    out_path = save_results(samples, scores)
    console.print(f"\n[dim]Results saved to: {out_path}[/dim]")
    console.rule("[bold green]exp_04_1 complete")


if __name__ == "__main__":
    main()
