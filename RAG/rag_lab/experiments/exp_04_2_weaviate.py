"""
Experiment 04-2 — Weaviate Backend (Dense + Sparse Storage, Dense-Only Retrieval)

Components used:
  - Ingestion:   load_directory (plain text files)             [baseline]
  - Chunking:    FixedSizeChunker (chunk_size=512, overlap=50) [baseline]
  - Embeddings:  SentenceTransformerEmbedder (all-MiniLM-L6-v2)[baseline]
  - VectorDB:    WeaviateVectorDB (stores BOTH dense + sparse)  [CHANGED]
  - Retrieval:   DenseRetriever (top_k=5, dense-only)          [baseline — UNCHANGED]
  - Reranking:   IdentityReranker (none)                       [baseline]
  - Generation:  OpenAIGenerator (gpt-4o-mini)                 [baseline]
  - Evaluation:  RAGAS (faithfulness + answer_relevancy + answer_similarity)

What this experiment tests:
  Backend swap to Weaviate. Each stored object contains TWO representations:
    1. Dense vector  — 384-dim MiniLM embedding, provided explicitly.
                       Weaviate stores it in an HNSW index. Searched via near_vector().
    2. Sparse vector — Weaviate builds a BM25 inverted index AUTOMATICALLY from
                       any TEXT property. By storing chunk text as a TEXT property,
                       the sparse index is created for free — no manual computation.

  Retrieval is UNCHANGED — still dense-only via DenseRetriever (near_vector).
  RAGAS scores should match baseline (delta ≈ 0).

Learning objective:
  - Understand that in Weaviate, you do NOT manually compute or serialize sparse
    vectors. You store text in a TEXT property and Weaviate handles BM25 indexing.
  - Dense vector: explicit (you provide it). Sparse index: implicit (Weaviate builds it).
  - Both live in the same Weaviate object and are both queryable — dense via
    near_vector(), sparse via bm25(), combined via hybrid().

Prerequisite — start Weaviate Docker container before running:
  docker run -d -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest

Run: uv run python experiments/exp_04_2_weaviate.py
"""

import json
import sys
import uuid
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
    DenseRetriever,
    IdentityReranker,
    OpenAIGenerator,
    EvalSample, evaluate_pipeline,
    EmbeddedChunk, RetrievedChunk,
)
from configs import BASELINE_CONFIG

# Weaviate-specific imports — self-contained to this experiment file
import weaviate
import weaviate.classes.config as wvc
from weaviate.classes.data import DataObject
from weaviate.classes.query import MetadataQuery

console = Console()
cfg = BASELINE_CONFIG

EMBEDDING_DIM = 384   # all-MiniLM-L6-v2 output dimension
COLLECTION    = "RagLabWeaviate"


# ── Weaviate Vector DB ────────────────────────────────────────────────────────
class WeaviateVectorDB:
    """
    Weaviate-backed vector store. Each stored object has TWO representations:

      1. Dense vector  — provided explicitly via vector=embedding on each DataObject.
                         Weaviate stores this in its HNSW index (same algorithm as Chroma).
                         Searched via near_vector().

      2. Sparse index  — Weaviate builds a BM25 inverted index AUTOMATICALLY from
                         every TEXT property. By writing chunk text into the 'text'
                         property, the sparse index is created for free — no manual
                         TF-IDF computation, no serialization, nothing to store yourself.

    This is how Weaviate natively holds both vector types in one object. The dense
    side is explicit (you provide it); the sparse side is implicit (Weaviate indexes it).
    To query the sparse index, use collection.query.bm25(). To combine both,
    use collection.query.hybrid(). This experiment uses only near_vector() (dense).

    Weaviate concepts:
      - Collection:         top-level container (like Chroma collection / SQL table)
      - vectorizer_config=none: no built-in embedding model; we provide vectors
      - DataObject:         the unit of storage — uuid + properties + vector
      - TEXT property:      auto-indexed for BM25 by Weaviate at insert time
      - near_vector():      ANN search over the HNSW dense index
      - insert_many():      batch insert, more efficient than one-by-one
    """

    def __init__(self, collection_name: str = COLLECTION, vector_size: int = EMBEDDING_DIM):
        try:
            self.client = weaviate.connect_to_local()
        except Exception as e:
            raise RuntimeError(
                "Cannot connect to Weaviate. Is the Docker container running?\n"
                "Start it with:\n"
                "  docker run -d -p 8080:8080 -p 50051:50051 "
                "cr.weaviate.io/semitechnologies/weaviate:latest"
            ) from e

        self.collection_name = collection_name
        self.vector_size = vector_size
        self._create_collection()

    def _create_collection(self) -> None:
        # Delete existing collection for a clean run (ignore_missing=True if absent)
        self.client.collections.delete(self.collection_name)

        self.collection = self.client.collections.create(
            name=self.collection_name,
            # none = no built-in vectorizer; we provide vectors ourselves
            vectorizer_config=wvc.Configure.Vectorizer.none(),
            properties=[
                # 'text' is a TEXT property → Weaviate auto-builds a BM25 inverted
                # index over it at insert time. No extra work needed for sparse storage.
                wvc.Property(name="text",        data_type=wvc.DataType.TEXT),
                wvc.Property(name="doc_id",      data_type=wvc.DataType.TEXT),
                wvc.Property(name="original_id", data_type=wvc.DataType.TEXT),
                wvc.Property(name="chunk_index", data_type=wvc.DataType.INT),
            ],
        )

    def add_chunks(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return
        objects = [
            DataObject(
                # uuid5 is deterministic: same chunk ID → same UUID across runs
                uuid=str(uuid.uuid5(uuid.NAMESPACE_DNS, c.id)),
                properties={
                    "text":        c.text,        # TEXT → Weaviate auto-builds BM25 index
                    "doc_id":      c.doc_id,
                    "original_id": c.id,
                    "chunk_index": c.chunk_index,
                },
                vector=c.embedding,   # dense vector → stored in HNSW index
            )
            for c in chunks
        ]
        result = self.collection.data.insert_many(objects)
        if result.errors:
            raise RuntimeError(f"Weaviate insert errors: {result.errors}")

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        # near_vector: ANN search using the HNSW dense index — retrieval unchanged
        results = self.collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )
        chunks = []
        for obj in results.objects:
            p = obj.properties
            # cosine distance = 1 - cosine_similarity → invert for score
            score = 1.0 - (obj.metadata.distance or 0.0)
            chunks.append(RetrievedChunk(
                id=p.get("original_id", str(obj.uuid)),
                text=p["text"],
                doc_id=p.get("doc_id", ""),
                chunk_index=int(p.get("chunk_index", 0)),
                score=score,
                metadata={},
            ))
        return chunks

    def count(self) -> int:
        return self.collection.aggregate.over_all(total_count=True).total_count

    def reset(self) -> None:
        self._create_collection()

    def close(self) -> None:
        self.client.close()


# ── 1. INGEST ─────────────────────────────────────────────────────────────────
def ingest(vectordb: WeaviateVectorDB, embedder: SentenceTransformerEmbedder) -> int:
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

    # Dense vectors — 384-dim MiniLM embeddings (explicit)
    embedded = embed_chunks(chunks, embedder)
    console.print(f"  {len(embedded)} dense vectors computed ({EMBEDDING_DIM}-dim)")

    # Insert — dense vector provided explicitly; Weaviate builds BM25 index from 'text' automatically
    vectordb.add_chunks(embedded)
    console.print(f"\n  {vectordb.count()} objects indexed in Weaviate")
    console.print(f"  [dim]Each object: dense vector (HNSW index) + BM25 inverted index (auto, from 'text' property)[/dim]")
    return len(chunks)


# ── 2. QUERY ──────────────────────────────────────────────────────────────────
def run_query(
    question: str,
    retriever: DenseRetriever,
    reranker: IdentityReranker,
    generator: OpenAIGenerator,
) -> tuple[str, list[str]]:
    # Retrieval is UNCHANGED — dense-only via near_vector; BM25 index exists but is not queried
    chunks = retriever.retrieve(question, top_k=cfg.retrieval.top_k)
    chunks = reranker.rerank(question, chunks)
    answer = generator.generate(question, chunks)
    return answer, [c.text for c in chunks]


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
    baseline_scores = {}
    if baseline_path and baseline_path.exists():
        baseline_scores = json.loads(baseline_path.read_text()).get("scores", {})

    qa_table = Table(title="Per-Question Results", show_lines=True)
    qa_table.add_column("Question", style="cyan", max_width=40)
    qa_table.add_column("Answer", style="white", max_width=60)
    qa_table.add_column("Chunks Retrieved", style="dim", justify="center")
    for s in samples:
        qa_table.add_row(s.question, s.answer, str(len(s.contexts)))
    console.print(qa_table)

    score_table = Table(title="RAGAS Scores — exp_04_2 vs Baseline", show_header=True)
    score_table.add_column("Metric", style="bold")
    score_table.add_column("Weaviate", justify="right")
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
        console.print("\n[dim]Expected: delta ≈ 0 — retrieval is unchanged (dense-only near_vector).[/dim]")
        console.print("[dim]BM25 index exists in Weaviate but is not used for retrieval here.[/dim]")


# ── 5. SAVE ───────────────────────────────────────────────────────────────────
def save_results(samples: list[EvalSample], scores: dict[str, float]) -> Path:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment":  "exp_04_2_weaviate",
        "timestamp":   timestamp,
        "config":      cfg.model_dump(),
        "vectordb":    "weaviate_local",
        "storage":     "dense_vector_hnsw_explicit + bm25_inverted_index_auto",
        "retrieval":   "dense_only_near_vector",
        "scores":      scores,
        "samples": [
            {
                "question":     s.question,
                "answer":       s.answer,
                "num_contexts": len(s.contexts),
                "ground_truth": s.ground_truth,
            }
            for s in samples
        ],
    }
    out_path = results_dir / f"exp_04_2_weaviate_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2))
    return out_path


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    console.rule("[bold]Experiment 04-2 — Weaviate (Dense + Sparse Storage)")
    console.print(
        f"Config: chunk_size={cfg.chunking.chunk_size}, "
        f"model={cfg.embedding.model_name}, "
        f"top_k={cfg.retrieval.top_k}, "
        f"vectordb=weaviate_local"
    )
    console.print("[dim]Stores dense + sparse vectors. Retrieval uses dense-only.[/dim]")

    vectordb = WeaviateVectorDB(collection_name=COLLECTION, vector_size=EMBEDDING_DIM)
    try:
        embedder  = SentenceTransformerEmbedder(model_name=cfg.embedding.model_name)
        retriever = DenseRetriever(embedder=embedder, vectordb=vectordb)
        reranker  = IdentityReranker()
        generator = OpenAIGenerator(model=cfg.generation.model, max_tokens=cfg.generation.max_tokens)

        console.print("\n[dim]Resetting Weaviate collection for clean run...[/dim]")
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

        results_dir = ROOT / "results"
        baseline_files = sorted(results_dir.glob("exp_01_baseline_*.json"), reverse=True)
        baseline_path  = baseline_files[0] if baseline_files else None

        display_results(samples, scores, baseline_path=baseline_path)

        out_path = save_results(samples, scores)
        console.print(f"\n[dim]Results saved to: {out_path}[/dim]")
        console.rule("[bold green]exp_04_2 complete")

    finally:
        # Always close the Weaviate connection, even if the experiment fails
        vectordb.close()


if __name__ == "__main__":
    main()
