"""
Experiment 04-3 — pgvector Backend (Dense-Only, PostgreSQL + SQLAlchemy ORM)

Components used:
  - Ingestion:   load_directory (plain text files)             [baseline]
  - Chunking:    FixedSizeChunker (chunk_size=512, overlap=50) [baseline]
  - Embeddings:  SentenceTransformerEmbedder (all-MiniLM-L6-v2)[baseline]
  - VectorDB:    PgVectorDB (PostgreSQL + pgvector + SQLAlchemy)[CHANGED]
  - Retrieval:   DenseRetriever (top_k=5, dense-only)          [baseline — UNCHANGED]
  - Reranking:   IdentityReranker (none)                       [baseline]
  - Generation:  OpenAIGenerator (gpt-4o-mini)                 [baseline]
  - Evaluation:  RAGAS (faithfulness + answer_relevancy + answer_similarity)

What this experiment tests:
  Backend swap to pgvector. The key architectural difference vs Qdrant/Weaviate:
    - pgvector is a PostgreSQL EXTENSION, not a purpose-built vector DB
    - Table and HNSW index are two SEPARATE constructs:
        Base.metadata.create_all()  →  creates the table (raw row storage)
        Index(...).create()         →  adds HNSW on top (query accelerator)
      Purpose-built DBs (Qdrant, Weaviate) combine both into "create collection".
    - Vector is just another ORM column: mapped_column(Vector(384))
    - ANN search uses pgvector's column methods: .cosine_distance(query_vec)
    - Full SQL power alongside vectors: filter(), join(), group_by(), etc.

  Retrieval is UNCHANGED — still dense-only. RAGAS scores should match baseline.

Learning objective:
  - See how SQLAlchemy ORM makes pgvector feel similar to Qdrant/Weaviate
  - Understand the table + index separation (vs baked-in collection approach)
  - See HNSW parameters (m, ef_construction, ef_search) surfaced at index level
  - Understand pgvector's fit: existing Postgres stack, need relational queries
    alongside vectors, ACID transactions, no separate infra to manage

Prerequisite — start pgvector Docker container before running:
  docker run -d -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg17

Run: uv run python experiments/exp_04_3_pgvector.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# SQLAlchemy ORM imports
from sqlalchemy import Index, Integer, Text, create_engine, func, select, text
from sqlalchemy.orm import DeclarativeBase, Session, mapped_column

# pgvector SQLAlchemy integration — Vector column type + distance methods
from pgvector.sqlalchemy import Vector

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

console = Console()
cfg = BASELINE_CONFIG

# ── Config ────────────────────────────────────────────────────────────────────
PG_DSN        = "postgresql://postgres:postgres@localhost:5432/postgres"
TABLE_NAME    = "rag_chunks_exp04"
EMBEDDING_DIM = 384   # all-MiniLM-L6-v2 output dimension

# HNSW tuning — same parameters that Qdrant/Weaviate set internally at collection
# creation. pgvector exposes them explicitly via the Index definition.
HNSW_M                = 16   # edges per node per layer — higher = better recall, more memory
HNSW_EF_CONSTRUCTION  = 64   # candidate list at build time — higher = better graph quality
HNSW_EF_SEARCH        = 40   # candidate list at query time — higher = better recall, slower


# ── ORM Model ─────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


class RagChunk(Base):
    """
    SQLAlchemy model for a chunk with its dense embedding.

    Vector(384): pgvector column type provided by pgvector-python.
    Behaves like any other SQLAlchemy column — readable, filterable, sortable.
    Exposes distance methods directly on the column:
      .cosine_distance(vec)    →  equivalent to SQL: embedding <=> vec
      .l2_distance(vec)        →  equivalent to SQL: embedding <-> vec
      .max_inner_product(vec)  →  equivalent to SQL: embedding <#> vec
    """
    __tablename__ = TABLE_NAME

    id          = mapped_column(Text,    primary_key=True)
    text        = mapped_column(Text,    nullable=False)
    doc_id      = mapped_column(Text)
    chunk_index = mapped_column(Integer)
    embedding   = mapped_column(Vector(EMBEDDING_DIM))


# HNSW index defined alongside the model — created separately from the table.
# This is the pgvector equivalent of Qdrant's VectorParams or Weaviate's
# vectorizer config, but explicit rather than implicit.
# postgresql_ops tells the index which distance operator to optimise for.
hnsw_index = Index(
    f"{TABLE_NAME}_hnsw_idx",
    RagChunk.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)


# ── pgvector DB ───────────────────────────────────────────────────────────────
class PgVectorDB:
    """
    PostgreSQL + pgvector via SQLAlchemy ORM.

    Compared to the raw psycopg2 approach:
      - No manual SQL strings for insert/query
      - No cursor management or transaction boilerplate
      - ORM objects (RagChunk) instead of raw tuples
      - Distance methods on the column: .cosine_distance() instead of <=>
      - Connection pooling handled by SQLAlchemy's engine automatically

    Compared to Qdrant / Weaviate:
      - Table (Base.metadata.create_all) and index (hnsw_index.create) are
        two explicit steps instead of one "create_collection" call
      - Same Python-object style for inserts (session.add_all) and queries
        (session.scalars / session.execute)
    """

    def __init__(self):
        try:
            self.engine = create_engine(PG_DSN)
            # Verify connection is reachable before proceeding
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            raise RuntimeError(
                "Cannot connect to PostgreSQL. Is the Docker container running?\n"
                "  docker run -d -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg17"
            ) from e

        self._setup()

    def _setup(self) -> None:
        with self.engine.begin() as conn:
            # pgvector extension must exist before the Vector column type is usable
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Create the table from the ORM model (idempotent — checkfirst=True)
        Base.metadata.create_all(self.engine)

        # Create the HNSW index separately (same separation as CREATE TABLE vs CREATE INDEX in SQL)
        with self.engine.begin() as conn:
            hnsw_index.create(conn, checkfirst=True)

    def add_chunks(self, chunks: list[EmbeddedChunk]) -> None:
        """
        Batch-insert chunks via ORM. session.add_all() + commit() sends
        all rows in a single transaction — atomic and efficient.
        """
        if not chunks:
            return
        with Session(self.engine) as session:
            session.add_all([
                RagChunk(
                    id=c.id,
                    text=c.text,
                    doc_id=c.doc_id,
                    chunk_index=c.chunk_index,
                    # float32 is pgvector's native precision
                    embedding=np.array(c.embedding, dtype=np.float32),
                )
                for c in chunks
            ])
            session.commit()

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        """
        ANN search via .cosine_distance() column method.

        cosine_distance ∈ [0, 2] for unit vectors (0 = identical).
        1 - distance converts to cosine similarity ∈ [-1, 1], practically [0, 1]
        for sentence-transformer embeddings (which are always positive-cosine similar).
        """
        q_vec = np.array(query_embedding, dtype=np.float32)

        with Session(self.engine) as session:
            # SET hnsw.ef_search controls recall vs speed for this session
            session.execute(text(f"SET hnsw.ef_search = {HNSW_EF_SEARCH}"))

            rows = session.execute(
                select(
                    RagChunk,
                    (1 - RagChunk.embedding.cosine_distance(q_vec)).label("score"),
                )
                .order_by(RagChunk.embedding.cosine_distance(q_vec))
                .limit(top_k)
            ).all()

        return [
            RetrievedChunk(
                id=row.RagChunk.id,
                text=row.RagChunk.text,
                doc_id=row.RagChunk.doc_id or "",
                chunk_index=row.RagChunk.chunk_index or 0,
                score=float(row.score),
                metadata={},
            )
            for row in rows
        ]

    def count(self) -> int:
        with Session(self.engine) as session:
            return session.scalar(select(func.count()).select_from(RagChunk))

    def reset(self) -> None:
        """Drop and recreate the table + HNSW index for a clean run."""
        Base.metadata.drop_all(self.engine)
        self._setup()

    def close(self) -> None:
        self.engine.dispose()


# ── 1. INGEST ─────────────────────────────────────────────────────────────────
def ingest(vectordb: PgVectorDB, embedder: SentenceTransformerEmbedder) -> int:
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
    console.print(f"  {vectordb.count()} rows inserted into PostgreSQL")
    console.print(
        f"  [dim]Table: {TABLE_NAME} | "
        f"HNSW index: m={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION} | "
        f"ef_search={HNSW_EF_SEARCH}[/dim]"
    )
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

    score_table = Table(title="RAGAS Scores — exp_04_3 vs Baseline", show_header=True)
    score_table.add_column("Metric", style="bold")
    score_table.add_column("pgvector", justify="right")
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
        console.print("\n[dim]Expected: delta ≈ 0 — only the backend changed (HNSW + cosine, same as baseline).[/dim]")


# ── 5. SAVE ───────────────────────────────────────────────────────────────────
def save_results(samples: list[EvalSample], scores: dict[str, float]) -> Path:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "exp_04_3_pgvector",
        "timestamp":  timestamp,
        "config":     cfg.model_dump(),
        "vectordb":   "pgvector_sqlalchemy",
        "index":      f"hnsw_cosine_m{HNSW_M}_efc{HNSW_EF_CONSTRUCTION}_efs{HNSW_EF_SEARCH}",
        "retrieval":  "dense_only_cosine",
        "scores":     scores,
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
    out_path = results_dir / f"exp_04_3_pgvector_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2))
    return out_path


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    console.rule("[bold]Experiment 04-3 — pgvector Backend (PostgreSQL + SQLAlchemy ORM)")
    console.print(
        f"Config: chunk_size={cfg.chunking.chunk_size}, "
        f"model={cfg.embedding.model_name}, "
        f"top_k={cfg.retrieval.top_k}, "
        f"vectordb=pgvector_sqlalchemy"
    )
    console.print(
        f"[dim]HNSW: m={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}, "
        f"ef_search={HNSW_EF_SEARCH}[/dim]"
    )

    vectordb = PgVectorDB()
    try:
        embedder  = SentenceTransformerEmbedder(model_name=cfg.embedding.model_name)
        retriever = DenseRetriever(embedder=embedder, vectordb=vectordb)
        reranker  = IdentityReranker()
        generator = OpenAIGenerator(model=cfg.generation.model, max_tokens=cfg.generation.max_tokens)

        console.print("\n[dim]Resetting table for clean run...[/dim]")
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
        console.rule("[bold green]exp_04_3 complete")

    finally:
        vectordb.close()


if __name__ == "__main__":
    main()
