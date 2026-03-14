"""
Experiment 03-5 — Embedding Model: nvidia/NV-Embed-v2 (long context, asymmetric)

What changed vs baseline:
  - Embeddings: all-MiniLM-L6-v2 (384-dim, 256-token context, symmetric)
              → nvidia/NV-Embed-v2 (4096-dim, 32768-token context, asymmetric)

Everything else is identical to the baseline:
  - Chunking:    FixedSizeChunker (chunk_size=512, overlap=50)
  - VectorDB:    ChromaVectorDB (cosine similarity, local)
  - Retrieval:   DenseRetriever (top_k=5)
  - Reranking:   IdentityReranker (none)
  - Generation:  OpenAIGenerator (gpt-4o-mini)
  - Evaluation:  RAGAS (faithfulness + answer_relevancy + answer_similarity)

Why NV-Embed-v2:
  - NVIDIA's embedding model built on Mistral-7B-v0.1 (7B parameters)
  - 4096-dim output — highest dimensional model in this experiment series
  - 32768-token context window — designed for long documents (vs 256-512 for MiniLM)
  - Asymmetric: queries get a task instruction, passages are encoded as-is
  - Query instruction format: "Instruct: <task_desc>\nQuery: <text>"
  - Requires explicit L2 normalization (not done automatically)
  - EOS token must be appended to all inputs before encoding (model requirement)

API notes (from official model card):
  - Use sentence-transformers with trust_remote_code=True
  - Set model.max_seq_length = 32768 and model.tokenizer.padding_side = "right"
  - Add EOS token to every input text
  - normalize_embeddings=True required for cosine similarity to work correctly
  - batch_size=2 recommended for large model (memory constraint)

NOTE on hardware:
  - This is a 7B model — CPU inference will be slow (several minutes per batch)
  - If you have a GPU, sentence-transformers will use it automatically

Run: uv run python experiments/exp_03_5_nv_embed.py
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

from sentence_transformers import SentenceTransformer
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

EXPERIMENT_MODEL = "nvidia/NV-Embed-v2"
EXPERIMENT_NAME  = "exp_03_5_nv_embed"
COLLECTION_NAME  = "rag_lab_exp03_5"

# Task instruction for retrieval queries (from official NV-Embed-v2 model card)
NV_QUERY_INSTRUCTION = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "

BASELINE_SCORES = {
    "faithfulness":      0.641,
    "answer_relevancy":  0.583,
    "answer_similarity": 0.538,
}


# ── NV-Embed-v2 embedder ──────────────────────────────────────────────────────
class NVEmbedEmbedder:
    """
    NVIDIA NV-Embed-v2 via sentence-transformers.

    Key requirements from the official model card:
      1. trust_remote_code=True — model uses custom pooling code
      2. max_seq_length = 32768 — enable full long-context window
      3. tokenizer.padding_side = "right" — required for correct attention masking
      4. EOS token appended to every input text
      5. normalize_embeddings=True — model outputs unnormalized vectors
      6. Queries get instruction prefix; passages do not

    NV-Embed-v2 is asymmetric:
      - embed(texts)       → passage encoding (no instruction, EOS token appended)
      - embed_as_query()   → query encoding (instruction prefix + EOS token appended)
    """

    def __init__(self, model_name: str = EXPERIMENT_MODEL):
        self.model_name = model_name
        console.print(f"  [dim]Loading {model_name} (7B model, this may take a while)...[/dim]")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model.max_seq_length = 32768
        self.model.tokenizer.padding_side = "right"
        self._eos = self.model.tokenizer.eos_token

    def _add_eos(self, texts: list[str]) -> list[str]:
        # Required: EOS token must be appended to every input before encoding
        return [t + self._eos for t in texts]

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Passage encoding — no instruction prefix, EOS token appended
        embeddings = self.model.encode(
            self._add_eos(texts),
            batch_size=2,               # conservative batch size for 7B model
            normalize_embeddings=True,  # required for cosine similarity
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_as_query(self, query: str) -> list[float]:
        # Query encoding — instruction prefix + EOS token
        embeddings = self.model.encode(
            self._add_eos([query]),
            batch_size=1,
            prompt=NV_QUERY_INSTRUCTION,    # prepended by sentence-transformers
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()[0]

    @property
    def dimension(self) -> int:
        return 4096


def embed_chunks_nv(chunks: list[Chunk], embedder: NVEmbedEmbedder) -> list[EmbeddedChunk]:
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
class NVDenseRetriever:
    def __init__(self, embedder: NVEmbedEmbedder, vectordb: ChromaVectorDB):
        self.embedder = embedder
        self.vectordb = vectordb

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed_as_query(query)
        return self.vectordb.search(query_embedding, top_k=top_k)


# ── 1. INGEST ─────────────────────────────────────────────────────────────────
def ingest(vectordb: ChromaVectorDB, embedder: NVEmbedEmbedder) -> int:
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
    console.print(f"  [dim]Embedding {len(chunks)} chunks with 7B model — expect slow CPU inference[/dim]")

    embedded = embed_chunks_nv(chunks, embedder)
    vectordb.add_chunks(embedded)
    console.print(f"  {vectordb.count()} chunks indexed in ChromaDB")
    return len(chunks)


# ── 2. QUERY ──────────────────────────────────────────────────────────────────
def run_query(
    question: str,
    retriever: NVDenseRetriever,
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
                "mode": "asymmetric",
                "query_instruction": NV_QUERY_INSTRUCTION,
                "dimensions": 4096,
                "context_tokens": 32768,
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
    console.rule(f"[bold]Experiment 03-5 — Embedding: {EXPERIMENT_MODEL} (long-context, asymmetric)")
    console.print(
        f"Swapping: [red]all-MiniLM-L6-v2 (384-dim, 256-token ctx)[/] → [green]{EXPERIMENT_MODEL} (4096-dim, 32768-token ctx)[/]\n"
        f"Base:     Mistral-7B-v0.1 — largest model in this experiment series\n"
        f"Prefix:   passages = none  |  queries = 'Instruct: ...\\nQuery: <text>'\n"
        f"Fixed:    chunk_size={cfg.chunking.chunk_size}, top_k={cfg.retrieval.top_k}, generator={cfg.generation.model}\n"
        f"[yellow]NOTE: 7B model — CPU inference will be slow[/yellow]"
    )

    embedder  = NVEmbedEmbedder(model_name=EXPERIMENT_MODEL)
    vectordb  = ChromaVectorDB(
        collection_name=COLLECTION_NAME,
        persist_dir=str(ROOT / cfg.vectordb.persist_dir),
    )
    retriever = NVDenseRetriever(embedder=embedder, vectordb=vectordb)
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
    console.rule(f"[bold green]exp_03_5 complete — check delta vs baseline above")


if __name__ == "__main__":
    main()
