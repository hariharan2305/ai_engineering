"""
Experiment 03-4 — Embedding Model: perplexity-ai/pplx-embed-v1-4b (HuggingFace local)

What changed vs baseline:
  - Embeddings: all-MiniLM-L6-v2 (local, 384-dim, symmetric)
              → perplexity-ai/pplx-embed-v1-4b (local via HF, 2560-dim, symmetric, no prefixes)

Everything else is identical to the baseline:
  - Chunking:    FixedSizeChunker (chunk_size=512, overlap=50)
  - VectorDB:    ChromaVectorDB (cosine similarity, local)
  - Retrieval:   DenseRetriever (top_k=5)
  - Reranking:   IdentityReranker (none)
  - Generation:  OpenAIGenerator (gpt-4o-mini)
  - Evaluation:  RAGAS (faithfulness + answer_relevancy + answer_similarity)

Why pplx-embed-v1-4b:
  - Perplexity's embedding model built on a 4B-parameter base (Qwen3 architecture)
  - Uses bidirectional attention via diffusion pretraining — richer context representation
    than standard causal LLM encoders
  - No instruction prefixes needed — symmetric model, embed queries and passages identically
  - 2560-dim output vectors (vs 384 baseline, 1024 for E5/BGE)
  - MIT license — available locally via HuggingFace (gated: requires HF_TOKEN)
  - Also available as a cloud API via Perplexity (see commented section below)

NOTE on input_type:
  - pplx-embed does NOT have an input_type parameter (unlike Voyage AI)
  - Queries and passages are encoded identically — no asymmetry
  - Differentiation happens at the application level (embed separately, compare with cosine)

TWO MODES (switch by commenting/uncommenting the embedder class in use):
  - [ACTIVE]    HuggingFace local:  requires HF_TOKEN in .env, downloads the model
  - [COMMENTED] Perplexity API:     requires PERPLEXITY_API_KEY in .env, no download

Requires: HF_TOKEN in .env (model is gated on HuggingFace)

Run: uv run python experiments/exp_03_4_pplx.py
"""

import json
import os
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
    IdentityReranker,
    OpenAIGenerator,
    EvalSample, evaluate_pipeline,
)
from configs import BASELINE_CONFIG

console = Console()
cfg = BASELINE_CONFIG

# HuggingFace model ID (gated — requires HF_TOKEN in .env)
# SentenceTransformer reads HF_TOKEN from the environment automatically.
EXPERIMENT_MODEL = "perplexity-ai/pplx-embed-v1-0.6b"
EXPERIMENT_NAME  = "exp_03_4_pplx"
COLLECTION_NAME  = "rag_lab_exp03_4"

# ── To switch to the Perplexity cloud API instead: ────────────────────────────
# pplx-embed is also available as a REST API (OpenAI-compatible endpoint).
# The API uses a different model name (no org prefix) and requires PERPLEXITY_API_KEY.
#
# Replace the SentenceTransformerEmbedder usage in main() with a custom embedder:
#
# from openai import OpenAI
#
# class PerplexityAPIEmbedder:
#     def __init__(self):
#         self.client = OpenAI(
#             api_key=os.environ["PERPLEXITY_API_KEY"],
#             base_url="https://api.perplexity.ai",
#         )
#     def embed(self, texts: list[str]) -> list[list[float]]:
#         response = self.client.embeddings.create(
#             input=texts,
#             model="pplx-embed-v1-4b",   # API name has no org prefix
#             # dimensions=N,             # optional: Matryoshka truncation
#         )
#         return [item.embedding for item in response.data]  # sorted by index
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_SCORES = {
    "faithfulness":      0.641,
    "answer_relevancy":  0.583,
    "answer_similarity": 0.538,
}


# ── 1. INGEST ─────────────────────────────────────────────────────────────────
def ingest(vectordb: ChromaVectorDB, embedder: SentenceTransformerEmbedder) -> int:
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
            "embedding": {"model_name": EXPERIMENT_MODEL, "mode": "symmetric", "provider": "huggingface_local"},
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
    console.rule(f"[bold]Experiment 03-4 — Embedding: {EXPERIMENT_MODEL} (HuggingFace local)")
    console.print(
        f"Swapping: [red]all-MiniLM-L6-v2 (384-dim, local)[/] → [green]{EXPERIMENT_MODEL} (2560-dim, local via HF)[/]\n"
        f"Mode:     symmetric — no prefixes for queries or passages\n"
        f"Fixed:    chunk_size={cfg.chunking.chunk_size}, top_k={cfg.retrieval.top_k}, generator={cfg.generation.model}"
    )

    # ── To switch to Perplexity API mode: ─────────────────────────────────────
    # 1. Swap the active embedder class (see commented class above)
    # 2. Replace the check below with:
    #    if not os.environ.get("PERPLEXITY_API_KEY"):
    #        console.print("[bold red]ERROR: PERPLEXITY_API_KEY not found in .env[/]")
    #        return
    # 3. Change EXPERIMENT_MODEL to the API name: "pplx-embed-v1-4b" (no org prefix)
    # ──────────────────────────────────────────────────────────────────────────
    if not os.environ.get("HF_TOKEN"):
        console.print("[bold red]ERROR: HF_TOKEN not found in .env (required for gated model)[/]")
        return

    embedder  = SentenceTransformerEmbedder(model_name=EXPERIMENT_MODEL)
    vectordb  = ChromaVectorDB(
        collection_name=COLLECTION_NAME,
        persist_dir=str(ROOT / cfg.vectordb.persist_dir),
    )
    retriever = DenseRetriever(embedder=embedder, vectordb=vectordb)
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
    console.rule(f"[bold green]exp_03_4 complete — check delta vs baseline above")


if __name__ == "__main__":
    main()
