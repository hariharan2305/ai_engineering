"""
Experiment 02i — Contextual Chunk Enrichment (Anthropic, 2024)

Variable changed vs baseline:
  - Chunk content: raw chunk text → LLM-generated context prepended before embedding
  - Base chunking strategy: RecursiveChunker (best performer from exp_02_recursive)
  - Everything else identical: same corpus, same test questions, same embedder/retriever

What this experiment tests:
  This is NOT a chunking strategy in the traditional sense. The chunk BOUNDARIES
  are the same as exp_02_recursive. What changes is what gets EMBEDDED.

  Standard pipeline:   chunk(doc) → embed(chunk.text) → store
  This experiment:     chunk(doc) → enrich(chunk) → embed(context + chunk.text) → store

  For each chunk, we call gpt-4o-mini with the surrounding document context and
  ask it to produce a 1-2 sentence description of what the chunk is about and
  where it sits in the document. That description is prepended to the chunk text
  before embedding. The vector store then contains enriched representations.

  Anthropic's "Contextual Retrieval" paper (2024) showed this consistently
  improves retrieval recall — the enriched embedding carries document-level
  signal that the raw chunk text lacks.

Example transformation:
  Raw chunk:
    "...uses exponential backoff with full jitter. On each failure, it calls
    exponential_backoff(attempt) to compute the wait time..."

  After enrichment:
    "This chunk describes the retry logic in LLMClient, specifically how
    API failures are handled using exponential backoff. [CHUNK] ...uses
    exponential backoff with full jitter..."

Cost warning:
  This experiment calls gpt-4o-mini ONCE PER CHUNK during ingestion.
  With ~30-40 chunks at RecursiveChunker defaults, expect ~35 API calls.
  At gpt-4o-mini pricing this is fractions of a cent — but it does add
  latency to ingestion. The experiment prints a cost estimate before running.

What to observe:
  - answer_similarity delta vs baseline: does document-level context in embeddings help?
  - faithfulness: are answers more grounded when retrieved chunks carry more context?
  - Compare against exp_02_recursive (same boundaries, no enrichment) for the pure
    enrichment effect

Run: uv run python experiments/exp_02_contextual_enrichment.py
"""

import json
import sys
import time
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
    RecursiveChunker, chunk_documents,
    SentenceTransformerEmbedder, embed_chunks,
    ChromaVectorDB,
    DenseRetriever,
    IdentityReranker,
    OpenAIGenerator,
    EvalSample, evaluate_pipeline,
    Chunk,
)
from configs import RAGConfig
from configs.rag_config import VectorDBConfig

cfg = RAGConfig(
    vectordb=VectorDBConfig(collection_name="rag_lab_exp02_contextual")
)

console = Console()

# Enrichment prompt — mirrors Anthropic's original formulation
ENRICHMENT_SYSTEM = (
    "You are a document analyst. Given a chunk of text from a larger document, "
    "produce a single concise sentence (max 30 words) that describes what this "
    "chunk is about and where it fits in the document. Be specific, not generic."
)
ENRICHMENT_USER_TEMPLATE = (
    "Document title: {title}\n\n"
    "Chunk text:\n{chunk_text}\n\n"
    "Provide a single sentence describing what this chunk covers."
)


# ── Enrichment Step ───────────────────────────────────────────────────────────

def enrich_chunk(chunk: Chunk, llm_client, doc_title: str) -> Chunk:
    """
    Call gpt-4o-mini to generate a context sentence for this chunk,
    then prepend it to chunk.text before embedding.

    The enriched text format:
        "[CONTEXT] <generated sentence> [CHUNK] <original chunk text>"

    The [CONTEXT] and [CHUNK] markers make it easy to strip the context
    back out at retrieval time if you want to show users only the original
    text — or to inspect the enrichment quality.
    """
    prompt = ENRICHMENT_USER_TEMPLATE.format(
        title=doc_title,
        chunk_text=chunk.text[:800],  # limit input to avoid excessive token usage
    )
    messages = [
        {"role": "system", "content": ENRICHMENT_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    context_sentence = llm_client.chat(messages, temperature=0.0, max_tokens=60)
    enriched_text = f"[CONTEXT] {context_sentence.strip()} [CHUNK] {chunk.text}"

    # Return a new Chunk with the enriched text; all other fields unchanged
    return Chunk(
        id=chunk.id,
        text=enriched_text,
        doc_id=chunk.doc_id,
        chunk_index=chunk.chunk_index,
        metadata={**chunk.metadata, "enriched": True, "context": context_sentence.strip()},
    )


def enrich_all_chunks(chunks: list[Chunk], llm_client, doc_title_map: dict) -> list[Chunk]:
    """
    Enrich all chunks with LLM-generated context sentences.
    Prints progress and a rough time estimate as it goes.
    """
    enriched = []
    console.print(f"\n[bold cyan]Enriching {len(chunks)} chunks with gpt-4o-mini...[/]")
    console.print("[dim](1 API call per chunk — expect ~1-2s per chunk)[/dim]\n")

    t0 = time.perf_counter()
    for i, chunk in enumerate(chunks):
        doc_title = doc_title_map.get(chunk.doc_id, "document")
        enriched_chunk = enrich_chunk(chunk, llm_client, doc_title)
        enriched.append(enriched_chunk)

        elapsed = time.perf_counter() - t0
        avg = elapsed / (i + 1)
        remaining = avg * (len(chunks) - i - 1)
        console.print(
            f"  [green]✓[/] Chunk {i + 1:02d}/{len(chunks)} "
            f"[dim]~{remaining:.0f}s remaining[/dim]"
        )

    total = time.perf_counter() - t0
    console.print(f"\n  Enrichment complete in {total:.1f}s ({len(chunks)} chunks)")
    return enriched


# ── 1. INGEST ─────────────────────────────────────────────────────────────────
def ingest(
    vectordb: ChromaVectorDB,
    embedder: SentenceTransformerEmbedder,
    llm_client,
) -> int:
    corpus_dir = ROOT / "corpus"
    console.print(f"\n[bold cyan]Loading corpus from:[/] {corpus_dir}")
    docs = load_directory(corpus_dir, extensions=[".txt"])
    console.print(f"  {len(docs)} documents loaded")

    # Step 1: chunk with RecursiveChunker (same boundaries as exp_02_recursive)
    chunker = RecursiveChunker(
        chunk_size=cfg.chunking.chunk_size,
        overlap=cfg.chunking.overlap,
    )
    chunks = chunk_documents(docs, chunker)
    console.print(
        f"  {len(chunks)} chunks created (RecursiveChunker — same boundaries as exp_02)"
    )

    # Build a doc_id → title map for the enrichment prompt
    doc_title_map = {doc.id: doc.metadata.get("source", "document") for doc in docs}

    # Step 2: enrich each chunk — THIS is the variable being tested
    enriched_chunks = enrich_all_chunks(chunks, llm_client, doc_title_map)

    # Step 3: embed enriched text (not original text) and store
    embedded = embed_chunks(enriched_chunks, embedder)
    vectordb.add_chunks(embedded)
    console.print(f"  {vectordb.count()} enriched chunks indexed in ChromaDB")
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
def load_baseline_scores() -> dict[str, float] | None:
    results_dir = ROOT / "results"
    baseline_files = sorted(results_dir.glob("exp_01_baseline_*.json"), reverse=True)
    if not baseline_files:
        return None
    return json.loads(baseline_files[0].read_text()).get("scores")


def load_recursive_scores() -> dict[str, float] | None:
    results_dir = ROOT / "results"
    recursive_files = sorted(results_dir.glob("exp_02_recursive_*.json"), reverse=True)
    if not recursive_files:
        return None
    return json.loads(recursive_files[0].read_text()).get("scores")


def display_results(samples: list[EvalSample], scores: dict[str, float]) -> None:
    qa_table = Table(title="Per-Question Results", show_lines=True)
    qa_table.add_column("Question", style="cyan", max_width=40)
    qa_table.add_column("Answer", style="white", max_width=60)
    qa_table.add_column("Chunks", style="dim", justify="center")
    for s in samples:
        qa_table.add_row(s.question, s.answer, str(len(s.contexts)))
    console.print(qa_table)

    baseline_scores = load_baseline_scores()
    recursive_scores = load_recursive_scores()

    # 3-way table: baseline vs recursive (same boundaries, no enrichment) vs this
    score_table = Table(
        title="RAGAS Scores — Contextual Enrichment vs Recursive vs Baseline",
        show_header=True,
    )
    score_table.add_column("Metric", style="bold")
    score_table.add_column("Baseline", justify="right")
    score_table.add_column("Recursive (no enrich)", justify="right")
    score_table.add_column("Enriched", justify="right")
    score_table.add_column("vs Recursive", justify="right")

    for metric, score in scores.items():
        color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"

        baseline_val = baseline_scores.get(metric) if baseline_scores else None
        recursive_val = recursive_scores.get(metric) if recursive_scores else None

        baseline_str = f"{baseline_val:.3f}" if baseline_val is not None else "n/a"
        recursive_str = f"{recursive_val:.3f}" if recursive_val is not None else "n/a"

        if recursive_val is not None:
            delta = score - recursive_val
            dc = "green" if delta > 0.01 else "red" if delta < -0.01 else "dim"
            delta_str = f"[{dc}]{delta:+.3f}[/{dc}]"
        else:
            delta_str = "n/a"

        score_table.add_row(
            metric,
            baseline_str,
            recursive_str,
            f"[{color}]{score:.3f}[/{color}]",
            delta_str,
        )
    console.print(score_table)
    console.print(
        "\n[dim]'vs Recursive' isolates the pure enrichment effect: "
        "same chunk boundaries, same everything — only the embedded text changed.[/dim]"
    )


# ── 5. SAVE ───────────────────────────────────────────────────────────────────
def save_results(samples: list[EvalSample], scores: dict[str, float]) -> Path:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "exp_02_contextual_enrichment",
        "timestamp": timestamp,
        "config": cfg.model_dump(),
        "variable_changed": "chunk_content: raw text → LLM context prepended before embedding",
        "base_chunker": "RecursiveChunker (same as exp_02_recursive)",
        "enrichment_model": "gpt-4o-mini",
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
    out_path = results_dir / f"exp_02_contextual_enrichment_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2))
    return out_path


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    console.rule("[bold]Experiment 02i — Contextual Chunk Enrichment")
    console.print("Variable changed: [yellow]what gets embedded[/] raw chunk → LLM context + chunk")
    console.print(
        "Base chunker: [dim]RecursiveChunker[/] (chunk BOUNDARIES unchanged vs exp_02_recursive)\n"
        "Pure enrichment effect: compare 'vs Recursive' column in the results table"
    )
    console.print(
        f"\n[yellow]⚠ Cost notice:[/] This experiment calls gpt-4o-mini once per chunk "
        f"during ingestion. Estimated: ~35 calls at <$0.01 total."
    )
    console.print(
        f"Config: chunk_size={cfg.chunking.chunk_size} chars, "
        f"overlap={cfg.chunking.overlap}, "
        f"top_k={cfg.retrieval.top_k}, "
        f"generator={cfg.generation.model}"
    )

    embedder = SentenceTransformerEmbedder(model_name=cfg.embedding.model_name)
    vectordb = ChromaVectorDB(
        collection_name=cfg.vectordb.collection_name,
        persist_dir=str(ROOT / "chroma_db"),
    )
    retriever = DenseRetriever(embedder=embedder, vectordb=vectordb)
    reranker = IdentityReranker()
    generator = OpenAIGenerator(model=cfg.generation.model, max_tokens=cfg.generation.max_tokens)
    # Separate LLMClient for enrichment — keeps token tracking independent
    from components.generation import OpenAIGenerator as Gen
    import openai as _oai
    class _EnrichClient:
        def __init__(self):
            self._client = _oai.OpenAI()
            self._tokens = 0
        def chat(self, messages, temperature=0.0, max_tokens=60):
            r = self._client.chat.completions.create(
                model="gpt-4o-mini", messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            self._tokens += r.usage.total_tokens
            return r.choices[0].message.content
        def get_token_usage(self):
            return {"total_tokens": self._tokens}

    llm_client = _EnrichClient()

    console.print("\n[dim]Resetting vector index for clean run...[/dim]")
    vectordb.reset()
    ingest(vectordb, embedder, llm_client)
    console.print(
        f"\n[dim]Enrichment used {llm_client.get_token_usage()['total_tokens']:,} tokens[/dim]"
    )

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
    console.rule("[bold green]Exp 02i complete — check 'vs Recursive' delta for enrichment effect")


if __name__ == "__main__":
    main()
