"""
Experiment 02c — Markdown Header Chunking

Variable changed vs baseline:
  - Chunking: FixedSizeChunker → MarkdownChunker (LangChain MarkdownHeaderTextSplitter)
  - Corpus: rag_advanced_techniques.md (Markdown-structured document)
  - Test questions: test_questions_markdown.json (5 questions tied to specific sections)

Why this is different from other chunking experiments:
  - Not a direct baseline comparison (different corpus + questions)
  - Tests whether structure-aware chunking on a Markdown document
    retrieves better answers than if we had chunked the same doc with fixed-size splits

What to observe:
  - Chunk count: one chunk per header section — much fewer, larger chunks than fixed-size
  - Chunk metadata: each chunk carries its header path (h1, h2, h3) — useful for filtering
  - answer_similarity: are section-aligned chunks better matched to section-specific questions?

Run: uv run python experiments/exp_02_markdown.py
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
    load_text_file,
    MarkdownChunker, chunk_documents,
    SentenceTransformerEmbedder, embed_chunks,
    ChromaVectorDB,
    DenseRetriever,
    IdentityReranker,
    OpenAIGenerator,
    EvalSample, evaluate_pipeline,
)
from configs import BASELINE_CONFIG, RAGConfig
from configs.rag_config import VectorDBConfig

cfg = RAGConfig(
    vectordb=VectorDBConfig(collection_name="rag_lab_exp02_markdown")
)

console = Console()


# ── 1. INGEST ─────────────────────────────────────────────────────────────────
def ingest(vectordb: ChromaVectorDB, embedder: SentenceTransformerEmbedder) -> int:
    corpus_file = ROOT / "corpus" / "rag_advanced_techniques.md"
    console.print(f"\n[bold cyan]Loading:[/] {corpus_file.name}")

    doc = load_text_file(str(corpus_file))
    console.print(f"  1 markdown document loaded ({len(doc.text)} chars)")

    chunker = MarkdownChunker()
    chunks = chunk_documents([doc], chunker)
    console.print(f"  {len(chunks)} chunks created (strategy=markdown_headers)")

    # Show what headers were found
    for c in chunks:
        header_path = " > ".join(
            v for k, v in c.metadata.items() if k in ("h1", "h2", "h3") and v
        )
        console.print(f"    [dim]chunk {c.chunk_index}: {header_path[:70]}[/dim]")

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
    return evaluate_pipeline(samples, embedder=embedder)


# ── 4. DISPLAY ────────────────────────────────────────────────────────────────
def display_results(samples: list[EvalSample], scores: dict[str, float]) -> None:
    qa_table = Table(title="Per-Question Results", show_lines=True)
    qa_table.add_column("Question", style="cyan", max_width=45)
    qa_table.add_column("Answer", style="white", max_width=60)
    qa_table.add_column("Chunks", style="dim", justify="center")
    for s in samples:
        qa_table.add_row(s.question[:45], s.answer[:200], str(len(s.contexts)))
    console.print(qa_table)

    score_table = Table(title="RAGAS Scores — Markdown Chunking", show_header=True)
    score_table.add_column("Metric", style="bold")
    score_table.add_column("Score", justify="right")
    score_table.add_column("Interpretation")

    interpretations = {
        "faithfulness":      "[LLM judge]   Grounded in context?",
        "answer_relevancy":  "[LLM judge]   Addresses the question?",
        "answer_similarity": "[Deterministic] Cosine sim vs ground truth",
    }
    for metric, score in scores.items():
        color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
        score_table.add_row(
            metric,
            f"[{color}]{score:.3f}[/{color}]",
            interpretations.get(metric, ""),
        )
    console.print(score_table)


# ── 5. SAVE ───────────────────────────────────────────────────────────────────
def save_results(samples: list[EvalSample], scores: dict[str, float]) -> Path:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "exp_02_markdown",
        "timestamp": timestamp,
        "config": cfg.model_dump(),
        "variable_changed": "chunking_strategy: fixed → markdown_headers",
        "corpus": "rag_advanced_techniques.md",
        "scores": scores,
        "samples": [
            {"question": s.question, "answer": s.answer,
             "num_contexts": len(s.contexts), "ground_truth": s.ground_truth}
            for s in samples
        ],
    }
    out_path = results_dir / f"exp_02_markdown_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2))
    return out_path


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    console.rule("[bold]Experiment 02c — Markdown Header Chunking")
    console.print("Corpus: [yellow]rag_advanced_techniques.md[/]")
    console.print("Strategy: [yellow]MarkdownHeaderTextSplitter[/] — splits on #, ##, ###")

    embedder = SentenceTransformerEmbedder(model_name=cfg.embedding.model_name)
    vectordb = ChromaVectorDB(
        collection_name=cfg.vectordb.collection_name,
        persist_dir=str(ROOT / "chroma_db"),
    )
    retriever = DenseRetriever(embedder=embedder, vectordb=vectordb)
    reranker = IdentityReranker()
    generator = OpenAIGenerator(model=cfg.generation.model, max_tokens=cfg.generation.max_tokens)

    console.print("\n[dim]Resetting vector index for clean run...[/dim]")
    vectordb.reset()
    ingest(vectordb, embedder)

    test_path = ROOT / "corpus" / "test_questions_markdown.json"
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
        console.print(f"  [green]✓[/] {item['question'][:65]}...")

    scores = run_evaluation(samples, embedder=embedder)
    display_results(samples, scores)

    out_path = save_results(samples, scores)
    console.print(f"\n[dim]Results saved to: {out_path}[/dim]")
    console.rule("[bold green]Exp 02c complete — Markdown header chunking")


if __name__ == "__main__":
    main()
