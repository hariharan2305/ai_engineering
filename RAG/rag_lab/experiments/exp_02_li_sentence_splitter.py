"""
Experiment 02e — LlamaIndex SentenceSplitter

Variable changed vs baseline:
  - Chunking: FixedSizeChunker → LISentenceSplitter (llama_index SentenceSplitter)
  - Same chunk_size (512) and overlap (50) — only the splitting strategy changes
  - Same corpus (.txt files), same test questions, same embedder/retriever/generator

Why this experiment:
  - Direct LlamaIndex counterpart to exp_02_recursive_chunking.py (LangChain RecursiveChunker)
  - Both respect sentence/paragraph boundaries instead of cutting on character count
  - Lets you compare LangChain vs LlamaIndex implementations on identical inputs

LlamaIndex concepts demonstrated:
  - llama_index.core.Document — LI's document wrapper
  - SentenceSplitter.from_defaults() — LI's standard constructor pattern
  - get_nodes_from_documents() — LI's processing entry point
  - TextNode — LI's chunk type (.text, .metadata, .node_id)

What to observe:
  - Chunk count vs recursive: similar (both respect sentence boundaries)
  - answer_similarity delta vs baseline — does sentence-aware splitting help?
  - Compare scores against exp_02_recursive_chunking results (same strategy, different framework)

Run: uv run python experiments/exp_02_li_sentence_splitter.py
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
    LISentenceSplitter, chunk_documents,
    SentenceTransformerEmbedder, embed_chunks,
    ChromaVectorDB,
    DenseRetriever,
    IdentityReranker,
    OpenAIGenerator,
    EvalSample, evaluate_pipeline,
)
from configs import RAGConfig
from configs.rag_config import VectorDBConfig

cfg = RAGConfig(
    vectordb=VectorDBConfig(collection_name="rag_lab_exp02_li_sentence")
)

console = Console()


# ── 1. INGEST ─────────────────────────────────────────────────────────────────
def ingest(vectordb: ChromaVectorDB, embedder: SentenceTransformerEmbedder) -> int:
    corpus_dir = ROOT / "corpus"
    console.print(f"\n[bold cyan]Loading corpus from:[/] {corpus_dir}")
    docs = load_directory(corpus_dir, extensions=[".txt"])
    console.print(f"  {len(docs)} documents loaded")

    chunker = LISentenceSplitter(
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.overlap,
    )
    chunks = chunk_documents(docs, chunker)
    console.print(
        f"  {len(chunks)} chunks created "
        f"(strategy=li_sentence_splitter, size={cfg.chunking.chunk_size}, overlap={cfg.chunking.overlap})"
    )

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
def load_baseline_scores() -> dict[str, float] | None:
    results_dir = ROOT / "results"
    baseline_files = sorted(results_dir.glob("exp_01_baseline_*.json"), reverse=True)
    if not baseline_files:
        return None
    return json.loads(baseline_files[0].read_text()).get("scores")


def display_results(samples: list[EvalSample], scores: dict[str, float]) -> None:
    qa_table = Table(title="Per-Question Results", show_lines=True)
    qa_table.add_column("Question", style="cyan", max_width=40)
    qa_table.add_column("Answer", style="white", max_width=60)
    qa_table.add_column("Chunks Retrieved", style="dim", justify="center")
    for s in samples:
        qa_table.add_row(s.question, s.answer, str(len(s.contexts)))
    console.print(qa_table)

    baseline_scores = load_baseline_scores()

    score_table = Table(title="RAGAS Scores — LI SentenceSplitter vs Baseline", show_header=True)
    score_table.add_column("Metric", style="bold")
    score_table.add_column("Baseline", justify="right")
    score_table.add_column("LI SentenceSplitter", justify="right")
    score_table.add_column("Delta", justify="right")
    score_table.add_column("Interpretation")

    interpretations = {
        "faithfulness":      "[LLM judge]   Grounded in context?",
        "answer_relevancy":  "[LLM judge]   Addresses the question?",
        "answer_similarity": "[Deterministic] Cosine sim vs ground truth — most reliable",
    }

    for metric, score in scores.items():
        color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
        baseline_val = baseline_scores.get(metric) if baseline_scores else None
        baseline_str = f"{baseline_val:.3f}" if baseline_val is not None else "n/a"

        if baseline_val is not None:
            delta = score - baseline_val
            delta_color = "green" if delta > 0.01 else "red" if delta < -0.01 else "dim"
            delta_str = f"[{delta_color}]{delta:+.3f}[/{delta_color}]"
        else:
            delta_str = "n/a"

        score_table.add_row(
            metric,
            baseline_str,
            f"[{color}]{score:.3f}[/{color}]",
            delta_str,
            interpretations.get(metric, ""),
        )
    console.print(score_table)


# ── 5. SAVE ───────────────────────────────────────────────────────────────────
def save_results(samples: list[EvalSample], scores: dict[str, float]) -> Path:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "exp_02_li_sentence_splitter",
        "timestamp": timestamp,
        "config": cfg.model_dump(),
        "variable_changed": "chunking_strategy: fixed → li_sentence_splitter",
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
    out_path = results_dir / f"exp_02_li_sentence_splitter_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2))
    return out_path


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    console.rule("[bold]Experiment 02e — LlamaIndex SentenceSplitter")
    console.print("Variable changed: [yellow]chunking strategy[/] fixed → LI SentenceSplitter")
    console.print(
        f"Config: chunk_size={cfg.chunking.chunk_size}, "
        f"overlap={cfg.chunking.overlap}, "
        f"top_k={cfg.retrieval.top_k}, "
        f"generator={cfg.generation.model}"
    )
    console.print("[dim]Compare against: exp_02_recursive_chunking (same strategy, LangChain)[/dim]")

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
    console.rule("[bold green]Exp 02e complete — LI SentenceSplitter vs baseline")


if __name__ == "__main__":
    main()
