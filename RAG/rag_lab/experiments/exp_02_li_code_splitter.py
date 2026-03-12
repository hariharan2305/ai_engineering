"""
Experiment 02h — LlamaIndex CodeSplitter (AST-aware chunking)

Variable changed vs baseline:
  - Chunking: FixedSizeChunker → LICodeSplitter (tree-sitter AST parser)
  - Corpus: prose .txt files → llm_pipeline_code.py (production Python code)
  - Test questions: test_questions.json → test_questions_code.json

This experiment is structurally different from all others:
  - Different corpus (code, not prose) — because AST chunking only makes
    sense on actual source code
  - Different test questions (code-comprehension, not RAG theory)
  - No direct metric comparison to baseline (different domain entirely)

What CodeSplitter does differently from every other chunker:
  All other chunkers in this lab operate on raw text:
    - Fixed: cut every N chars, done
    - Recursive: prefer paragraph/sentence boundaries
    - Semantic: embed sentences, split on similarity drops
    - Token: count tokens, cut when budget exceeded

  CodeSplitter uses tree-sitter to build an Abstract Syntax Tree of the
  source code, then splits on SYNTACTIC boundaries:
    - Function definitions stay together (def ... → end of body)
    - Class definitions stay together (class ... → end of body)
    - Docstrings remain attached to the code they document
    - Import blocks stay as a unit

  A chunk that cuts through the middle of a function body is syntactically
  broken — no embedder can make good sense of it. AST-aware splitting
  eliminates that entire class of problem.

Knobs (different from all other chunkers):
  chunk_lines:         Target lines per chunk (not chars, not tokens)
  chunk_lines_overlap: Lines of overlap between chunks
  max_chars:           Hard character ceiling — safety net for huge functions

What to observe:
  - Inspect the chunks printed during ingest — they should be whole functions
  - Do the retrieved chunks actually answer code-comprehension questions well?
  - Does faithfulness stay high? (The LLM has clean, complete code in context)

Run: uv run python experiments/exp_02_li_code_splitter.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from components import (
    load_directory,
    LICodeSplitter, chunk_documents,
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
    vectordb=VectorDBConfig(collection_name="rag_lab_exp02_li_code")
)

console = Console()


# ── 1. INGEST ─────────────────────────────────────────────────────────────────
def ingest(vectordb: ChromaVectorDB, embedder: SentenceTransformerEmbedder) -> int:
    corpus_dir = ROOT / "corpus"
    console.print(f"\n[bold cyan]Loading corpus from:[/] {corpus_dir}")

    # Only load .py files — CodeSplitter needs source code, not prose
    docs = load_directory(corpus_dir, extensions=[".py"])
    console.print(f"  {len(docs)} Python file(s) loaded")

    # tree-sitter will parse Python AST and split on function/class boundaries.
    # chunk_lines=40: target ~40 lines per chunk (a typical function body)
    # chunk_lines_overlap=10: 10-line overlap so docstrings aren't orphaned
    # max_chars=1500: hard ceiling — very large classes get split at max_chars
    chunker = LICodeSplitter(
        language="python",
        chunk_lines=40,
        chunk_lines_overlap=10,
        max_chars=1500,
    )
    chunks = chunk_documents(docs, chunker)
    console.print(
        f"  {len(chunks)} chunks created (strategy=li_code_splitter, language=python)"
    )

    # Print each chunk so you can see AST-aware boundaries in action
    console.print("\n[bold yellow]Chunks produced by tree-sitter AST parser:[/]")
    for i, chunk in enumerate(chunks):
        first_line = chunk.text.split("\n")[0][:70]
        console.print(f"  [dim]Chunk {i + 1:02d}:[/] {first_line}...")

    embedded = embed_chunks(chunks, embedder)
    vectordb.add_chunks(embedded)
    console.print(f"\n  {vectordb.count()} chunks indexed in ChromaDB")
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
    qa_table = Table(title="Per-Question Results (Code Corpus)", show_lines=True)
    qa_table.add_column("Question", style="cyan", max_width=45)
    qa_table.add_column("Answer", style="white", max_width=60)
    qa_table.add_column("Chunks", style="dim", justify="center")
    for s in samples:
        qa_table.add_row(s.question, s.answer, str(len(s.contexts)))
    console.print(qa_table)

    score_table = Table(
        title="RAGAS Scores — Code Corpus (no baseline comparison, different domain)",
        show_header=True,
    )
    score_table.add_column("Metric", style="bold")
    score_table.add_column("Score", justify="right")
    score_table.add_column("Interpretation")

    interpretations = {
        "faithfulness":      "[LLM judge]   Is the answer grounded in the retrieved code?",
        "answer_relevancy":  "[LLM judge]   Does the answer address the code question?",
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
    console.print(
        "\n[dim]Note: No baseline comparison — this experiment uses a different corpus "
        "(code vs prose). The question is whether AST-aware chunks produce "
        "semantically coherent, answerable context for code questions.[/dim]"
    )


# ── 5. SAVE ───────────────────────────────────────────────────────────────────
def save_results(samples: list[EvalSample], scores: dict[str, float]) -> Path:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "exp_02_li_code_splitter",
        "timestamp": timestamp,
        "config": cfg.model_dump(),
        "variable_changed": "chunking_strategy: fixed_char → AST-aware (tree-sitter python)",
        "corpus": "llm_pipeline_code.py",
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
    out_path = results_dir / f"exp_02_li_code_splitter_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2))
    return out_path


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    console.rule("[bold]Experiment 02h — LlamaIndex CodeSplitter (AST-aware)")
    console.print("Variable changed: [yellow]chunking strategy[/] fixed chars → tree-sitter AST")
    console.print(
        "Corpus: [yellow]llm_pipeline_code.py[/] — production LLM/RAG Python utilities\n"
        "  Classes: LLMClient, EmbeddingCache, RAGPipeline, RAGResponse\n"
        "  Functions: exponential_backoff, cosine_similarity, truncate_to_token_limit"
    )
    console.print(
        f"Config: chunk_lines=40, overlap=10, max_chars=1500, "
        f"top_k={cfg.retrieval.top_k}, generator={cfg.generation.model}"
    )

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

    test_path = ROOT / "corpus" / "test_questions_code.json"
    test_data = json.loads(test_path.read_text())
    console.print(f"\n[bold cyan]Running {len(test_data)} code-comprehension questions...[/]")

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
    console.rule("[bold green]Exp 02h complete — review chunk boundaries printed above")


if __name__ == "__main__":
    main()
