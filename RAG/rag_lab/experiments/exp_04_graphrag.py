"""
Experiment 04-GraphRAG — Microsoft GraphRAG (Knowledge Graph Retrieval)

What changed vs all previous experiments:
  - Architecture: flat chunk index → knowledge graph (entities + relationships + communities)
  - Retrieval:    vector similarity → graph traversal + community summary synthesis
  - Index cost:   ~1 LLM call per chunk → ~10–50x more LLM calls (entity extraction,
                  community detection, community summarization)

Two search modes:
  LOCAL SEARCH  — entity-centric graph traversal. Best for specific factual questions.
                  Walks the graph from matched entities, returns subgraph + source text.
  GLOBAL SEARCH — community-summary synthesis. Best for thematic/overview questions.
                  Map-reduce across LLM-generated community summaries. This mode is
                  impossible with any chunk-based retrieval system.

Supported corpus formats (markitdown input type):
  .txt, .md, .html, .pdf, .docx — copy any of these to graphrag_workspace/input/

Workspace structure (graphrag_workspace/):
  input/           — corpus files (auto-copied from corpus/*.txt at setup)
  output/          — parquet files produced by indexing pipeline
  output/lancedb/  — entity embeddings for local search
  cache/           — LLM call cache (avoids re-runs on repeated index)
  logs/            — indexing run logs
  prompts/         — prompt files (auto-generated from graphrag built-ins at setup)
  settings.yaml    — GraphRAG configuration
  .env             — GRAPHRAG_API_KEY (auto-created from project .env)

Cost warning:
  Indexing 3 documents with gpt-4o-mini typically costs $0.50–2.00.
  Run --phase index only once; results persist in output/.

Usage:
  # Step 1 — setup + build knowledge graph (run once, expensive):
  uv run python experiments/exp_04_graphrag.py --phase index

  # Step 2 — query + RAGAS evaluation (run repeatedly, cheap):
  uv run python experiments/exp_04_graphrag.py --phase query --search local
  uv run python experiments/exp_04_graphrag.py --phase query --search global

  # Both phases in sequence:
  uv run python experiments/exp_04_graphrag.py --phase all --search local
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "graphrag_workspace"
sys.path.insert(0, str(ROOT))

# Load project .env first (OPENAI_API_KEY), then workspace .env (GRAPHRAG_API_KEY)
load_dotenv(ROOT / ".env")
load_dotenv(WORKSPACE / ".env", override=False)

from components import EvalSample, SentenceTransformerEmbedder, evaluate_pipeline
from configs import BASELINE_CONFIG

console = Console()
cfg = BASELINE_CONFIG

BASELINE_SCORES = {
    "faithfulness":      0.641,
    "answer_relevancy":  0.583,
    "answer_similarity": 0.538,
}


# ── SETUP ─────────────────────────────────────────────────────────────────────

def _write_default_prompts() -> None:
    """
    Export graphrag's built-in default prompts to the workspace prompts/ directory.
    This avoids the extra LLM cost of generate_indexing_prompts while still giving
    graphrag the prompt files it needs. Domain-specific prompt tuning can be done
    later by running `graphrag prompt-tune --root graphrag_workspace`.
    """
    from graphrag.prompts.index.extract_graph import GRAPH_EXTRACTION_PROMPT
    from graphrag.prompts.index.summarize_descriptions import SUMMARIZE_PROMPT
    from graphrag.prompts.index.community_report import COMMUNITY_REPORT_PROMPT
    from graphrag.prompts.index.extract_claims import EXTRACT_CLAIMS_PROMPT
    from graphrag.prompts.query.local_search_system_prompt import LOCAL_SEARCH_SYSTEM_PROMPT
    from graphrag.prompts.query.global_search_map_system_prompt import MAP_SYSTEM_PROMPT
    from graphrag.prompts.query.global_search_reduce_system_prompt import REDUCE_SYSTEM_PROMPT
    from graphrag.prompts.query.global_search_knowledge_system_prompt import (
        GENERAL_KNOWLEDGE_INSTRUCTION,
    )
    from graphrag.prompts.query.drift_search_system_prompt import DRIFT_LOCAL_SYSTEM_PROMPT, DRIFT_REDUCE_PROMPT
    from graphrag.prompts.query.basic_search_system_prompt import BASIC_SEARCH_SYSTEM_PROMPT

    prompts_dir = WORKSPACE / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    prompt_map = {
        "extract_graph.txt":                         GRAPH_EXTRACTION_PROMPT,
        "summarize_descriptions.txt":                SUMMARIZE_PROMPT,
        "community_report_graph.txt":                COMMUNITY_REPORT_PROMPT,
        "community_report_text.txt":                 COMMUNITY_REPORT_PROMPT,
        "extract_claims.txt":                        EXTRACT_CLAIMS_PROMPT,
        "local_search_system_prompt.txt":            LOCAL_SEARCH_SYSTEM_PROMPT,
        "global_search_map_system_prompt.txt":       MAP_SYSTEM_PROMPT,
        "global_search_reduce_system_prompt.txt":    REDUCE_SYSTEM_PROMPT,
        "global_search_knowledge_system_prompt.txt": GENERAL_KNOWLEDGE_INSTRUCTION,
        "drift_search_system_prompt.txt":            DRIFT_LOCAL_SYSTEM_PROMPT,
        "drift_search_reduce_prompt.txt":            DRIFT_REDUCE_PROMPT,
        "basic_search_system_prompt.txt":            BASIC_SEARCH_SYSTEM_PROMPT,
    }

    written = 0
    for filename, content in prompt_map.items():
        dest = prompts_dir / filename
        if not dest.exists():
            dest.write_text(content)
            written += 1

    console.print(
        f"  Prompts: {len(list(prompts_dir.glob('*.txt')))} prompt files in {prompts_dir}"
        + (f" ({written} newly written)" if written else " (already present)")
    )


def setup_workspace() -> None:
    """
    Prepare the graphrag workspace:
      1. Copy corpus .txt files into input/
      2. Write .env with GRAPHRAG_API_KEY from project environment
      3. Export default prompt files (no extra LLM cost)
    """
    # 1. Corpus input files
    input_dir = WORKSPACE / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Extensions supported by graphrag's markitdown input type.
    # Excludes .json (test question files) and .py (not prose corpus).
    CORPUS_EXTENSIONS = {".txt", ".md", ".html", ".pdf", ".docx"}

    corpus_dir = ROOT / "corpus"
    copied = 0
    for f in corpus_dir.iterdir():
        if f.suffix.lower() in CORPUS_EXTENSIONS:
            dest = input_dir / f.name
            if not dest.exists():
                shutil.copy(f, dest)
                copied += 1

    console.print(
        f"  Corpus: {len(list(input_dir.glob('*')))} files in {input_dir}"
        + (f" ({copied} newly copied)" if copied else " (already present)")
    )

    # 2. .env — write GRAPHRAG_API_KEY from OPENAI_API_KEY if not set
    graphrag_env = WORKSPACE / ".env"
    if not graphrag_env.exists():
        api_key = os.environ.get("OPENAI_API_KEY", "")
        graphrag_env.write_text(f"GRAPHRAG_API_KEY={api_key}\n")
        # Reload so load_config can read it
        load_dotenv(graphrag_env, override=True)
        console.print(f"  Env: wrote {graphrag_env}")
    else:
        load_dotenv(graphrag_env, override=True)
        console.print(f"  Env: {graphrag_env} already exists")

    # 3. Default prompt files
    _write_default_prompts()


def _load_config():
    """Load GraphRAG config from workspace settings.yaml."""
    from graphrag.config.load_config import load_config
    return load_config(root_dir=str(WORKSPACE))


# ── PHASE 1: INDEX ────────────────────────────────────────────────────────────

def reset_output() -> None:
    """
    Delete output/ and cache/ directories to ensure a clean re-index.
    Required when changing embedding models — LanceDB schema is fixed at creation time
    and cannot be updated in-place. Any embedding dimension change (e.g. 1536 → 3072)
    requires a full reset.
    """
    import shutil as _shutil
    for d in ("output", "cache"):
        target = WORKSPACE / d
        if target.exists():
            _shutil.rmtree(target)
            console.print(f"  [dim]Cleared {target}[/dim]")


async def run_index(reset: bool = False) -> None:
    """
    Build the GraphRAG knowledge graph from the corpus.

    Internal pipeline steps:
      1. Chunk documents (chunk_size=1200, overlap=100)
      2. LLM extracts entities and relationships per chunk
      3. Deduplicate entities across the whole corpus into a unified graph
      4. Community detection (Leiden algorithm) clusters related entities
      5. LLM generates community summaries for each cluster
      6. Entity embeddings written to lancedb; graph data to output/*.parquet

    Pass reset=True (--reset flag) to wipe output/ before indexing.
    Required after any change to the embedding model in settings.yaml —
    LanceDB schema is fixed at creation and dimension mismatches will error.
    """
    from graphrag.api import build_index
    from graphrag.config.enums import IndexingMethod

    if reset:
        console.print("\n[dim]Resetting output and cache for clean index...[/dim]")
        reset_output()

    config = _load_config()

    console.print(
        "\n[bold yellow]Starting indexing pipeline "
        "(entity extraction + community detection + summarization)...[/]"
    )
    console.print("[dim]Estimated cost: $0.50–2.00 with gpt-4o-mini on this corpus.[/dim]")

    # build_index is async in graphrag
    results = await build_index(
        config=config,
        method=IndexingMethod.Standard,
        is_update_run=False,
        verbose=False,
    )

    errors = [r for r in results if r.error]
    if errors:
        for r in errors:
            console.print(f"  [red]✗[/] Workflow '{r.workflow}': {r.error}")
    else:
        console.print(f"  [green]✓[/] All {len(results)} indexing workflows completed")

    output_dir = WORKSPACE / "output"
    parquet_files = sorted(output_dir.glob("*.parquet"))
    console.print(f"\n  Output parquet files ({len(parquet_files)}):")
    for f in parquet_files:
        console.print(f"    [dim]{f.name}[/] ({f.stat().st_size // 1024} KB)")


# ── PHASE 2: QUERY ────────────────────────────────────────────────────────────

def load_indexed_data() -> dict:
    """Load parquet files written by the indexing pipeline."""
    output_dir = WORKSPACE / "output"

    required = [
        "entities.parquet", "communities.parquet",
        "community_reports.parquet", "text_units.parquet",
        "relationships.parquet",
    ]
    missing = [f for f in required if not (output_dir / f).exists()]
    if missing:
        console.print(f"[red]Missing index files: {missing}[/]")
        console.print("[yellow]Run --phase index first.[/]")
        sys.exit(1)

    data = {
        "entities":          pd.read_parquet(output_dir / "entities.parquet"),
        "communities":       pd.read_parquet(output_dir / "communities.parquet"),
        "community_reports": pd.read_parquet(output_dir / "community_reports.parquet"),
        "text_units":        pd.read_parquet(output_dir / "text_units.parquet"),
        "relationships":     pd.read_parquet(output_dir / "relationships.parquet"),
    }
    cov = output_dir / "covariates.parquet"
    data["covariates"] = pd.read_parquet(cov) if cov.exists() else None

    console.print(
        f"  Index: {len(data['entities'])} entities, "
        f"{len(data['relationships'])} relationships, "
        f"{len(data['communities'])} communities, "
        f"{len(data['community_reports'])} community reports"
    )
    return data


def _extract_contexts(context) -> list[str]:
    """
    Pull source text from the context object returned by local/global search.
    GraphRAG returns context as a list of DataFrames or a dict.
    We extract text_unit content where available for RAGAS faithfulness eval.
    """
    contexts = []
    if isinstance(context, dict):
        for key in ("sources", "reports", "text_units", "entities"):
            if key in context:
                for item in context[key]:
                    if isinstance(item, dict):
                        text = item.get("content") or item.get("text") or item.get("title", "")
                    else:
                        text = str(item)
                    if text:
                        contexts.append(str(text)[:600])
                if contexts:
                    break
    elif isinstance(context, list):
        for df in context:
            if hasattr(df, "iterrows"):
                for _, row in df.iterrows():
                    text = row.get("content") or row.get("text") or row.get("title", "")
                    if text:
                        contexts.append(str(text)[:600])
            if contexts:
                break
    return contexts or ["[no source context extracted]"]


async def _local_search(question: str, data: dict, config) -> tuple[str, list[str]]:
    """
    Local search — entity-centric graph traversal.
    Matches entities to the query, walks their graph neighborhood,
    collects source text_units from the matched subgraph.
    """
    from graphrag.api import local_search

    response, context = await local_search(
        config=config,
        entities=data["entities"],
        communities=data["communities"],
        community_reports=data["community_reports"],
        text_units=data["text_units"],
        relationships=data["relationships"],
        covariates=data["covariates"],
        community_level=2,
        response_type="Single paragraph",
        query=question,
        verbose=False,
    )
    return str(response), _extract_contexts(context)


async def _global_search(question: str, data: dict, config) -> tuple[str, list[str]]:
    """
    Global search — community-summary map-reduce synthesis.
    Reads LLM-generated community reports, generates partial answers per community,
    synthesises into a final answer. Handles cross-corpus thematic questions that
    no chunk-based system can answer.
    """
    from graphrag.api import global_search

    response, context = await global_search(
        config=config,
        entities=data["entities"],
        communities=data["communities"],
        community_reports=data["community_reports"],
        community_level=2,
        dynamic_community_selection=False,
        response_type="Single paragraph",
        query=question,
        verbose=False,
    )
    return str(response), _extract_contexts(context)


async def run_queries(search_mode: str) -> list[EvalSample]:
    config = _load_config()
    data = load_indexed_data()

    test_path = ROOT / "corpus" / "test_questions.json"
    test_data = json.loads(test_path.read_text())
    console.print(f"\n[bold cyan]Running {len(test_data)} questions ({search_mode} search)...[/]")

    samples = []
    for item in test_data:
        question = item["question"]
        if search_mode == "local":
            answer, contexts = await _local_search(question, data, config)
        else:
            answer, contexts = await _global_search(question, data, config)

        samples.append(EvalSample(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=item.get("ground_truth", ""),
        ))
        console.print(f"  [green]✓[/] {question[:60]}...")

    return samples


# ── DISPLAY + SAVE ────────────────────────────────────────────────────────────

def display_results(
    samples: list[EvalSample],
    scores: dict[str, float],
    experiment_name: str,
) -> None:
    qa_table = Table(title="Per-Question Results", show_lines=True)
    qa_table.add_column("Question", style="cyan", max_width=40)
    qa_table.add_column("Answer", style="white", max_width=60)
    qa_table.add_column("Contexts", style="dim", justify="center")
    for s in samples:
        qa_table.add_row(s.question, s.answer[:120], str(len(s.contexts)))
    console.print(qa_table)

    score_table = Table(
        title=f"RAGAS Scores — {experiment_name} vs Baseline", show_header=True
    )
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


def save_results(
    samples: list[EvalSample],
    scores: dict[str, float],
    search_mode: str,
) -> Path:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    experiment_name = f"exp_04_graphrag_{search_mode}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "experiment": experiment_name,
        "timestamp": timestamp,
        "retrieval_strategy": f"GraphRAG {search_mode} search (microsoft/graphrag v3.x)",
        "search_mode": search_mode,
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

    out_path = results_dir / f"{experiment_name}_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2))
    return out_path


# ── MAIN ──────────────────────────────────────────────────────────────────────

async def async_main(phase: str, search_mode: str, reset: bool = False) -> None:
    if phase in ("index", "all"):
        console.rule("[bold]GraphRAG — Phase 1: Setup + Index (build knowledge graph)")
        console.print(
            "[yellow]⚠  This will make many LLM calls (entity extraction per chunk,\n"
            "   community detection, community summarization).\n"
            "   Estimated cost: $0.50–2.00 with gpt-4o-mini.[/]\n"
        )
        setup_workspace()
        await run_index(reset=reset)
        console.rule("[bold green]Indexing complete — graph saved to graphrag_workspace/output/")

    if phase in ("query", "all"):
        console.rule(f"[bold]GraphRAG — Phase 2: Query ({search_mode} search)")
        desc = (
            "Entity-centric graph traversal — matches entities → walks neighborhood → source text"
            if search_mode == "local"
            else "Community map-reduce synthesis — reads community reports → synthesises across corpus"
        )
        console.print(f"[cyan]{desc}[/]\n")

        samples = await run_queries(search_mode)

        console.print("\n[bold cyan]Running RAGAS evaluation...[/]")
        embedder = SentenceTransformerEmbedder(model_name=cfg.embedding.model_name)
        scores = evaluate_pipeline(samples, embedder=embedder)

        experiment_name = f"exp_04_graphrag_{search_mode}"
        display_results(samples, scores, experiment_name)

        out_path = save_results(samples, scores, search_mode)
        console.print(f"\n[dim]Results saved to: {out_path}[/dim]")
        console.rule(
            f"[bold green]GraphRAG {search_mode} search complete — "
            "check delta vs baseline above"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 04-GraphRAG")
    parser.add_argument(
        "--phase",
        choices=["index", "query", "all"],
        default="query",
        help="index=build knowledge graph (run once), query=evaluate, all=both",
    )
    parser.add_argument(
        "--search",
        choices=["local", "global"],
        default="local",
        help="local=entity graph traversal, global=community summary synthesis",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help=(
            "Wipe output/ and cache/ before indexing. Required when changing the "
            "embedding model in settings.yaml — LanceDB schema is fixed at creation "
            "time and dimension mismatches will error without a reset."
        ),
    )
    args = parser.parse_args()
    asyncio.run(async_main(args.phase, args.search, reset=args.reset))


if __name__ == "__main__":
    main()
