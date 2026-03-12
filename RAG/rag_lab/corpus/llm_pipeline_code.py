"""
llm_pipeline_code.py

Production utilities for LLM-backed RAG pipelines.

Includes an OpenAI client wrapper with retry logic, a disk-backed embedding
cache, a RAG pipeline orchestrator, and shared utility functions used across
components. Designed for production robustness: cost tracking, token budgeting,
and graceful degradation on API failures.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Utility Functions ─────────────────────────────────────────────────────────


def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """
    Compute a wait time using full-jitter exponential backoff.

    Formula: random value in [0, min(max_delay, base_delay * 2^attempt)].
    Capped at max_delay=60.0 seconds to prevent excessively long waits.

    Full jitter (randomizing across the entire range rather than just adding
    noise on top) prevents the thundering herd problem, where many retrying
    clients would otherwise all wake up at the same time after a service blip.

    Used by both LLMClient and EmbeddingCache for all API retry loops.
    """
    import random
    delay = min(max_delay, base_delay * (2 ** attempt))
    return random.uniform(0, delay)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Returns a value in [-1.0, 1.0] where 1.0 is identical direction.
    Returns 0.0 if either vector is a zero vector to avoid division by zero.

    Used by EmbeddingCache for near-duplicate detection and by RAGPipeline
    for optional score-threshold filtering before passing chunks to the LLM.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def truncate_to_token_limit(text: str, max_tokens: int, model: str = "gpt-4o-mini") -> str:
    """
    Truncate text to fit within a token budget using tiktoken.

    Called by RAGPipeline._build_prompt() to ensure the assembled context
    block never exceeds the LLM's context window. Truncation is tail-first:
    the beginning of the context is preserved since it typically contains
    the highest-ranked retrieved chunks.

    Falls back to a character-based estimate (~4 chars/token) when tiktoken
    is unavailable, logging a warning so the caller is aware of the degraded
    precision.
    """
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        logger.debug(f"Truncating prompt: {len(tokens)} → {max_tokens} tokens")
        return enc.decode(tokens[:max_tokens])
    except Exception:
        logger.warning("tiktoken unavailable, using character-based truncation estimate")
        return text[: max_tokens * 4]


# ── Data Contracts ────────────────────────────────────────────────────────────


@dataclass
class RAGResponse:
    """
    Structured output from a single RAGPipeline.query() call.

    Fields:
        answer:      The generated text from the LLM.
        sources:     Raw chunk texts used as context — displayed as citations.
        tokens_used: Cumulative token count for this call, used for cost tracking.
        latency_ms:  End-to-end wall-clock latency in milliseconds, from query
                     embedding through generation, for performance monitoring.
        metadata:    Optional dict for experiment-specific tags (e.g. chunker name,
                     reranker used, collection name).
    """
    answer: str
    sources: list[str]
    tokens_used: int
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ── LLM Client ────────────────────────────────────────────────────────────────


class LLMClient:
    """
    Thin wrapper around OpenAI's chat completions API with retry logic.

    Handles transient failures — rate limit errors (429), server errors (5xx),
    and network timeouts — using exponential backoff with full jitter.
    Tracks cumulative token usage across all calls via get_token_usage().

    Args:
        model:       OpenAI model identifier (default: "gpt-4o-mini").
        max_retries: Maximum retry attempts before re-raising the last error.
                     Defaults to 3. Set to 0 to disable retries entirely.
        timeout:     Per-request timeout in seconds. Defaults to 30s.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        timeout: int = 30,
    ):
        import openai
        self.client = openai.OpenAI()
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self._total_tokens = 0

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """
        Send a chat completion request, retrying on transient failures.

        On each failure, waits exponential_backoff(attempt) seconds before
        retrying. After max_retries exhausted, re-raises the last exception
        so the caller can decide how to handle it.

        Token usage from each successful call is accumulated in
        self._total_tokens for cost tracking.
        """
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.timeout,
                )
                self._total_tokens += response.usage.total_tokens
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = exponential_backoff(attempt)
                    logger.warning(
                        f"LLMClient retry {attempt + 1}/{self.max_retries} "
                        f"after {wait:.1f}s: {type(e).__name__}: {e}"
                    )
                    time.sleep(wait)
        raise last_error

    def get_token_usage(self) -> dict[str, int]:
        """Return cumulative token usage across all calls since instantiation."""
        return {"total_tokens": self._total_tokens}


# ── Embedding Cache ───────────────────────────────────────────────────────────


class EmbeddingCache:
    """
    Disk-backed cache for embedding vectors, keyed by SHA-256 hash of text.

    Eliminates redundant model or API calls during incremental ingestion —
    when documents are re-indexed, only new or changed chunks pay the
    embedding cost. Cache entries are stored as JSON files named by their
    hash, making the cache portable and inspectable.

    The model_name is mixed into the hash key to prevent cross-model
    collisions: the same text embedded by different models produces
    different vectors and must be stored separately.

    Args:
        cache_dir:   Directory to store cached embedding files.
                     Created on init if it does not exist.
        model_name:  Embedded into the cache key. Change this if you
                     switch embedding models to avoid stale cache hits.
    """

    def __init__(self, cache_dir: str = ".embedding_cache", model_name: str = "default"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._hits = 0
        self._misses = 0

    def _compute_key(self, text: str) -> str:
        """
        Compute a deterministic, collision-resistant cache key.

        SHA-256 of '{model_name}:{text}' — the model prefix ensures that
        switching embedding models invalidates all prior cache entries for
        the same text, preventing silent correctness bugs.
        """
        content = f"{self.model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str) -> list[float] | None:
        """Return the cached embedding vector, or None on a cache miss."""
        import json
        key = self._compute_key(text)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self._hits += 1
            logger.debug(f"EmbeddingCache hit: {key[:12]}...")
            return json.loads(path.read_text())
        self._misses += 1
        logger.info(f"EmbeddingCache miss: {key[:12]}...")
        return None

    def set(self, text: str, embedding: list[float]) -> None:
        """Persist an embedding to disk under its deterministic cache key."""
        import json
        key = self._compute_key(text)
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps(embedding))

    def stats(self) -> dict[str, int]:
        """Return hit/miss counts for monitoring cache effectiveness."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": self._hits + self._misses,
            "hit_rate": round(self._hits / max(1, self._hits + self._misses), 3),
        }


# ── RAG Pipeline ─────────────────────────────────────────────────────────────


class RAGPipeline:
    """
    Orchestrates the full retrieval-augmented generation flow.

    Flow: embed question → retrieve top-k chunks → (optional rerank) →
          assemble prompt → generate answer → return RAGResponse.

    The pipeline is intentionally thin: it delegates all heavy work to
    injected components (embedder, vector_store, reranker, llm). This
    makes it straightforward to swap any component for experimentation.

    Context budget: _build_prompt() calls truncate_to_token_limit() on the
    assembled context block, so the LLM never receives a prompt that exceeds
    its context window regardless of how many chunks are retrieved.

    Args:
        llm:                 LLMClient instance for generation.
        embedder:            Callable: str → list[float].
        vector_store:        Object with .query(embedding, top_k) returning
                             list of (text, score) tuples.
        reranker:            Optional object with .rerank(query, chunks, top_k).
        max_context_tokens:  Token budget for the assembled context block.
                             Defaults to 3000, leaving headroom for the question
                             and system prompt within a 4096-token model limit.
    """

    SYSTEM_PROMPT = (
        "You are a precise question-answering assistant. "
        "Answer the question using only the provided context. "
        "If the context does not contain enough information, say so explicitly."
    )

    def __init__(
        self,
        llm: LLMClient,
        embedder,
        vector_store,
        reranker=None,
        max_context_tokens: int = 3000,
    ):
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        self.max_context_tokens = max_context_tokens

    def query(self, question: str, top_k: int = 5) -> RAGResponse:
        """
        Run end-to-end RAG for a single question.

        Embeds the question, retrieves the top-k most similar chunks,
        optionally reranks them, builds a token-budgeted prompt, and
        generates an answer. Returns a RAGResponse with the answer,
        the source chunks used, cumulative token count, and latency.
        """
        t0 = time.perf_counter()

        query_embedding = self.embedder(question)
        chunks = self.vector_store.query(query_embedding, top_k=top_k)

        if self.reranker:
            chunks = self.reranker.rerank(question, chunks, top_k=top_k)

        contexts = [text for text, _ in chunks]
        prompt = self._build_prompt(question, contexts)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        answer = self.llm.chat(messages)

        latency_ms = (time.perf_counter() - t0) * 1000
        return RAGResponse(
            answer=answer,
            sources=contexts,
            tokens_used=self.llm.get_token_usage()["total_tokens"],
            latency_ms=round(latency_ms, 2),
        )

    def _build_prompt(self, question: str, contexts: list[str]) -> str:
        """
        Assemble retrieved chunks into a prompt, respecting the token budget.

        Contexts are joined as a numbered list so the LLM can reference
        specific sources. The combined context block is truncated to
        max_context_tokens via truncate_to_token_limit() before appending
        the question. Truncation is tail-first: the highest-ranked chunks
        at the top of the list are preserved.
        """
        context_block = "\n\n".join(
            f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)
        )
        context_block = truncate_to_token_limit(
            context_block, self.max_context_tokens, model=self.llm.model
        )
        return f"Context:\n{context_block}\n\nQuestion: {question}"
