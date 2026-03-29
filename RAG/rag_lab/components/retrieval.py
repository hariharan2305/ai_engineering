import re
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from .base import Chunk, RetrievedChunk
from .embeddings import SentenceTransformerEmbedder, embed_query
from .vectordb import ChromaVectorDB


class DenseRetriever:
    """
    Retrieves chunks by embedding the query and running cosine similarity search.
    Baseline strategy — pure dense retrieval, no hybrid, no reranking.

    Experiment knobs:
      - top_k: retrieve more candidates before reranking (set higher when adding a reranker)
    """

    def __init__(self, embedder: SentenceTransformerEmbedder, vectordb: ChromaVectorDB):
        self.embedder = embedder
        self.vectordb = vectordb

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_embedding = embed_query(query, self.embedder)
        return self.vectordb.search(query_embedding, top_k=top_k)


class BM25Retriever:
    """
    Sparse keyword retriever using BM25Okapi (rank_bm25).

    No embeddings or vector DB involved — operates purely on term frequencies.
    Strengths: exact keyword matches, rare technical terms, product codes, proper nouns.
    Weaknesses: no semantic understanding; "fast" and "quick" are unrelated tokens.

    Usage:
        retriever = BM25Retriever()
        retriever.index(chunks)
        results = retriever.retrieve(query, top_k=5)
    """

    def __init__(self):
        self._bm25 = None
        self._chunks: list[Chunk] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # Lowercase, strip punctuation, split on whitespace
        return re.sub(r"[^\w\s]", "", text.lower()).split()

    def index(self, chunks: list[Chunk]) -> None:
        """Build the BM25 index from a list of Chunk objects."""
        from rank_bm25 import BM25Okapi
        self._chunks = chunks
        tokenized_corpus = [self._tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self._bm25 is None:
            raise RuntimeError("Call index() before retrieve()")
        import numpy as np
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(
                id=self._chunks[i].id,
                text=self._chunks[i].text,
                doc_id=self._chunks[i].doc_id,
                chunk_index=self._chunks[i].chunk_index,
                score=float(scores[i]),
                metadata=self._chunks[i].metadata,
            )
            for i in top_indices
        ]


class HybridRetriever:
    """
    Hybrid retriever: BM25 (sparse) + Dense (vector), fused with Reciprocal Rank Fusion.

    How it works:
      1. Both retrievers run independently on the same query, each returning top_k candidates.
      2. RRF scores each document based on its rank position in each list:
             rrf_score(doc) = Σ  1 / (k + rank_i)   [k=60, rank is 1-based]
         Scale independence: BM25 scores and cosine similarities live on different scales —
         RRF only uses rank order, so no normalization needed.
      3. The union of both lists is re-ranked by RRF score; top_k are returned.

    Experiment knob:
      - rrf_k: higher values flatten rank differences (less winner-takes-all).
                k=60 is the standard default from the original RRF paper.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        rrf_k: int = 60,
    ):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k

    def _reciprocal_rank_fusion(
        self,
        ranked_lists: list[list[RetrievedChunk]],
    ) -> list[RetrievedChunk]:
        """Fuse multiple ranked lists into one using RRF. Returns deduped, re-ranked results."""
        rrf_scores: dict[str, float] = {}
        chunks_by_id: dict[str, RetrievedChunk] = {}

        for ranked in ranked_lists:
            for rank, chunk in enumerate(ranked, start=1):
                rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (self.rrf_k + rank)
                chunks_by_id[chunk.id] = chunk  # last write wins (same chunk either way)

        sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
        return [
            RetrievedChunk(
                id=chunks_by_id[cid].id,
                text=chunks_by_id[cid].text,
                doc_id=chunks_by_id[cid].doc_id,
                chunk_index=chunks_by_id[cid].chunk_index,
                score=rrf_scores[cid],   # RRF score replaces original similarity score
                metadata=chunks_by_id[cid].metadata,
            )
            for cid in sorted_ids
        ]

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        # Fetch more candidates from each retriever so fusion has a richer pool
        candidate_k = top_k * 2
        bm25_results = self.bm25.retrieve(query, top_k=candidate_k)
        dense_results = self.dense.retrieve(query, top_k=candidate_k)
        fused = self._reciprocal_rank_fusion([bm25_results, dense_results])
        return fused[:top_k]


class MultiQueryRetriever:
    """
    Query-side retrieval improvement using LLM-generated rephrasings.

    How it works:
      1. An LLM generates `num_queries` alternative phrasings of the original query.
         Each rephrasing explores a different vocabulary region of the embedding space.
      2. All queries (original + rephrasings) are run against the base retriever in parallel.
      3. The N result lists are fused with RRF — chunks appearing across multiple query
         results get boosted (cross-query consensus = stronger relevance signal).
      4. Top-k from the fused list are returned.

    Why this helps:
      Dense retrieval encodes one specific phrasing. If the corpus uses different vocabulary
      for the same concept, that chunk scores low. Rephrasings cover those vocabulary gaps
      without changing the retriever or the index at all.

    Experiment knobs:
      - num_queries: more rephrasings = better coverage, higher LLM cost + latency
      - base_retriever: wraps any retriever (Dense, BM25, Hybrid) — isolates the query-side effect
      - rrf_k: same RRF dampening constant (default 60)
    """

    _REPHRASE_PROMPT = """Generate {n} different ways to ask the following question.
Each rephrasing should preserve the original intent but use different vocabulary or framing.
Return only the rephrased questions, one per line, no numbering or extra text.

Original question: {query}"""

    def __init__(
        self,
        base_retriever,
        num_queries: int = 3,
        model: str = "gpt-4o-mini",
        rrf_k: int = 60,
    ):
        self.base_retriever = base_retriever
        self.num_queries = num_queries
        self.rrf_k = rrf_k
        self._client = OpenAI()
        self._model = model

    def _generate_queries(self, query: str) -> list[str]:
        """Ask LLM for N rephrasings. Always prepends the original query."""
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": self._REPHRASE_PROMPT.format(n=self.num_queries, query=query),
                }
            ],
        )
        rephrasings = [
            line.strip()
            for line in response.choices[0].message.content.strip().splitlines()
            if line.strip()
        ]
        # print(rephrasings)  # Debug: see the generated rephrasings in the console
        # Original always included — ensures we never regress below base retriever quality
        return [query] + rephrasings[: self.num_queries]

    def _rrf_fuse(self, ranked_lists: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
        rrf_scores: dict[str, float] = {}
        chunks_by_id: dict[str, RetrievedChunk] = {}

        for ranked in ranked_lists:
            for rank, chunk in enumerate(ranked, start=1):
                rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (self.rrf_k + rank)
                chunks_by_id[chunk.id] = chunk

        sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
        return [
            RetrievedChunk(
                id=chunks_by_id[cid].id,
                text=chunks_by_id[cid].text,
                doc_id=chunks_by_id[cid].doc_id,
                chunk_index=chunks_by_id[cid].chunk_index,
                score=rrf_scores[cid],
                metadata=chunks_by_id[cid].metadata,
            )
            for cid in sorted_ids
        ]

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        queries = self._generate_queries(query)
        # print(f"Generated {len(queries)-1} rephrasings for query: '{query}'")  # Debug log
        # print("All query variants:")  # Debug log
        # for q in queries:
        #     print(f"  - {q}")
        candidate_k = top_k * 2

        # All retrieval calls run in parallel — one thread per query variant
        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            all_results = list(
                executor.map(lambda q: self.base_retriever.retrieve(q, top_k=candidate_k), queries)
            )

        fused = self._rrf_fuse(all_results)
        return fused[:top_k]


# ── LangChain MultiQueryRetriever wrapper ─────────────────────────────────────

class _DenseRetrieverAdapter:
    """
    Adapts our DenseRetriever to LangChain's BaseRetriever interface.

    LangChain's MultiQueryRetriever expects a BaseRetriever (Pydantic model) that
    returns list[Document]. This adapter translates both directions:
      - Incoming: LangChain calls _get_relevant_documents(query) → we call DenseRetriever.retrieve()
      - Outgoing: our RetrievedChunk → LangChain Document (chunk metadata preserved in doc.metadata)
    """

    def __new__(cls, dense_retriever, top_k: int = 10):
        # Lazily import LangChain types — keeps the module importable even if langchain_classic
        # is not installed, and avoids polluting the module namespace.
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.documents import Document
        from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
        from pydantic import ConfigDict

        class _Adapter(BaseRetriever):
            # arbitrary_types_allowed so Pydantic accepts our plain DenseRetriever instance
            model_config = ConfigDict(arbitrary_types_allowed=True)

            _dense: object
            _k: int

            def __init__(self, dense, k):
                super().__init__()
                # Use object.__setattr__ to bypass Pydantic field validation for
                # private attributes that aren't declared as model fields
                object.__setattr__(self, "_dense", dense)
                object.__setattr__(self, "_k", k)

            def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> list[Document]:
                chunks = self._dense.retrieve(query, top_k=self._k)
                return [
                    Document(
                        page_content=chunk.text,
                        metadata={
                            **chunk.metadata,
                            "_chunk_id": chunk.id,
                            "_doc_id": chunk.doc_id,
                            "_chunk_index": chunk.chunk_index,
                        },
                    )
                    for chunk in chunks
                ]

        return _Adapter(dense_retriever, top_k)


class LangChainMultiQueryRetriever:
    """
    Wraps langchain_classic.retrievers.MultiQueryRetriever with data type translation.

    How it works (what LangChain does under the hood):
      1. from_llm() wires an LLMChain using the default prompt:
             "Generate 3 different versions of the given question..."
         The LLM output is parsed line-by-line into separate query strings.
      2. Each query is sent to the base retriever; results are collected.
      3. Deduplication by page_content — no RRF, just unique doc union.
         (This is the key difference from our custom implementation which uses RRF.)
      4. include_original=True ensures the original query is always included.

    Difference vs CustomMultiQueryRetriever:
      - LangChain uses simple union + dedup (no RRF cross-query scoring)
      - Our custom implementation uses RRF (chunks appearing in multiple lists score higher)
      - LangChain doesn't expose similarity scores on returned docs (score=1.0 placeholder)

    Data translation:
      DenseRetriever → _DenseRetrieverAdapter (our chunks → LC Documents)
      LC Documents → RetrievedChunk (LC page_content + metadata → our dataclass)
    """

    def __init__(
        self,
        base_retriever,
        num_queries: int = 3,
        model: str = "gpt-4o-mini",
    ):
        from langchain_classic.retrievers import MultiQueryRetriever as _LCRetriever
        from langchain_openai import ChatOpenAI

        adapter = _DenseRetrieverAdapter(base_retriever, top_k=num_queries * 10)
        llm = ChatOpenAI(model=model, temperature=0)
        self._lc_retriever = _LCRetriever.from_llm(
            retriever=adapter,
            llm=llm,
            include_original=True,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        docs = self._lc_retriever.invoke(query)
        results = []
        for doc in docs[:top_k]:
            results.append(RetrievedChunk(
                id=doc.metadata.get("_chunk_id", doc.page_content[:32]),
                text=doc.page_content,
                doc_id=doc.metadata.get("_doc_id", ""),
                chunk_index=doc.metadata.get("_chunk_index", 0),
                score=1.0,  # LC MultiQueryRetriever doesn't expose per-doc scores
                metadata={k: v for k, v in doc.metadata.items() if not k.startswith("_")},
            ))
        return results


# ── LlamaIndex QueryFusionRetriever wrapper ───────────────────────────────────

class LlamaIndexMultiQueryRetriever:
    """
    Wraps llama_index.core.retrievers.QueryFusionRetriever with data type translation.

    How it works (what LlamaIndex does under the hood):
      1. QueryFusionRetriever generates num_queries alternative queries using an LLM
         and the built-in QUERY_GEN_PROMPT.
      2. All queries are run against the provided retrievers list (one retriever here).
      3. Results are fused using the specified mode:
           RECIPROCAL_RANK — RRF fusion (same as our custom implementation)
           SIMPLE          — union + dedup, highest score wins
           RELATIVE_SCORE  — score normalization across lists before merge
      4. similarity_top_k controls the final output size.

    Key difference from LangChain wrapper:
      - LlamaIndex exposes multiple fusion strategies via FUSION_MODES enum
      - We use RECIPROCAL_RANK to match our custom RRF implementation
      - use_async=False keeps it synchronous (simpler for lab bench usage)
      - NodeWithScore carries a real score from the base retriever (cosine similarity)

    Data translation:
      DenseRetriever → _DenseRetrieverLIAdapter (our chunks → LI NodeWithScore)
      LI NodeWithScore → RetrievedChunk (LI node.text + metadata → our dataclass)
    """

    def __init__(
        self,
        base_retriever,
        num_queries: int = 3,
        model: str = "gpt-4o-mini",
    ):
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
        from langchain_openai import ChatOpenAI

        adapter = _DenseRetrieverLIAdapter(base_retriever)
        # QueryFusionRetriever accepts LangChain BaseLanguageModel directly via resolve_llm
        llm = ChatOpenAI(model=model, temperature=0)
        self._li_retriever = QueryFusionRetriever(
            retrievers=[adapter],
            llm=llm,
            mode=FUSION_MODES.RECIPROCAL_RANK,
            num_queries=num_queries,
            use_async=False,
            verbose=False,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        nodes = self._li_retriever.retrieve(query)
        results = []
        for nws in nodes[:top_k]:
            meta = nws.node.metadata
            results.append(RetrievedChunk(
                id=meta.get("_chunk_id", nws.node.id_),
                text=nws.node.text,
                doc_id=meta.get("_doc_id", ""),
                chunk_index=meta.get("_chunk_index", 0),
                score=nws.score or 0.0,
                metadata={k: v for k, v in meta.items() if not k.startswith("_")},
            ))
        return results


def _DenseRetrieverLIAdapter(dense_retriever):
    """
    Adapts our DenseRetriever to LlamaIndex's BaseRetriever interface.

    LlamaIndex expects _retrieve(QueryBundle) → List[NodeWithScore].
    Translates our RetrievedChunk → TextNode + NodeWithScore, preserving
    chunk metadata (id, doc_id, chunk_index) for the reverse translation.
    """
    from llama_index.core.base.base_retriever import BaseRetriever
    from llama_index.core.schema import QueryBundle, TextNode, NodeWithScore

    class _Adapter(BaseRetriever):
        def __init__(self, dense):
            super().__init__()
            self._dense = dense

        def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
            chunks = self._dense.retrieve(query_bundle.query_str, top_k=20)
            return [
                NodeWithScore(
                    node=TextNode(
                        id_=chunk.id,
                        text=chunk.text,
                        metadata={
                            **chunk.metadata,
                            "_chunk_id": chunk.id,
                            "_doc_id": chunk.doc_id,
                            "_chunk_index": chunk.chunk_index,
                        },
                    ),
                    score=chunk.score,
                )
                for chunk in chunks
            ]

    return _Adapter(dense_retriever)


# ── HyDE Retriever ────────────────────────────────────────────────────────────

class HyDERetriever:
    """
    Hypothetical Document Embeddings (HyDE) retriever.

    The representation alignment problem:
      Standard dense retrieval embeds the *question* and searches for similar chunks.
      But questions and answers are representationally asymmetric — they encode into
      different regions of embedding space even when semantically related.
      A question is a request for information; a document chunk is information itself.

    HyDE's fix:
      1. Ask an LLM to generate a hypothetical answer to the question.
         The answer doesn't need to be factually correct — it just needs the right
         shape and vocabulary to point the embedding vector toward the correct
         neighborhood in the corpus.
      2. Embed the hypothetical answer (not the original query).
      3. Search the vector index with that embedding.
      Result: answer-to-answer similarity search instead of question-to-answer.
              Both the hypothetical and real chunks live in the same region of space.

    Risk:
      If the LLM generates a hallucinated answer with wrong domain terminology,
      the embedding points to the wrong neighborhood. Higher risk on highly
      specialized domains where the LLM has weak training coverage.

    Experiment knob:
      - model: the LLM used to generate the hypothetical answer
    """

    _HYDE_PROMPT = """Write a short passage (2-4 sentences) that directly answers the following question.
Write it as if it were an excerpt from a technical document or article.
Do not say 'I' or reference being an AI — write it as factual prose.

Question: {query}"""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        model: str = "gpt-4o-mini",
    ):
        self._dense = dense_retriever
        self._client = OpenAI()
        self._model = model

    def _generate_hypothetical_answer(self, query: str) -> str:
        """Generate a hypothetical answer passage for the query."""
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": self._HYDE_PROMPT.format(query=query),
                }
            ],
        )
        return response.choices[0].message.content.strip()

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        hypothetical_answer = self._generate_hypothetical_answer(query)
        # Embed the hypothetical answer — not the original query
        hyp_embedding = embed_query(hypothetical_answer, self._dense.embedder)
        return self._dense.vectordb.search(hyp_embedding, top_k=top_k)


# ── Query Decomposition Retriever ─────────────────────────────────────────────

class QueryDecompositionRetriever:
    """
    Parallel query decomposition retriever.

    Problem it solves (different from multi-query):
      Multi-query generates rephrasings of the same question — covers vocabulary gaps.
      Query decomposition breaks a complex question into independent sub-questions —
      covers knowledge gaps where a single retrieval cannot surface context for
      all parts of the question simultaneously.

      Example: "How does BM25 handle term frequency differently from TF-IDF,
                and why does that make it better for longer documents?"
      → sub-q1: "How does BM25 handle term frequency differently from TF-IDF?"
      → sub-q2: "Why is BM25 better than TF-IDF for longer documents?"

    How it works:
      1. LLM decomposes the original query into max_sub_queries independent sub-questions.
      2. All sub-questions are sent to the base retriever in parallel (ThreadPoolExecutor).
      3. All result lists are fused with RRF — chunks relevant across multiple sub-questions
         score higher (cross-sub-question consensus = stronger relevance signal).
      4. Top-k from the fused list are returned to the generator.

    Parallel vs Sequential:
      This implementation uses parallel decomposition — sub-questions are independent
      and retrieved simultaneously. Sequential decomposition (where answer to sub-q1
      informs sub-q2) is needed for multi-hop reasoning but cannot be parallelized.
      Parallel covers the majority of production cases at lower latency cost.

    Experiment knobs:
      - max_sub_queries: how many sub-questions to generate (2-4 is typical)
      - base_retriever: wraps any retriever; isolates the decomposition effect
    """

    _DECOMPOSE_PROMPT = """Break the following question into {n} simpler, independent sub-questions.
Each sub-question should target a distinct piece of information needed to answer the original.
Return only the sub-questions, one per line, no numbering or extra text.
If the question is already simple and cannot be meaningfully decomposed, return it as-is on a single line.

Question: {query}"""

    def __init__(
        self,
        base_retriever,
        max_sub_queries: int = 3,
        model: str = "gpt-4o-mini",
        rrf_k: int = 60,
    ):
        self._base = base_retriever
        self._max_sub = max_sub_queries
        self._rrf_k = rrf_k
        self._client = OpenAI()
        self._model = model

    def _decompose(self, query: str) -> list[str]:
        """Ask the LLM to break the query into independent sub-questions."""
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": self._DECOMPOSE_PROMPT.format(n=self._max_sub, query=query),
                }
            ],
        )
        sub_questions = [
            line.strip()
            for line in response.choices[0].message.content.strip().splitlines()
            if line.strip()
        ]
        return sub_questions[: self._max_sub]

    def _rrf_fuse(self, ranked_lists: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
        rrf_scores: dict[str, float] = {}
        chunks_by_id: dict[str, RetrievedChunk] = {}

        for ranked in ranked_lists:
            for rank, chunk in enumerate(ranked, start=1):
                rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1.0 / (self._rrf_k + rank)
                chunks_by_id[chunk.id] = chunk

        sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
        return [
            RetrievedChunk(
                id=chunks_by_id[cid].id,
                text=chunks_by_id[cid].text,
                doc_id=chunks_by_id[cid].doc_id,
                chunk_index=chunks_by_id[cid].chunk_index,
                score=rrf_scores[cid],
                metadata=chunks_by_id[cid].metadata,
            )
            for cid in sorted_ids
        ]

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        sub_questions = self._decompose(query)
        candidate_k = top_k * 2

        # All sub-question retrievals run in parallel
        with ThreadPoolExecutor(max_workers=len(sub_questions)) as executor:
            all_results = list(
                executor.map(
                    lambda q: self._base.retrieve(q, top_k=candidate_k),
                    sub_questions,
                )
            )

        fused = self._rrf_fuse(all_results)
        return fused[:top_k]


# ── Parent Document Retriever — Qdrant native ─────────────────────────────────

class QdrantParentRetriever:
    """
    Parent document retrieval using Qdrant as the single store.

    Architecture:
      Index time:
        - Split corpus into parent chunks (large, e.g. 512 tokens)
        - Split each parent into child chunks (small, e.g. 128 tokens)
        - Embed child chunks → upsert into Qdrant with parent_text + parent_id in payload
        - No separate doc store needed — Qdrant payload carries the parent text

      Query time:
        - Embed query → search Qdrant for top-k child matches
        - Extract parent_text directly from each hit's payload
        - Deduplicate by parent_id (multiple children may share a parent)
        - Return parent texts as RetrievedChunk objects

    Why Qdrant for this:
      Qdrant's payload field stores arbitrary JSON alongside each vector.
      Storing parent text in payload means retrieval is a single network call —
      no secondary doc store lookup needed. Practical for corpora where
      parent chunks are small enough to fit comfortably in payload (~1-2KB).

    Requires: Qdrant running on localhost:6333
      docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
    """

    def __init__(
        self,
        collection_name: str,
        embedder: SentenceTransformerEmbedder,
        embedding_dim: int = 384,
        qdrant_url: str = "http://localhost:6333",
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance

        self._embedder = embedder
        self._collection = collection_name
        self._client = QdrantClient(url=qdrant_url)

        existing = [c.name for c in self._client.get_collections().collections]
        if collection_name in existing:
            self._client.delete_collection(collection_name)
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )

    def index(self, parent_chunks: list, child_chunks_by_parent: dict) -> int:
        """
        Upsert child embeddings into Qdrant with parent text in payload.

        Args:
            parent_chunks: list[Chunk] — large chunks; text sent to generator
            child_chunks_by_parent: dict[parent_id → list[Chunk]] — small chunks for retrieval
        Returns:
            total child chunks indexed
        """
        from qdrant_client.models import PointStruct

        parent_text_by_id = {p.id: p.text for p in parent_chunks}
        points = []
        point_id = 0

        for parent_id, children in child_chunks_by_parent.items():
            parent_text = parent_text_by_id[parent_id]
            for child in children:
                embedding = embed_query(child.text, self._embedder)
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "parent_id":   parent_id,
                        "parent_text": parent_text,
                        "child_text":  child.text,
                        "doc_id":      child.doc_id,
                        "chunk_index": child.chunk_index,
                    },
                ))
                point_id += 1

        batch_size = 100
        for i in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=self._collection,
                wait=True,
                points=points[i: i + batch_size],
            )
        return len(points)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_embedding = embed_query(query, self._embedder)
        hits = self._client.query_points(
            collection_name=self._collection,
            query=query_embedding,
            with_payload=True,
            limit=top_k * 4,
        ).points

        # Deduplicate by parent_id — keep highest-scoring child per parent
        seen_parents: set[str] = set()
        results: list[RetrievedChunk] = []
        for hit in hits:
            parent_id = hit.payload["parent_id"]
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                results.append(RetrievedChunk(
                    id=parent_id,
                    text=hit.payload["parent_text"],
                    doc_id=hit.payload["doc_id"],
                    chunk_index=hit.payload["chunk_index"],
                    score=hit.score,
                    metadata={},
                ))
            if len(results) == top_k:
                break
        return results


# ── Parent Document Retriever — ChromaDB + Redis ──────────────────────────────

class RedisParentRetriever:
    """
    Parent document retrieval using ChromaDB for children + Redis for parent doc store.

    Architecture:
      Index time:
        - Split corpus into parent chunks (large) → store in Redis keyed by parent_id
        - Split each parent into child chunks (small) → embed → store in ChromaDB
          with parent_id in metadata
        - Two stores, two responsibilities: ChromaDB for vector search, Redis for text

      Query time:
        - Embed query → search ChromaDB for top-k child matches
        - Extract parent_ids from ChromaDB metadata
        - Batch fetch parent texts from Redis via pipeline (O(1) per key)
        - Deduplicate by parent_id → return parent texts as RetrievedChunk objects

    Why this architecture matters in production:
      ChromaDB handles high-dimensional similarity search.
      Redis handles fast key-value lookup at scale — sub-millisecond reads,
      horizontal scalability, persistence options. The two stores scale
      independently: grow the Redis cluster for larger corpora without
      rebuilding the vector index, and vice versa.

    Requires: Redis running on localhost:6379
      docker run -p 6379:6379 redis
    """

    def __init__(
        self,
        vectordb: ChromaVectorDB,
        embedder: SentenceTransformerEmbedder,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
    ):
        import redis as redis_lib
        self._vectordb = vectordb
        self._embedder = embedder
        self._redis = redis_lib.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True,
        )

    def index(self, parent_chunks: list, child_embedded_chunks: list) -> int:
        """
        Store parent texts in Redis and child embeddings in ChromaDB.

        Args:
            parent_chunks: list[Chunk] — large chunks stored in Redis
            child_embedded_chunks: list[EmbeddedChunk] — small chunks with parent_id in metadata
        Returns:
            total child chunks indexed
        """
        pipe = self._redis.pipeline()
        for parent in parent_chunks:
            pipe.set(f"parent:{parent.id}", parent.text)
        pipe.execute()

        self._vectordb.add_chunks(child_embedded_chunks)
        return len(child_embedded_chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_embedding = embed_query(query, self._embedder)
        child_hits = self._vectordb.search(query_embedding, top_k=top_k * 4)

        # Deduplicate by parent_id — keep highest-scoring child per parent
        seen_parents: set[str] = set()
        parent_ids_ordered: list[str] = []
        scores_by_parent: dict[str, float] = {}

        for hit in child_hits:
            parent_id = hit.metadata.get("parent_id", hit.doc_id)
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent_ids_ordered.append(parent_id)
                scores_by_parent[parent_id] = hit.score
            if len(parent_ids_ordered) == top_k:
                break

        # Batch fetch parent texts from Redis in one pipeline round-trip
        pipe = self._redis.pipeline()
        for pid in parent_ids_ordered:
            pipe.get(f"parent:{pid}")
        parent_texts = pipe.execute()

        results = []
        for pid, text in zip(parent_ids_ordered, parent_texts):
            if text:
                results.append(RetrievedChunk(
                    id=pid,
                    text=text,
                    doc_id=pid,
                    chunk_index=0,
                    score=scores_by_parent[pid],
                    metadata={},
                ))
        return results
