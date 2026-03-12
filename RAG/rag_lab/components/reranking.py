from .base import RetrievedChunk


class IdentityReranker:
    """
    Passthrough reranker. Returns chunks in the same order they were retrieved.
    Baseline: no reranking. Swap this out in the Reranking chapter experiment.

    Experiment knobs (future):
      - Replace with CrossEncoderReranker using a sentence-transformers cross-encoder
      - Replace with CohereReranker using the Cohere API
      - Observe precision@k delta vs the identity baseline
    """

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return chunks
