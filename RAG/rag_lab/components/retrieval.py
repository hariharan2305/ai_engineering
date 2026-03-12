from .base import RetrievedChunk
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
