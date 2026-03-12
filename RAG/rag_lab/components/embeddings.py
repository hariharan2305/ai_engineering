from sentence_transformers import SentenceTransformer

from .base import Chunk, EmbeddedChunk


class SentenceTransformerEmbedder:
    """
    Local embedding model via sentence-transformers. No API cost, runs on CPU.

    Experiment knobs:
      - model_name: swap to a larger/multilingual model and observe quality delta
        Common options:
          "all-MiniLM-L6-v2"       — fast, 384-dim, good general purpose (baseline)
          "all-mpnet-base-v2"       — slower, 768-dim, better quality
          "paraphrase-multilingual-MiniLM-L12-v2" — multilingual
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


def embed_chunks(chunks: list[Chunk], embedder: SentenceTransformerEmbedder) -> list[EmbeddedChunk]:
    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)
    return [
        EmbeddedChunk(
            id=chunk.id,
            text=chunk.text,
            doc_id=chunk.doc_id,
            chunk_index=chunk.chunk_index,
            embedding=emb,
            metadata=chunk.metadata,
        )
        for chunk, emb in zip(chunks, embeddings)
    ]


def embed_query(query: str, embedder: SentenceTransformerEmbedder) -> list[float]:
    return embedder.embed([query])[0]
