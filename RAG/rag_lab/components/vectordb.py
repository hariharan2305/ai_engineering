from pathlib import Path

import chromadb

from .base import EmbeddedChunk, RetrievedChunk


class ChromaVectorDB:
    """
    Local persistent vector store via ChromaDB. No server required.

    Experiment knobs:
      - distance metric: cosine (default) vs l2 vs ip — observe recall differences
      - collection_name: use different names to maintain parallel experiment indexes
      - persist_dir: where the index is stored on disk
    """

    def __init__(self, collection_name: str = "rag_lab", persist_dir: str = "./chroma_db"):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def add_chunks(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return
        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=[c.embedding for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {**c.metadata, "doc_id": c.doc_id, "chunk_index": c.chunk_index}
                for c in chunks
            ],
        )

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            chunks.append(RetrievedChunk(
                id=results["ids"][0][i],
                text=results["documents"][0][i],
                doc_id=meta.get("doc_id", ""),
                chunk_index=meta.get("chunk_index", 0),
                score=1.0 - results["distances"][0][i],  # cosine distance -> similarity
                metadata=meta,
            ))
        return chunks

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection. Use at the start of each experiment for a clean run."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            configuration={"hnsw": {"space": "cosine"}},
        )
