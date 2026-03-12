from dataclasses import dataclass, field


@dataclass
class Document:
    """Raw document loaded from a file."""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A text segment produced by splitting a Document."""
    id: str
    text: str
    doc_id: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


@dataclass
class EmbeddedChunk:
    """A Chunk with its vector embedding attached."""
    id: str
    text: str
    doc_id: str
    chunk_index: int
    embedding: list[float]
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A Chunk returned from a similarity search, with its relevance score."""
    id: str
    text: str
    doc_id: str
    chunk_index: int
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class RAGResult:
    """The full output of one RAG pipeline pass."""
    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    metadata: dict = field(default_factory=dict)
