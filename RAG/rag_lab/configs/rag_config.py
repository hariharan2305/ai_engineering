from pydantic import BaseModel


class ChunkingConfig(BaseModel):
    chunk_size: int = 512
    overlap: int = 50


class EmbeddingConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"


class VectorDBConfig(BaseModel):
    collection_name: str = "rag_lab"
    persist_dir: str = "./chroma_db"


class RetrievalConfig(BaseModel):
    top_k: int = 5


class GenerationConfig(BaseModel):
    model: str = "gpt-4o-mini"
    max_tokens: int = 512


class EvaluationConfig(BaseModel):
    judge_model: str = "gpt-4o-mini"


class RAGConfig(BaseModel):
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    vectordb: VectorDBConfig = VectorDBConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


# The canonical baseline — all experiments measure delta from this
BASELINE_CONFIG = RAGConfig()
