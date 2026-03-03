"""
This example demonstrates how to store and search semantic embeddings using
PostgreSQL + pgvector, accessed via the Supabase async client in a FastAPI app.

Key concepts:
- PostgreSQL remains the primary database; pgvector is a PostgreSQL extension
  that enables vector storage and similarity search.
- Embeddings are stored as fixed-length float vectors inside normal tables.
- Supabase is used as a managed database service; FastAPI interacts with Postgres
  through an async HTTP API client, not direct SQL connections.

Embedding workflow:
- Text is converted into an embedding vector (mocked here for learning purposes).
- The text and its embedding are stored together in the database.
- Semantic search is performed by comparing vectors using pgvector operators
  (cosine distance) inside PostgreSQL.

Query pattern:
- Simple CRUD uses table().insert() / select().
- Vector similarity search is executed via database RPCs, where Postgres performs
  the vector math and ranking.

Important notes:
- mock_embed() exists to teach system architecture without calling a real
  embedding model or external API.
- In production, mock_embed() would be replaced with a real embedding model
  (e.g., OpenAI text-embedding-3-small).

Key takeaway:
- LLMs generate embeddings.
- PostgreSQL stores embeddings.
- pgvector compares embeddings.
- FastAPI only orchestrates the flow.

This pattern forms the foundation of Retrieval-Augmented Generation (RAG)
without introducing a separate vector database.

"""

import random
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Annotated

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from supabase import acreate_client, AsyncClient


class Settings(BaseSettings):
    supabase_url: str
    supabase_secret_key: str

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()


async def get_supabase() -> AsyncGenerator[AsyncClient, None]:
    client = await acreate_client(settings.supabase_url, settings.supabase_secret_key)
    yield client


SupabaseClient = Annotated[AsyncClient, Depends(get_supabase)]


# --- Schemas ---

class MessageWithEmbedding(BaseModel):
    conversation_id: uuid.UUID
    role: str
    content: str

class SimilarMessage(BaseModel):
    id: str
    content: str
    role: str
    similarity: float


# --- Mock embedding function ---

def mock_embed(text: str) -> list[float]:
    """
    Generate a deterministic mock embedding from text.
    In production, replace with: await openai.embeddings.create(input=text, model="text-embedding-3-small")
    """
    random.seed(hash(text) % (2**32))
    return [random.gauss(0, 0.1) for _ in range(1536)]


# --- App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✅ pgvector preview ready")
    print("   Replace mock_embed() with a real embedding model in Phase 5")
    yield


app = FastAPI(title="Ex 6: pgvector Preview", lifespan=lifespan)


@app.post("/messages/embed", status_code=201)
async def store_message_with_embedding(
    data: MessageWithEmbedding,
    client: SupabaseClient,
):
    """Store a message with a mock embedding vector."""
    embedding = mock_embed(data.content)

    result = await (
        client.table("messages")
        .insert({
            "id": str(uuid.uuid4()),
            "conversation_id": str(data.conversation_id),
            "role": data.role,
            "content": data.content,
            "embedding": embedding,
        })
        .execute()
    )
    return {
        "stored": True,
        "id": result.data[0]["id"] if result.data else None,
        "embedding_dims": len(embedding),
        "note": "Mock embedding used. Replace mock_embed() with real model in Phase 5.",
    }


@app.get("/messages/search", response_model=list[SimilarMessage])
async def search_similar_messages(
    query: str,
    client: SupabaseClient,
    match_count: int = 5
):
    """Find messages semantically similar to the query using cosine similarity."""
    query_embedding = mock_embed(query)

    result = await client.rpc(
        "match_messages",
        {
            "query_embedding": query_embedding,
            "match_count": match_count,
        },
    ).execute()

    return [
        SimilarMessage(
            id=row["id"],
            content=row["content"],
            role=row["role"],
            similarity=round(row["similarity"], 4),
        )
        for row in result.data
    ]


@app.get("/pgvector/info")
async def pgvector_info(client: SupabaseClient):
    """Verify pgvector extension and show embedding column info."""
    ext_result = await client.rpc(
        "check_pgvector",
        {}
    ).execute()

    count_result = await (
        client.table("messages")
        .select("id", count="exact")
        .not_.is_("embedding", "null")
        .execute()
    )

    return {
        "messages_with_embeddings": count_result.count,
        "note": "Phase 5 RAG will replace mock_embed() with text-embedding-3-small or similar",
        "similarity_operator": "<=> (cosine distance, lower = more similar)",
    }