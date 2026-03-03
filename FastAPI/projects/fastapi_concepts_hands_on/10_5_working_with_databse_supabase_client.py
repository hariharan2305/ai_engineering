"""
This example demonstrates using the Supabase *async Python client* in a FastAPI app,
showing how anon and service keys interact with PostgreSQL Row Level Security (RLS).

Key concepts:
- Using Supabase as a managed database service accessed via an async HTTP client
  (not direct SQL connections or SQLAlchemy sessions).
- Creating Supabase clients via FastAPI dependencies (get_supabase_anon, get_supabase_service).
- RLS in action: the same query returns different results depending on the key used.
- When to use keys:
  - anon key → user-facing endpoints, RLS enforced
  - service key → trusted server-side code, RLS bypassed (admin / background jobs)

Important architectural distinction:
- Supabase manages PostgreSQL connections, pooling, and schema lifecycle.
- FastAPI only creates lightweight API clients; no engine, sessionmaker, or table creation
  is needed at application startup.

Key takeaway:
- Anon and service keys access the *same PostgreSQL database*.
- The only difference is whether RLS policies are enforced.
- Security lives in the database, not the application.
  Even if FastAPI code is buggy, RLS prevents accidental data exposure.

"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Annotated

from fastapi import FastAPI, Depends
from pydantic_settings import BaseSettings
from supabase import acreate_client, AsyncClient


class Settings(BaseSettings):
    supabase_url: str
    supabase_anon_key: str
    supabase_secret_key: str

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()


# --- Dependencies ---

async def get_supabase_anon() -> AsyncGenerator[AsyncClient, None]:
    """Anon client — respects Row Level Security (RLS)."""
    client = await acreate_client(settings.supabase_url, settings.supabase_anon_key)
    yield client


async def get_supabase_service() -> AsyncGenerator[AsyncClient, None]:
    """Service client — bypasses RLS. Use only in trusted server-side code."""
    client = await acreate_client(settings.supabase_url, settings.supabase_secret_key)
    yield client


AnonClient = Annotated[AsyncClient, Depends(get_supabase_anon)]
ServiceClient = Annotated[AsyncClient, Depends(get_supabase_service)]


# --- App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✅ Supabase clients ready (created per-request via dependency)")
    yield


app = FastAPI(title="Ex 5: Supabase Async Client + RLS", lifespan=lifespan)


@app.get("/conversations/anon")
async def list_conversations_anon(client: AnonClient):
    """
    Uses the ANON key — subject to RLS.
    Returns [] unless a JWT for an authenticated user is provided in the header.
    This demonstrates RLS silently filtering rows.
    """
    result = await client.table("conversations").select("id, title, model").execute()
    return {
        "key_type": "anon",
        "rls_active": True,
        "note": "Returns [] without a valid user JWT (RLS requires authenticated user)",
        "count": len(result.data),
        "conversations": result.data,
    }


@app.get("/conversations/service")
async def list_conversations_service(client: ServiceClient):
    """
    Uses the SERVICE key — bypasses RLS.
    Returns ALL conversations regardless of owner.
    Use ONLY in trusted server-side contexts (admin, background jobs).
    """
    result = await client.table("conversations").select("id, title, model").execute()
    return {
        "key_type": "service",
        "rls_active": False,
        "note": "Service key bypasses RLS — sees all rows",
        "count": len(result.data),
        "conversations": result.data,
    }


@app.post("/conversations/service", status_code=201)
async def create_conversation_service(
    client: ServiceClient,
    title: str,
    model: str = "claude-3-5-sonnet-20241022"    
):
    """Insert a conversation via supabase-py (service key — no RLS on writes here)."""
    import uuid
    result = await (
        client.table("conversations")
        .insert({
            "id": str(uuid.uuid4()),
            "title": title,
            "model": model,
            # user_id omitted intentionally — shows schema requires it
        })
        .execute()
    )
    return {"created": result.data}


@app.get("/rls-demo")
async def rls_demo(anon: AnonClient, service: ServiceClient):
    """
    Side-by-side comparison: same query, different keys.
    Demonstrates RLS in action without switching endpoints.
    """
    anon_result = await anon.table("conversations").select("id, title").execute()
    service_result = await service.table("conversations").select("id, title").execute()

    return {
        "anon_key_count": len(anon_result.data),
        "service_key_count": len(service_result.data),
        "rls_working": len(anon_result.data) < len(service_result.data),
        "explanation": (
            "anon_key_count < service_key_count proves RLS is filtering rows. "
            "The anon client sees only rows where auth.uid() matches user_id. "
            "With no JWT, auth.uid() is NULL, so no rows match."
        ),
    }