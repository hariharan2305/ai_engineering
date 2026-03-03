"""
This example demonstrates advanced Redis usage in a GenAI backend, showing how
Redis acts as a real-time control and memory layer alongside a traditional database.

Key concepts:
- Redis is used as shared, in-memory state across requests and server instances.
- PostgreSQL (or Supabase) remains the source of truth; Redis holds fast, temporary data.
- The backend orchestrates Redis, database access, and LLM calls to optimize
  latency, cost, and reliability.

What Redis is responsible for in this script:
1. Sliding-window rate limiting
   - Implemented using Redis SORTED SETs with timestamps
   - Enforces global request limits across all API instances
   - Prevents abuse before expensive operations (DB queries or LLM calls)

2. Conversation context “hot cache”
   - Stores only the last N messages of a conversation in Redis
   - Avoids fetching full history from the database on every request
   - Reduces LLM token usage by controlling context window size
   - Uses TTL so inactive conversations automatically expire

Request flow (high-level):
1. Incoming request → rate limit check in Redis (fast, fail-early)
2. Load recent conversation context from Redis (hot path)
3. Fallback to database if cache is missing (cold path)
4. Build prompt and call the LLM (mocked here)
5. Update Redis context immediately for the next request
6. Persist full conversation asynchronously to the database (conceptual)

Lifecycle vs dependencies:
- Redis connection ownership and health checks are handled in FastAPI lifespan
  (startup/shutdown).
- Dependencies are used only to access the shared Redis client per request.

Key takeaways:
- Redis is not a database replacement; it is a real-time memory and coordination layer.
- Redis is ideal for rate limiting, caching, and short-lived conversational state.
- LLMs are stateless; conversation "threads" are reconstructed on every request
  using Redis (working memory) and the database (long-term memory).

This architecture mirrors how ChatGPT-style systems achieve low latency,
controlled costs, and seamless conversational experiences at scale.
"""


import json
import time
import uuid
import os
from contextlib import asynccontextmanager
from typing import Annotated

import redis.asyncio as aioredis
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    rate_limit_requests: int = 5   # max requests
    rate_limit_window: int = 60    # per N seconds
    context_window_size: int = 10  # messages to keep in hot cache
    context_ttl: int = 1800        # 30 minutes

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()

redis_client: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global redis_client
    if redis_client is None:
        redis_client = aioredis.Redis.from_url(settings.redis_url, decode_responses=True)
    return redis_client

RedisClient = Annotated[aioredis.Redis, Depends(get_redis)]


# ─── Sliding Window Rate Limiter ──────────────────────────────────────────────

async def check_rate_limit(r: aioredis.Redis, user_id: str) -> tuple[bool, int]:
    """
    Sliding window rate limiter using Redis SORTED SET.
    Returns (allowed: bool, requests_in_window: int).
    """
    key = f"ratelimit:{user_id}"
    now = time.time()
    window_start = now - settings.rate_limit_window

    pipe = r.pipeline()
    pipe.zremrangebyscore(key, 0, window_start)          # remove old entries
    pipe.zadd(key, {str(uuid.uuid4()): now})             # add current request
    pipe.zcount(key, window_start, "+inf")               # count in window
    pipe.expire(key, settings.rate_limit_window)         # auto-expire key
    results = await pipe.execute()

    request_count = results[2]
    allowed = request_count <= settings.rate_limit_requests
    return allowed, request_count


# ─── Conversation Context Hot Cache ──────────────────────────────────────────

async def get_context_from_cache(
    r: aioredis.Redis, conv_id: str
) -> list[dict] | None:
    """Returns cached context or None if not in Redis."""
    cached = await r.get(f"context:{conv_id}")
    return json.loads(cached) if cached else None


async def set_context_in_cache(
    r: aioredis.Redis, conv_id: str, messages: list[dict]
) -> None:
    """Store context in Redis, keeping only the last N messages."""
    trimmed = messages[-settings.context_window_size:]
    await r.set(f"context:{conv_id}", json.dumps(trimmed), ex=settings.context_ttl)


async def append_to_context(
    r: aioredis.Redis, conv_id: str, role: str, content: str
) -> list[dict]:
    """Append a message to the cached context without hitting PostgreSQL."""
    cached = await r.get(f"context:{conv_id}")
    context = json.loads(cached) if cached else []
    context.append({"role": role, "content": content})
    trimmed = context[-settings.context_window_size:]
    await r.set(f"context:{conv_id}", json.dumps(trimmed), ex=settings.context_ttl)
    return trimmed


# ─── Schemas ─────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    conversation_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    context_size: int
    context_source: str   # "redis_cache" or "database"
    rate_limit_used: int
    rate_limit_max: int


# ─── App ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    r = await get_redis()
    await r.ping()
    print(f"✅ Redis connected (rate limit: {settings.rate_limit_requests} req / {settings.rate_limit_window}s)")
    yield
    if redis_client:
        await redis_client.aclose()


app = FastAPI(title="Ex 8: Rate Limit + Context Cache + Upstash", lifespan=lifespan)


@app.post("/chat/full", response_model=ChatResponse)
async def full_chat(
    request: ChatRequest,
    r: RedisClient,
    x_user_id: str = Header(default="anonymous"),
):
    """
    Full pipeline:
    1. Rate limit check (Redis SORTED SET)
    2. Load conversation context (Redis hot cache → fallback mock)
    3. Append user message + mock LLM reply
    4. Update context cache
    """
    # 1. Rate limit check
    allowed, count = await check_rate_limit(r, x_user_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "requests_in_window": count,
                "limit": settings.rate_limit_requests,
                "window_seconds": settings.rate_limit_window,
                "retry_after": settings.rate_limit_window,
            },
            headers={"Retry-After": str(settings.rate_limit_window)},
        )

    # 2. Load context
    context = await get_context_from_cache(r, request.conversation_id)
    context_source = "redis_cache"

    if context is None:
        # In production: fetch from PostgreSQL
        context = []  # mock: empty history
        context_source = "database_fallback"

    # 3. Build messages for LLM (context + new user message)
    context.append({"role": "user", "content": request.message})

    # Mock LLM call (replace with real provider in Topic 7)
    reply = f"[Mock LLM] Responding to: '{request.message[:60]}' (context_msgs={len(context)-1})"

    context.append({"role": "assistant", "content": reply})

    # 4. Update context cache
    updated_context = await set_context_in_cache(
        r, request.conversation_id, context
    ) or context  # returns None from set, use local variable

    return ChatResponse(
        reply=reply,
        context_size=len(context),
        context_source=context_source,
        rate_limit_used=count,
        rate_limit_max=settings.rate_limit_requests,
    )


@app.get("/rate-limit/status/{user_id}")
async def rate_limit_status(user_id: str, r: RedisClient):
    """Check how many requests a user has made in the current window."""
    key = f"ratelimit:{user_id}"
    now = time.time()
    window_start = now - settings.rate_limit_window

    await r.zremrangebyscore(key, 0, window_start)
    count = await r.zcount(key, window_start, "+inf")

    return {
        "user_id": user_id,
        "requests_in_window": count,
        "limit": settings.rate_limit_requests,
        "window_seconds": settings.rate_limit_window,
        "remaining": max(0, settings.rate_limit_requests - count),
    }


@app.delete("/context/{conv_id}")
async def invalidate_context(conv_id: str, r: RedisClient):
    """Invalidate the context cache for a conversation (e.g., after a conversation reset)."""
    deleted = await r.delete(f"context:{conv_id}")
    return {"invalidated": bool(deleted), "conversation_id": conv_id}