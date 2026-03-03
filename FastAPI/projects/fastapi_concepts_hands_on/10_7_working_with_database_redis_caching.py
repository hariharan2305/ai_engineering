"""
This example demonstrates using Redis as an in-memory cache for LLM responses
in an async FastAPI application.

Key concepts:
- Redis is a fast, in-memory key-value store used for temporary data, not a
  primary database or source of truth.
- The cache-aside pattern is used:
  1) Check Redis for a cached response
  2) If found (cache HIT), return immediately
  3) If not found (cache MISS), call the LLM and store the result in Redis

Why Redis is useful for LLM systems:
- LLM calls are slow and expensive
- Identical prompts are often repeated
- Redis avoids repeated LLM calls by caching deterministic responses

Important design choices:
- Only temperature=0 (deterministic) responses are cached
- Cache keys are deterministic hashes of model + messages + temperature
- Cached entries use a TTL so Redis automatically evicts old data

Lifecycle vs dependencies:
- Redis connection ownership is handled in FastAPI lifespan (startup/shutdown)
- A health check (PING) ensures Redis is available before serving traffic
- Dependencies are used only to access the shared Redis client per request

Key takeaway:
- Redis acts as short-term memory
- PostgreSQL (or another DB) remains long-term memory
- Redis exists to avoid repeating expensive LLM work

This pattern is foundational for building scalable, cost-efficient GenAI APIs.

"""

import asyncio
import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

import redis.asyncio as aioredis
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from pydantic_settings import BaseSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()

# --- Redis connection pool (single instance, shared across requests) ---
redis_pool: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global redis_pool
    if redis_pool is None:
        redis_pool = aioredis.Redis.from_url(settings.redis_url, decode_responses=True)
    return redis_pool


RedisClient = Annotated[aioredis.Redis, Depends(get_redis)]


# --- Cache key ---

def make_cache_key(model: str, messages: list[dict], temperature: float = 0.0) -> str:
    """
    Deterministic cache key. Only use for temperature=0 (deterministic) calls.
    Same inputs → same SHA256 → same cache key.
    """
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature},
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"llm:cache:{digest}"


# --- Simulated LLM call ---

async def simulate_llm_call(model: str, messages: list[dict]) -> dict:
    """Simulate an LLM API call with realistic latency."""
    await asyncio.sleep(0.5)  # simulate 500ms LLM response time
    last_msg = messages[-1]["content"] if messages else ""
    return {
        "content": f"[{model}] Response to: '{last_msg[:50]}'",
        "tokens": len(last_msg) // 4 + 20,
        "model": model,
    }


# --- Cache-aside pattern ---

async def cached_llm_call(
    r: aioredis.Redis,
    model: str,
    messages: list[dict],
    ttl: int = 86400,  # 24 hours default
) -> tuple[dict, bool]:
    """
    Returns (response, cache_hit).
    cache_hit=True means response came from Redis, not LLM.
    """
    key = make_cache_key(model, messages)

    # Try cache first
    cached = await r.get(key)
    if cached:
        logger.info("Cache HIT  | key=%s", key[:20] + "...")
        return json.loads(cached), True

    # Cache miss: call LLM
    logger.info("Cache MISS | key=%s — calling LLM", key[:20] + "...")
    response = await simulate_llm_call(model, messages)

    # Store in cache with TTL
    await r.set(key, json.dumps(response), ex=ttl)
    return response, False


# --- Schemas ---

class ChatRequest(BaseModel):
    model: str = "claude-3-5-sonnet-20241022"
    messages: list[dict]  # [{"role": "user", "content": "..."}]
    temperature: float = 0.0  # cache only makes sense for temp=0


class ChatResponse(BaseModel):
    content: str
    tokens: int
    model: str
    cache_hit: bool
    latency_ms: int


# --- App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    r = await get_redis()
    await r.ping()
    print("✅ Redis connected")
    yield
    if redis_pool:
        await redis_pool.aclose()
    print("✅ Redis connection closed")


app = FastAPI(title="Ex 7: Redis LLM Response Caching", lifespan=lifespan)


@app.post("/chat/cached", response_model=ChatResponse)
async def cached_chat(request: ChatRequest, r: RedisClient):
    """
    LLM endpoint with response caching.
    - First call: cache MISS, calls LLM, stores result
    - Second identical call: cache HIT, returns instantly
    """
    if request.temperature > 0:
        # Don't cache non-deterministic responses
        response = await simulate_llm_call(request.model, request.messages)
        return ChatResponse(cache_hit=False, latency_ms=500, **response)

    start = time.monotonic()
    response, cache_hit = await cached_llm_call(r, request.model, request.messages)
    latency_ms = int((time.monotonic() - start) * 1000)

    return ChatResponse(
        content=response["content"],
        tokens=response["tokens"],
        model=response["model"],
        cache_hit=cache_hit,
        latency_ms=latency_ms,
    )


@app.delete("/cache/flush")
async def flush_cache(r: RedisClient):
    """Flush all LLM cache keys (dev/testing only)."""
    keys = await r.keys("llm:cache:*")
    if keys:
        await r.delete(*keys)
    return {"flushed": len(keys), "keys": [k[:30] + "..." for k in keys]}


@app.get("/cache/stats")
async def cache_stats(r: RedisClient):
    """Show cache key count and memory info."""
    keys = await r.keys("llm:cache:*")
    info = await r.info("memory")
    return {
        "cached_responses": len(keys),
        "redis_memory_used": info.get("used_memory_human"),
    }