# Topic 6: Database Integration

> **Why this matters for GenAI builders:** Every production chat app needs to store conversations, enforce per-user access controls, track token costs, and cache expensive LLM calls. This topic builds the complete data layer under your GenAI backend—from async PostgreSQL sessions through Supabase's managed platform (with Row Level Security for multi-tenancy) to Redis caching and rate limiting. Skip this and you're rebuilding SQLite side-projects. Learn this and you have the persistence layer for a real product.

---

## Table of Contents

1. [Part 1: Async SQLAlchemy + PostgreSQL](#part-1-async-sqlalchemy--postgresql)
2. [Part 2: GenAI Data Models](#part-2-genai-data-models)
3. [Part 3: CRUD + Pagination](#part-3-crud--pagination)
4. [Part 4: Supabase](#part-4-supabase)
5. [Part 5: Redis Deep Dive](#part-5-redis-deep-dive)
6. [Quick Reference](#quick-reference)
7. [Next Steps](#next-steps)

---

## Part 1: Async SQLAlchemy + PostgreSQL

### Why Async DB for GenAI

In a synchronous backend, every database query blocks the thread. If an LLM call takes 8 seconds and triggers 3 DB writes, your thread is tied up for the full duration. Async DB releases the thread back to the event loop during every I/O wait—the same principle you learned in Topic 5.

Think of it like Spark DAGs: DB operations are stages in a DAG, not sequential blocking steps. Async execution lets the scheduler run other tasks while waiting for I/O to complete.

```
SYNC (blocks thread):
Thread 1: [LLM call 8s ─────────────────] [DB write] [DB write] [DB read]
Thread 2:                                 waiting...

ASYNC (event loop):
Task 1:   [LLM call 8s ─────────────────] [DB write ─] [DB write ─] [DB read ─]
Task 2:   ──[DB read ──] [process] [respond]
Task 3:         ──[DB read ──] [process] [respond]
                          ↑ event loop runs these during Task 1's I/O waits
```

### Setting Up the Async Engine

Install the async PostgreSQL driver alongside SQLAlchemy:

```bash
# asyncpg is the fastest async PostgreSQL driver for Python
uv add asyncpg sqlalchemy
```

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

# ✅ Correct: use postgresql+asyncpg:// scheme
DATABASE_URL = "postgresql+asyncpg://user:password@host:5432/dbname"

engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,          # ← Higher than typical CRUD apps
    max_overflow=10,       # Allow burst capacity
    pool_pre_ping=True,    # Verify connections before use
    echo=False,            # Set True for SQL query logging in dev
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # ← Critical: don't expire objects after commit
)

class Base(DeclarativeBase):
    pass
```

**Why `pool_size=20` for GenAI?** Standard CRUD apps use pool_size=5. GenAI apps are different: each LLM call holds a "logical transaction" open for 5–30 seconds while waiting for the AI response. Multiple concurrent users need multiple connections. Size your pool to your expected concurrency.

**Why `expire_on_commit=False`?** By default, SQLAlchemy expires all object attributes after a commit, forcing a re-fetch on next access. In async code, that re-fetch requires `await`, and forgetting it causes `MissingGreenlet` errors. Setting `expire_on_commit=False` keeps the objects populated after commit.

### FastAPI Session Dependency

The standard pattern: an async generator dependency that creates a session, yields it, and guarantees cleanup via `try/finally`.

```python
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

```python
# Use in endpoints via Annotated + Depends
from typing import Annotated
from fastapi import Depends

DBSession = Annotated[AsyncSession, Depends(get_db)]

@app.post("/conversations")
async def create_conversation(
    data: ConversationCreate,
    db: DBSession,           # ← injected automatically
):
    ...
```

---

## Part 2: GenAI Data Models

### The Data Hierarchy

Every chat application maps to this relationship:

```
User (1) ──────────────────── (N) Conversation
│  id: UUID                        │  id: UUID
│  email: str                      │  user_id: FK → User
│  token_limit: int                │  title: str
│  tokens_used: int                │  system_prompt: str
│  created_at: datetime            │  model: str ("claude-3-5-sonnet")
                                   │  created_at: datetime
                                   │  deleted_at: datetime (soft delete)
                                   │
                                   └── (N) Message
                                           id: UUID
                                           conversation_id: FK → Conversation
                                           role: str ("user" | "assistant")
                                           content: str
                                           tokens: int         ← per-message
                                           latency_ms: int     ← LLM response time
                                           provider_metadata: JSON
                                           created_at: datetime
```

### Why Track `tokens` and `latency_ms` Per Message

These two fields unlock production analytics:

- **`tokens`**: Cost attribution per user, per conversation, per model. Without this, you're flying blind on infrastructure costs.
- **`latency_ms`**: Identify slow providers, slow models, or slow prompts. Compare Claude Haiku vs Sonnet vs GPT-4o latency on your actual workload.
- **`provider_metadata` (JSON)**: Store raw provider response data (`finish_reason`, `model_version`, `usage` breakdown) without adding columns every time the API changes.

### SQLAlchemy Models

```python
import uuid
from datetime import datetime
from sqlalchemy import String, Integer, ForeignKey, Text, JSON, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    token_limit: Mapped[int] = mapped_column(Integer, default=100_000)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="user", lazy="noload"  # ← noload prevents implicit sync I/O
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    system_prompt: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(100), default="claude-3-5-sonnet-20241022")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped["User"] = relationship(back_populates="conversations", lazy="noload")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", lazy="noload", order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    provider_metadata: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    conversation: Mapped["Conversation"] = relationship(
        back_populates="messages", lazy="noload"
    )
```

> **★ Key Insight:** Always use `lazy="noload"` on relationships in async SQLAlchemy. The default `lazy="select"` triggers implicit synchronous I/O on attribute access, causing `MissingGreenlet` errors. With `noload`, relationships are never loaded unless you explicitly join them.

---

## Part 3: CRUD + Pagination

### Async Session Patterns

The four operations you'll use constantly:

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# CREATE
async def create_conversation(db: AsyncSession, data: ConversationCreate) -> Conversation:
    conv = Conversation(**data.model_dump())
    db.add(conv)
    await db.commit()
    await db.refresh(conv)   # ← refresh to get server-generated fields (id, created_at)
    return conv

# READ (single)
async def get_conversation(db: AsyncSession, conv_id: uuid.UUID) -> Conversation | None:
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conv_id,
            Conversation.deleted_at.is_(None)  # exclude soft-deleted
        )
    )
    return result.scalar_one_or_none()

# UPDATE
async def update_conversation_title(
    db: AsyncSession, conv_id: uuid.UUID, title: str
) -> Conversation:
    conv = await get_conversation(db, conv_id)
    conv.title = title
    await db.commit()
    await db.refresh(conv)
    return conv

# SOFT DELETE (preferred for conversation history)
async def delete_conversation(db: AsyncSession, conv_id: uuid.UUID) -> None:
    conv = await get_conversation(db, conv_id)
    conv.deleted_at = datetime.utcnow()
    await db.commit()
```

### Cursor-Based Pagination

For chat history, OFFSET-based pagination breaks down fast:

```
❌ OFFSET pagination:
SELECT * FROM messages
WHERE conversation_id = $1
ORDER BY created_at
OFFSET 1000 LIMIT 20;
-- Problem: Must scan 1020 rows to return 20. Slows with every page.
-- Problem: If a new message arrives during pagination, rows shift → duplicate/skip
```

```
✅ Cursor-based pagination:
SELECT * FROM messages
WHERE conversation_id = $1
  AND id > $cursor          -- start after last seen item
ORDER BY id                 -- stable, index-friendly ordering
LIMIT $limit;
-- O(log n) via btree index. Stable regardless of concurrent inserts.
```

```python
from sqlalchemy import select

async def get_messages_page(
    db: AsyncSession,
    conversation_id: uuid.UUID,
    limit: int = 50,
    cursor: uuid.UUID | None = None,
) -> tuple[list[Message], uuid.UUID | None]:
    """Returns (messages, next_cursor). next_cursor is None when no more pages."""
    query = select(Message).where(
        Message.conversation_id == conversation_id
    ).order_by(Message.id).limit(limit + 1)  # fetch 1 extra to detect next page

    if cursor:
        query = query.where(Message.id > cursor)

    result = await db.execute(query)
    messages = result.scalars().all()

    next_cursor = None
    if len(messages) > limit:
        messages = messages[:limit]
        next_cursor = messages[-1].id

    return list(messages), next_cursor
```

### Soft Deletes

```python
# ✅ Soft delete (preferred for conversations)
# Preserves history, enables audit trails, allows recovery
conv.deleted_at = datetime.utcnow()
await db.commit()

# All queries filter: .where(Conversation.deleted_at.is_(None))

# ❌ Hard delete — data is gone permanently
await db.delete(conv)
await db.commit()
```

> **★ Key Insight:** Use soft deletes for anything users might want back. For GenAI apps, deleted conversations might still be needed for training data, billing disputes, or compliance. Soft delete = safety net. Add a background job to hard-delete rows older than 90 days if storage costs become a concern.

---

## Part 4: Supabase

### What Supabase Is

Supabase is a hosted PostgreSQL platform that bundles everything a GenAI backend needs in one service:

```
┌─────────────────────────────────────────────────────────┐
│                    SUPABASE PLATFORM                     │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │  PostgreSQL │  │    Auth     │  │   Storage (S3)  │ │
│  │  + pgvector │  │  (JWT/OAuth)│  │  (file uploads) │ │
│  └──────┬──────┘  └──────┬──────┘  └─────────────────┘ │
│         │                │                              │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌─────────────────┐ │
│  │ PostgREST   │  │    RLS      │  │    Realtime     │ │
│  │ (auto REST) │  │  (row-level │  │  (websockets)   │ │
│  │             │  │  security)  │  │                 │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
                          │
              Your FastAPI backend talks to
              PostgreSQL directly (SQLAlchemy)
              OR via supabase-py client
```

### The `acreate_client()` Dependency

```python
from supabase import acreate_client, AsyncClient
from typing import AsyncGenerator
import os

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]  # anon key for user-facing ops

async def get_supabase() -> AsyncGenerator[AsyncClient, None]:
    """FastAPI dependency that yields an async Supabase client."""
    client = await acreate_client(SUPABASE_URL, SUPABASE_KEY)
    yield client
    # supabase-py manages connection lifecycle internally

SupabaseClient = Annotated[AsyncClient, Depends(get_supabase)]

@app.get("/conversations")
async def list_conversations(client: SupabaseClient):
    result = await client.table("conversations").select("*").execute()
    return result.data
```

### Row Level Security (RLS) — The Killer Feature

RLS enforces that users can only access their own data at the database level—not in application code. Even if a bug in your FastAPI code tries to fetch another user's conversations, the database silently returns empty.

**How it works:**
```sql
-- 1. Enable RLS on the table (rows are invisible to all by default)
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;

-- 2. Create a policy for authenticated users
-- ✅ Correct pattern (per Supabase docs):
CREATE POLICY "Users can only see their own conversations"
ON conversations
FOR ALL
TO authenticated
USING ((select auth.uid()) = user_id);
--      ↑ wrap in (select ...) — evaluated once per query, not per row
--                               ↑ TO authenticated = only applies to logged-in users

-- 3. Add index for performance (RLS adds a WHERE clause on every query)
CREATE INDEX idx_conversations_user_id ON conversations (user_id);
```

**Demonstrating RLS in action:**

```python
# With ANON KEY → respects RLS → only sees auth.uid()'s rows
anon_client = await acreate_client(SUPABASE_URL, SUPABASE_ANON_KEY)
result = await anon_client.table("conversations").select("*").execute()
# → returns [] if no JWT token set (user not authenticated)

# With SERVICE KEY → bypasses RLS → sees ALL rows
service_client = await acreate_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
result = await service_client.table("conversations").select("*").execute()
# → returns all conversations from all users
```

```
✅ Correct RLS policy performance pattern:
USING ((select auth.uid()) = user_id)
↑ Evaluated ONCE per query

❌ Slow pattern:
USING (auth.uid() = user_id)
↑ Evaluated ONCE PER ROW — catastrophic on large tables
```

> **★ Key Insight:** RLS is the reason Supabase works well for direct client access (React → Supabase, no FastAPI middleware). Your database enforces security, not just your application. This is genuinely different from anything you can build quickly with raw PostgreSQL.

### supabase-py vs SQLAlchemy: Decision Matrix

| Use Case | supabase-py | SQLAlchemy |
|----------|-------------|------------|
| Simple CRUD with RLS | ✅ Preferred | Works, more code |
| Auth-integrated queries | ✅ Native | Requires custom auth |
| Complex JOIN queries | ❌ Limited | ✅ Full SQL power |
| Bulk inserts/updates | ❌ Slow | ✅ Batch operations |
| Fine-grained transactions | ❌ Limited | ✅ Full control |
| Prototyping / admin scripts | ✅ Fast | Works |
| Production app data layer | Both — often used together | Both |

**Production pattern:** Use `supabase-py` for user-facing CRUD (RLS + auth), use SQLAlchemy for analytics queries, bulk operations, and complex joins.

### pgvector Preview

pgvector is PostgreSQL's vector similarity extension — the foundation of RAG (Phase 5). Here's the preview:

```sql
-- Enable the extension (Supabase has this built-in)
CREATE EXTENSION IF NOT EXISTS vector;

-- Add an embedding column to messages
ALTER TABLE messages ADD COLUMN embedding VECTOR(1536);

-- Create an index for fast similarity search
CREATE INDEX ON messages
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

```python
# Store a mock embedding (in production this comes from an embedding model)
import json

await client.table("messages").update({
    "embedding": json.dumps([0.1, 0.2, 0.3, ...])  # 1536 floats
}).eq("id", message_id).execute()

# Cosine similarity search: find the 5 most similar messages
result = await client.rpc(
    "match_messages",
    {"query_embedding": embedding, "match_count": 5}
).execute()
```

The `match_messages` RPC function you'd create in Supabase:
```sql
CREATE FUNCTION match_messages(
    query_embedding VECTOR(1536),
    match_count INT DEFAULT 5
)
RETURNS TABLE (id UUID, content TEXT, similarity FLOAT)
LANGUAGE SQL STABLE AS $$
    SELECT id, content, 1 - (embedding <=> query_embedding) AS similarity
    FROM messages
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> query_embedding   -- cosine distance operator
    LIMIT match_count;
$$;
```

> **★ Key Insight:** This exact pattern — embed → store → search — is the entire technical core of RAG. Phase 5 builds on this foundation. The pgvector setup you do here is not throwaway code.

---

## Part 5: Redis Deep Dive

### Why Redis for GenAI Backends

Redis is the real-time backbone of production GenAI apps. It handles the millisecond-speed operations that PostgreSQL is too slow for:

```
Request arrives at FastAPI
        │
        ├─► Rate limit check (Redis SORTED SET) ← 1ms
        │
        ├─► Check response cache (Redis STRING) ← 2ms
        │         │
        │   cache HIT → return cached response (skip LLM)
        │   cache MISS ↓
        │
        ├─► Load conversation context (Redis HASH) ← 2ms
        │         │
        │   available → use Redis cache
        │   not found → query PostgreSQL + warm Redis
        │
        ├─► Call LLM API ← 1000–30000ms
        │
        ├─► Increment token counter (Redis INCR) ← 1ms
        │
        ├─► Store response in cache (Redis SET EX) ← 1ms
        │
        └─► Update conversation context in Redis ← 1ms
```

### Redis Data Structures for GenAI

**STRING + INCR — Atomic Token Counters**
```python
import redis.asyncio as redis

r = redis.Redis.from_url("redis://localhost:6379")

# Increment token counter atomically
await r.incrby(f"user:{user_id}:tokens:daily", tokens_used)
await r.expire(f"user:{user_id}:tokens:daily", 86400)  # expires in 24h

# Check budget
used = int(await r.get(f"user:{user_id}:tokens:daily") or 0)
if used > TOKEN_LIMIT:
    raise HTTPException(429, "Daily token limit exceeded")
```

**HASH — Conversation Metadata Cache**
```python
# Cache conversation metadata to avoid DB lookup on every message
await r.hset(f"conv:{conv_id}", mapping={
    "model": "claude-3-5-sonnet-20241022",
    "system_prompt": "You are a helpful assistant.",
    "user_id": str(user_id),
})
await r.expire(f"conv:{conv_id}", 1800)  # 30 min TTL

# Retrieve
conv_data = await r.hgetall(f"conv:{conv_id}")
```

**SORTED SET + ZREMRANGEBYSCORE — Sliding Window Rate Limiter**
```python
import time
import uuid

async def check_rate_limit(
    r: redis.Redis,
    user_id: str,
    limit: int = 10,
    window_seconds: int = 60
) -> bool:
    """Returns True if request is allowed. False if rate limited."""
    key = f"ratelimit:{user_id}"
    now = time.time()
    window_start = now - window_seconds

    pipe = r.pipeline()
    # Remove entries older than the window
    pipe.zremrangebyscore(key, 0, window_start)
    # Add current request
    pipe.zadd(key, {str(uuid.uuid4()): now})
    # Count requests in window
    pipe.zcount(key, window_start, "+inf")
    # Set key expiry (auto-cleanup)
    pipe.expire(key, window_seconds)
    results = await pipe.execute()

    request_count = results[2]
    return request_count <= limit
```

```
Sliding window visualization (limit=3, window=60s):

Timeline:    |──────────60s────────────|
Requests:    req1   req2    req3   req4
Time (s):    0      20      40     55

At req4 (t=55s):
  - Window start = t-60 = -5s
  - req1 (t=0) is still in window
  - Count = 4 → RATE LIMITED (429)

At req5 (t=70s):
  - Window start = t-60 = 10s
  - req1 (t=0) is outside window → removed
  - Count = 3 (req2, req3, req4) → ALLOWED
```

**LIST — Message Queues**
```python
# Push message to async processing queue
await r.rpush("llm:queue", json.dumps({
    "conversation_id": str(conv_id),
    "content": user_message,
}))

# Worker pops from queue (blocking pop with 1s timeout)
item = await r.blpop("llm:queue", timeout=1)
```

**STRING + EX — LLM Response Cache with TTL**
```python
import hashlib
import json

def make_cache_key(model: str, messages: list, temperature: float) -> str:
    """Deterministic cache key for LLM calls."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }, sort_keys=True)
    return f"llm:cache:{hashlib.sha256(payload.encode()).hexdigest()}"

async def cached_llm_call(r: redis.Redis, model: str, messages: list) -> str:
    """Cache-aside pattern for LLM responses."""
    key = make_cache_key(model, messages, temperature=0.0)

    # Check cache
    cached = await r.get(key)
    if cached:
        return json.loads(cached)["content"]

    # Call LLM (mock)
    response = await call_llm(model, messages)

    # Store with TTL
    await r.set(key, json.dumps({"content": response}), ex=86400)  # 24h
    return response
```

### LLM Response Caching Strategy

```
✅ Cache these:
├─ FAQ-style requests (temperature=0, deterministic)
├─ System prompt + fixed query patterns (product listings, definitions)
├─ Embedding requests (same text → same vector)
└─ Classification requests (same input → same category)

❌ Don't cache these:
├─ Creative generation (temperature > 0)
├─ Requests with current timestamps or dates
├─ Tool calls with side effects (send email, write to DB)
├─ Personalized responses that depend on user history
└─ Streaming responses (partial results don't cache well)
```

**TTL Tiers:**

| Cache Type | TTL | Reason |
|------------|-----|--------|
| Exact FAQ match | 24 hours | Static content, high reuse |
| Session-scoped context | 30 minutes | Per-conversation, expires with session |
| Rate limit windows | 60–3600 seconds | Matches the limit window |
| Token counters (daily) | 24 hours | Reset daily |
| Conversation hot cache | 30 minutes | Active sessions |

### Conversation Context Hot Cache

The hot cache pattern avoids PostgreSQL on every LLM request:

```python
CONTEXT_WINDOW_SIZE = 10  # keep last N messages in Redis

async def get_conversation_context(
    r: redis.Redis,
    db: AsyncSession,
    conv_id: uuid.UUID,
) -> list[dict]:
    """Get recent messages from Redis cache, falling back to PostgreSQL."""
    cache_key = f"context:{conv_id}"

    # Try Redis first
    cached = await r.get(cache_key)
    if cached:
        return json.loads(cached)

    # Fall back to PostgreSQL
    messages, _ = await get_messages_page(db, conv_id, limit=CONTEXT_WINDOW_SIZE)
    context = [{"role": m.role, "content": m.content} for m in messages]

    # Warm the cache
    await r.set(cache_key, json.dumps(context), ex=1800)
    return context

async def append_to_context_cache(
    r: redis.Redis, conv_id: uuid.UUID, role: str, content: str
) -> None:
    """Append a new message to the context cache without DB read."""
    cache_key = f"context:{conv_id}"
    cached = await r.get(cache_key)
    context = json.loads(cached) if cached else []
    context.append({"role": role, "content": content})

    # Keep only last CONTEXT_WINDOW_SIZE messages
    context = context[-CONTEXT_WINDOW_SIZE:]
    await r.set(cache_key, json.dumps(context), ex=1800)
```

### Upstash vs Local Redis

| | Local Redis | Upstash Redis |
|--|-------------|---------------|
| Connection | Persistent TCP | Stateless HTTP |
| Latency | < 1ms (same machine) | 10–50ms (network) |
| Serverless-friendly | ❌ Requires always-on server | ✅ Per-request billing |
| Pricing | Free (self-hosted) | Free tier, then per-request |
| Full command set | ✅ Yes | ✅ Yes |
| Migration effort | — | Change 1 env var |

**The migration:** Change one environment variable, change the import.

```python
# Local Redis (development)
import redis.asyncio as redis
r = redis.Redis.from_url(os.environ["REDIS_URL"])  # redis://localhost:6379

# Upstash Redis (production / serverless)
from upstash_redis.asyncio import Redis
r = Redis.from_env()  # reads UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN
```

The Python API is identical after initialization. Design your code around `REDIS_URL` from day one and this migration is literally a one-line change per file.

> **★ Key Insight:** The `upstash-redis` package uses HTTP (REST API) under the hood, not the Redis binary protocol. This is why it works in serverless environments (AWS Lambda, Vercel Edge Functions) that can't maintain persistent TCP connections. The tradeoff is ~10–50ms of additional latency vs < 1ms for local Redis. For most GenAI apps where LLM calls take 1–30 seconds, this is irrelevant.

---

## Quick Reference

| Operation | Tool | Pattern |
|-----------|------|---------|
| Async DB session | SQLAlchemy + asyncpg | `async with AsyncSessionLocal() as session:` |
| Create record | SQLAlchemy | `session.add(obj)` → `await session.commit()` → `await session.refresh(obj)` |
| Cursor pagination | SQLAlchemy | `WHERE id > cursor ORDER BY id LIMIT n+1` |
| Supabase client | supabase-py | `await acreate_client(url, key)` |
| RLS policy | SQL | `USING ((select auth.uid()) = user_id) TO authenticated` |
| Vector search | pgvector | `ORDER BY embedding <=> query_embedding LIMIT n` |
| Token counter | Redis STRING | `INCRBY key n` + `EXPIRE key seconds` |
| Rate limiter | Redis SORTED SET | `ZADD` + `ZREMRANGEBYSCORE` + `ZCOUNT` |
| Response cache | Redis STRING | `GET key` → miss → LLM call → `SET key value EX ttl` |
| Context cache | Redis STRING | `GET context:{conv_id}` → miss → PostgreSQL → `SET` |
| Upstash migration | env var | `REDIS_URL` → `UPSTASH_REDIS_REST_URL` + token |

---

## Next Steps

- **Practice:** [06_Database_Integration_Practice.md](./06_Database_Integration_Practice.md) — 8 exercises building the full data layer
- **Next Topic:** Topic 7: LLM Integration — connect to real LLM providers and use the conversation storage you built here
- **Phase 5 preview:** Everything in Part 4 (pgvector setup) directly enables the RAG patterns in Topic 13
