# Topic 6: Database Integration — Hands-On Practice

> **Before you start:** Read `06_Database_Integration.md` to understand the concepts. This document is your step-by-step build guide — 8 exercises that assemble a complete data layer for a GenAI chat backend.

**Estimated time:** 5 hours hands-on
**Files you'll create:** `10_1_*` through `10_8_*` in `FastAPI/projects/fastapi_concepts_hands_on/`

---

## Setup

You need:
1. A Supabase project (URL + anon key + service key + connection string) — see Exercise 5
2. A local Redis instance: `docker run -d -p 6379:6379 redis:alpine` or `brew install redis && redis-server`
3. An Upstash account (free tier) for Exercise 8: https://upstash.com

Create a `.env` file in `FastAPI/projects/fastapi_concepts_hands_on/`:

```
# PostgreSQL (Supabase connection string — use "Transaction" mode pooler)
DATABASE_URL=postgresql+asyncpg://postgres.YOURREF:PASSWORD@aws-0-REGION.pooler.supabase.com:6543/postgres

# Supabase API
SUPABASE_URL=https://YOURREF.supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_KEY=eyJ...

# Redis
REDIS_URL=redis://localhost:6379

# Upstash (Exercise 8)
UPSTASH_REDIS_REST_URL=https://YOUR-URL.upstash.io
UPSTASH_REDIS_REST_TOKEN=your_token
```

---

## Exercise 1: Async SQLAlchemy Setup

**Goal:** Connect to PostgreSQL (Supabase) using async SQLAlchemy, verify the connection with a health endpoint.

**File:** `10_1_async_sqlalchemy_setup.py`
**Estimated time:** 30 minutes

### Steps

1. Create the async engine using the `DATABASE_URL` from `.env`
2. Create `AsyncSessionLocal` with `async_sessionmaker`
3. Write a `get_db` FastAPI dependency (async generator with try/finally)
4. Create a `Base` declarative base
5. Write a `/health/db` endpoint that executes `SELECT 1` to verify connection

### Complete Code

```python
"""
Exercise 1: Async SQLAlchemy Setup
Demonstrates: create_async_engine, AsyncSession, FastAPI dependency injection
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Annotated

from fastapi import FastAPI, Depends
from pydantic_settings import BaseSettings
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase


class Settings(BaseSettings):
    database_url: str

    class Config:
        env_file = ".env"


settings = Settings()


# --- Database setup ---

engine = create_async_engine(
    settings.database_url,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


# --- Dependency ---

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


DBSession = Annotated[AsyncSession, Depends(get_db)]


# --- App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Verify DB connection on startup
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    print("✅ Database connection verified")
    yield
    await engine.dispose()
    print("✅ Database engine disposed")


app = FastAPI(title="Ex 1: Async SQLAlchemy Setup", lifespan=lifespan)


@app.get("/health/db")
async def db_health(db: DBSession):
    """Verify database connectivity."""
    result = await db.execute(text("SELECT 1 AS ok"))
    row = result.first()
    return {"status": "ok", "db": row.ok == 1}


@app.get("/")
async def root():
    return {"message": "Exercise 1: Async SQLAlchemy Setup"}
```

### Test It

```bash
cd FastAPI/projects/fastapi_concepts_hands_on
uv run uvicorn 10_1_async_sqlalchemy_setup:app --reload --port 8001
```

```bash
curl http://localhost:8001/health/db
```

### What You Should See

```json
{"status": "ok", "db": true}
```

Startup logs should show `✅ Database connection verified`.

### Key Takeaway

The `expire_on_commit=False` setting is non-negotiable for async SQLAlchemy. Without it, accessing object attributes after `commit()` triggers implicit synchronous I/O and raises `MissingGreenlet`. The `pool_pre_ping=True` setting prevents "connection was closed" errors after PostgreSQL drops idle connections (Supabase does this after ~10 minutes of inactivity).

---

## Exercise 2: GenAI Data Models

**Goal:** Define User, Conversation, and Message SQLAlchemy models with proper relationships.

**File:** `10_2_genai_data_models.py`
**Estimated time:** 40 minutes

### Steps

1. Import the `Base` and `engine` from Exercise 1 (or redefine them)
2. Create `User` model with UUID primary key, email, token tracking
3. Create `Conversation` model with FK to User, system prompt, model selection
4. Create `Message` model with FK to Conversation, role, content, tokens, latency, metadata
5. Add `create_tables` lifespan event that calls `Base.metadata.create_all()`
6. Write a `/models/info` endpoint that returns the table names

### Complete Code

```python
"""
Exercise 2: GenAI Data Models
Demonstrates: SQLAlchemy 2.0 mapped_column, relationships, UUID PKs, JSON columns
"""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Annotated

from fastapi import FastAPI, Depends
from pydantic_settings import BaseSettings
from sqlalchemy import (
    String, Integer, Text, JSON, DateTime, ForeignKey, func, inspect,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession, async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Settings(BaseSettings):
    database_url: str

    class Config:
        env_file = ".env"


settings = Settings()

engine = create_async_engine(settings.database_url, pool_size=10, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


# --- Models ---

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
        back_populates="user", lazy="noload"
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
    model: Mapped[str] = mapped_column(
        String(100), default="claude-3-5-sonnet-20241022"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped["User"] = relationship(back_populates="conversations", lazy="noload")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        lazy="noload",
        order_by="Message.created_at",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
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


# --- Dependency ---

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


DBSession = Annotated[AsyncSession, Depends(get_db)]


# --- App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Tables created (or already exist)")
    yield
    await engine.dispose()


app = FastAPI(title="Ex 2: GenAI Data Models", lifespan=lifespan)


@app.get("/models/info")
async def models_info():
    """Show created table names."""
    return {
        "tables": list(Base.metadata.tables.keys()),
        "models": ["User", "Conversation", "Message"],
    }
```

### Test It

```bash
uv run uvicorn 10_2_genai_data_models:app --reload --port 8002
curl http://localhost:8002/models/info
```

### What You Should See

```json
{
  "tables": ["users", "conversations", "messages"],
  "models": ["User", "Conversation", "Message"]
}
```

Check your Supabase dashboard — you should see the three tables created under the `public` schema.

### Key Takeaway

`lazy="noload"` is mandatory in async SQLAlchemy. Any other lazy loading strategy (`select`, `dynamic`) will trigger implicit synchronous I/O on attribute access, crashing with `MissingGreenlet`. Always use `noload` and load relationships explicitly when needed (via `selectinload` or `joinedload` in your query).

---

## Exercise 3: Async CRUD + Cursor Pagination

**Goal:** Build create/read/list endpoints for conversations and messages, with cursor-based pagination.

**File:** `10_3_async_crud_pagination.py`
**Estimated time:** 45 minutes

### Steps

1. Reuse models from Exercise 2
2. Create Pydantic schemas for request/response
3. Implement `create_user`, `create_conversation`, `add_message` async functions
4. Implement `get_messages_page` with cursor-based pagination
5. Wire up POST/GET endpoints
6. Test: create user → create conversation → add 5 messages → paginate

### Complete Code

```python
"""
Exercise 3: Async CRUD + Cursor Pagination
Demonstrates: async session patterns, cursor pagination vs OFFSET, soft deletes
"""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Annotated

from fastapi import FastAPI, Depends, HTTPException, Query
from pydantic import BaseModel, EmailStr
from pydantic_settings import BaseSettings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession, async_sessionmaker,
)

# --- (re-import models from Exercise 2 setup — copy Base + models here) ---
# For brevity, assume User, Conversation, Message, Base, engine, AsyncSessionLocal
# are defined as in Exercise 2. In your file, paste all the model definitions.


class Settings(BaseSettings):
    database_url: str
    class Config:
        env_file = ".env"

settings = Settings()
engine = create_async_engine(settings.database_url, pool_size=10, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# [paste Base + User + Conversation + Message models from Exercise 2 here]


# --- Pydantic Schemas ---

class UserCreate(BaseModel):
    email: str

class ConversationCreate(BaseModel):
    user_id: uuid.UUID
    title: str
    system_prompt: str | None = None
    model: str = "claude-3-5-sonnet-20241022"

class MessageCreate(BaseModel):
    conversation_id: uuid.UUID
    role: str  # "user" or "assistant"
    content: str
    tokens: int | None = None
    latency_ms: int | None = None

class MessageResponse(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    tokens: int | None
    latency_ms: int | None
    created_at: datetime

    class Config:
        from_attributes = True

class MessagePage(BaseModel):
    messages: list[MessageResponse]
    next_cursor: uuid.UUID | None
    has_more: bool


# --- Dependency ---

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

DBSession = Annotated[AsyncSession, Depends(get_db)]


# --- CRUD Functions ---

async def create_user(db: AsyncSession, email: str) -> "User":
    user = User(email=email)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def create_conversation(db: AsyncSession, data: ConversationCreate) -> "Conversation":
    conv = Conversation(**data.model_dump())
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


async def add_message(db: AsyncSession, data: MessageCreate) -> "Message":
    msg = Message(**data.model_dump())
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg


async def get_messages_page(
    db: AsyncSession,
    conversation_id: uuid.UUID,
    limit: int = 20,
    cursor: uuid.UUID | None = None,
) -> tuple[list["Message"], uuid.UUID | None]:
    """Cursor-based pagination — O(log n), stable under concurrent inserts."""
    query = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.id)
        .limit(limit + 1)  # fetch one extra to detect if there's a next page
    )
    if cursor:
        query = query.where(Message.id > cursor)

    result = await db.execute(query)
    messages = list(result.scalars().all())

    next_cursor = None
    if len(messages) > limit:
        messages = messages[:limit]
        next_cursor = messages[-1].id

    return messages, next_cursor


# --- App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(title="Ex 3: Async CRUD + Pagination", lifespan=lifespan)


@app.post("/users", status_code=201)
async def create_user_endpoint(data: UserCreate, db: DBSession):
    user = await create_user(db, data.email)
    return {"id": str(user.id), "email": user.email}


@app.post("/conversations", status_code=201)
async def create_conversation_endpoint(data: ConversationCreate, db: DBSession):
    conv = await create_conversation(db, data)
    return {"id": str(conv.id), "title": conv.title, "model": conv.model}


@app.post("/messages", status_code=201)
async def add_message_endpoint(data: MessageCreate, db: DBSession):
    msg = await add_message(db, data)
    return {"id": str(msg.id), "role": msg.role, "tokens": msg.tokens}


@app.get("/conversations/{conv_id}/messages", response_model=MessagePage)
async def list_messages(
    conv_id: uuid.UUID,
    db: DBSession,
    limit: int = Query(default=10, ge=1, le=100),
    cursor: uuid.UUID | None = Query(default=None),
):
    messages, next_cursor = await get_messages_page(db, conv_id, limit, cursor)
    return MessagePage(
        messages=messages,
        next_cursor=next_cursor,
        has_more=next_cursor is not None,
    )


@app.delete("/conversations/{conv_id}", status_code=204)
async def soft_delete_conversation(conv_id: uuid.UUID, db: DBSession):
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conv_id,
            Conversation.deleted_at.is_(None),
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(404, "Conversation not found")
    conv.deleted_at = datetime.utcnow()
    await db.commit()
```

### Test It

```bash
uv run uvicorn 10_3_async_crud_pagination:app --reload --port 8003
```

Open `http://localhost:8003/docs` and run through the sequence:
1. `POST /users` → grab the `id`
2. `POST /conversations` → use the user `id`, grab the conversation `id`
3. `POST /messages` × 5 — add 5 messages with different roles
4. `GET /conversations/{id}/messages?limit=3` — verify you get 3 + a `next_cursor`
5. `GET /conversations/{id}/messages?limit=3&cursor={next_cursor}` — verify you get the remaining messages
6. `DELETE /conversations/{id}` — soft delete
7. Verify in Supabase dashboard that `deleted_at` is set (not physically deleted)

### What You Should See

First page (limit=3, 5 total messages):
```json
{
  "messages": [/* msg 1, 2, 3 */],
  "next_cursor": "uuid-of-msg-3",
  "has_more": true
}
```

Second page (with cursor):
```json
{
  "messages": [/* msg 4, 5 */],
  "next_cursor": null,
  "has_more": false
}
```

### Key Takeaway

The `LIMIT + 1` trick is the cleanest way to detect "is there a next page?" — fetch one more than requested. If you get `limit + 1` rows, there are more results; strip the extra row and use the last row's ID as the cursor. This avoids a separate `COUNT(*)` query.

---

## Exercise 4: Token Usage Tracking

**Goal:** Track tokens per message, aggregate per user, enforce token limits via a FastAPI dependency.

**File:** `10_4_token_usage_tracking.py`
**Estimated time:** 40 minutes

### Steps

1. Reuse models from Exercise 2
2. Write `increment_user_tokens(db, user_id, tokens)` function
3. Write a `check_token_budget` dependency that reads `User.token_limit` and `User.tokens_used`
4. Create a `/chat` endpoint that simulates an LLM call, stores the message, and updates token usage
5. Test: create user with low token limit → verify 429 after budget exhausted

### Complete Code

```python
"""
Exercise 4: Token Usage Tracking
Demonstrates: per-message token tracking, user budget enforcement via dependency
"""

import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Annotated

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession, async_sessionmaker,
)

# [paste Base + User + Conversation + Message models from Exercise 2]


class Settings(BaseSettings):
    database_url: str
    class Config:
        env_file = ".env"

settings = Settings()
engine = create_async_engine(settings.database_url, pool_size=10, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# --- Schemas ---

class ChatRequest(BaseModel):
    user_id: uuid.UUID
    conversation_id: uuid.UUID
    message: str

class ChatResponse(BaseModel):
    reply: str
    tokens_used: int
    user_tokens_remaining: int


# --- Dependency ---

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

DBSession = Annotated[AsyncSession, Depends(get_db)]


# --- Token functions ---

async def get_user_or_404(db: AsyncSession, user_id: uuid.UUID) -> "User":
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, f"User {user_id} not found")
    return user


async def increment_user_tokens(
    db: AsyncSession, user_id: uuid.UUID, tokens: int
) -> None:
    """Atomically increment tokens_used for a user."""
    await db.execute(
        update(User)
        .where(User.id == user_id)
        .values(tokens_used=User.tokens_used + tokens)
    )
    await db.commit()


async def check_token_budget(user_id: uuid.UUID, db: DBSession) -> "User":
    """Dependency: raises 429 if user has exceeded their token budget."""
    user = await get_user_or_404(db, user_id)
    if user.tokens_used >= user.token_limit:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "token_budget_exceeded",
                "tokens_used": user.tokens_used,
                "token_limit": user.token_limit,
            },
        )
    return user


# --- App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(title="Ex 4: Token Usage Tracking", lifespan=lifespan)


@app.post("/users", status_code=201)
async def create_user(db: DBSession, email: str, token_limit: int = 100_000):
    user = User(email=email, token_limit=token_limit)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return {"id": str(user.id), "email": user.email, "token_limit": user.token_limit}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: DBSession):
    """Simulate an LLM call with token tracking and budget enforcement."""
    # Check budget BEFORE calling LLM
    user = await check_token_budget(request.user_id, db)

    # Simulate: count tokens (rough estimate: 1 token ≈ 4 chars)
    input_tokens = len(request.message) // 4
    reply = f"Echo: {request.message} (simulated LLM response)"
    output_tokens = len(reply) // 4
    total_tokens = input_tokens + output_tokens

    # Check if this call would exceed the budget
    remaining = user.token_limit - user.tokens_used
    if total_tokens > remaining:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "insufficient_token_budget",
                "tokens_needed": total_tokens,
                "tokens_remaining": remaining,
            },
        )

    # Store user message
    db.add(Message(
        conversation_id=request.conversation_id,
        role="user",
        content=request.message,
        tokens=input_tokens,
    ))

    # Store assistant reply
    db.add(Message(
        conversation_id=request.conversation_id,
        role="assistant",
        content=reply,
        tokens=output_tokens,
        latency_ms=250,  # simulated
        provider_metadata={"model": "mock", "finish_reason": "stop"},
    ))

    # Update user token count
    await increment_user_tokens(db, request.user_id, total_tokens)
    await db.commit()

    return ChatResponse(
        reply=reply,
        tokens_used=total_tokens,
        user_tokens_remaining=remaining - total_tokens,
    )


@app.get("/users/{user_id}/usage")
async def get_token_usage(user_id: uuid.UUID, db: DBSession):
    user = await get_user_or_404(db, user_id)
    return {
        "tokens_used": user.tokens_used,
        "token_limit": user.token_limit,
        "tokens_remaining": user.token_limit - user.tokens_used,
        "usage_pct": round(user.tokens_used / user.token_limit * 100, 1),
    }
```

### Test It

```bash
uv run uvicorn 10_4_token_usage_tracking:app --reload --port 8004
```

1. Create a user with `token_limit=50` (tiny budget for testing)
2. POST to `/chat` a few times until budget is exhausted
3. Verify the 429 response with the budget details
4. GET `/users/{id}/usage` to see the usage percentage

### What You Should See

After budget exhaustion:
```json
{
  "detail": {
    "error": "token_budget_exceeded",
    "tokens_used": 50,
    "token_limit": 50
  }
}
```

### Key Takeaway

Always check the token budget _before_ the LLM call — not after. If you call the LLM first and then discover the budget is exceeded, you've already spent money and can't refund it. The pre-check is an optimistic guard; the post-check prevents edge cases where the call overshoots the remaining budget.

---

## Exercise 5: Supabase Async Client + RLS Demonstration

**Goal:** Use `acreate_client()` as a FastAPI dependency, perform CRUD via supabase-py, and demonstrate RLS in action (anon key vs service key).

**File:** `10_5_supabase_async_client.py`
**Estimated time:** 45 minutes

### Steps

1. Ensure your `.env` has `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY`
2. Enable RLS on the `conversations` table in Supabase SQL Editor
3. Create an RLS policy (see SQL below)
4. Build FastAPI dependency for supabase-py client
5. Create two endpoints: one using anon key, one using service key
6. Observe that anon key returns empty until authenticated; service key bypasses RLS

### Supabase SQL to Run First

In your Supabase project → SQL Editor → New Query:

```sql
-- Enable RLS on conversations table
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;

-- Policy: authenticated users see only their own conversations
-- Note: (select auth.uid()) — evaluated once per query, not per row
CREATE POLICY "Users see only their own conversations"
ON conversations
FOR ALL
TO authenticated
USING ((select auth.uid()) = user_id);

-- Performance index
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations (user_id);
```

### Complete Code

```python
"""
Exercise 5: Supabase Async Client + RLS Demonstration
Demonstrates: acreate_client(), supabase-py CRUD, RLS enforcement difference
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
    supabase_service_key: str

    class Config:
        env_file = ".env"


settings = Settings()


# --- Dependencies ---

async def get_supabase_anon() -> AsyncGenerator[AsyncClient, None]:
    """Anon client — respects Row Level Security (RLS)."""
    client = await acreate_client(settings.supabase_url, settings.supabase_anon_key)
    yield client


async def get_supabase_service() -> AsyncGenerator[AsyncClient, None]:
    """Service client — bypasses RLS. Use only in trusted server-side code."""
    client = await acreate_client(settings.supabase_url, settings.supabase_service_key)
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
    title: str,
    model: str = "claude-3-5-sonnet-20241022",
    client: ServiceClient = Depends(get_supabase_service),
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
```

### Test It

```bash
uv run uvicorn 10_5_supabase_async_client:app --reload --port 8005
```

1. First, insert some rows via Supabase SQL Editor:
   ```sql
   INSERT INTO conversations (id, user_id, title, model)
   VALUES
     (gen_random_uuid(), gen_random_uuid(), 'Test Conv 1', 'claude-3-5-sonnet-20241022'),
     (gen_random_uuid(), gen_random_uuid(), 'Test Conv 2', 'gpt-4o');
   ```
2. `GET /conversations/service` → should return 2 conversations
3. `GET /conversations/anon` → should return `[]` (RLS blocks unauthenticated access)
4. `GET /rls-demo` → should show `"rls_working": true`

### What You Should See

```json
{
  "anon_key_count": 0,
  "service_key_count": 2,
  "rls_working": true,
  "explanation": "anon_key_count < service_key_count proves RLS is filtering rows..."
}
```

### Key Takeaway

The anon key and service key use the same underlying PostgreSQL database — the only difference is whether RLS policies are enforced. This is the killer feature: your security model lives in the database, not in your application code. A bug in your FastAPI layer can't accidentally expose data that RLS protects.

---

## Exercise 6: pgvector Preview

**Goal:** Enable pgvector, add an embedding column to messages, store a mock embedding, and run cosine similarity search.

**File:** `10_6_pgvector_preview.py`
**Estimated time:** 35 minutes

### Steps

1. Enable pgvector in Supabase (SQL Editor)
2. Add `embedding VECTOR(1536)` column to messages
3. Create the `match_messages` SQL function
4. Write FastAPI endpoints to store and search embeddings
5. Use random vectors as mock embeddings (no embedding model needed yet)

### Supabase SQL to Run First

```sql
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Add embedding column to messages table
ALTER TABLE messages ADD COLUMN IF NOT EXISTS embedding VECTOR(1536);

-- Create index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS idx_messages_embedding
ON messages USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create similarity search function
CREATE OR REPLACE FUNCTION match_messages(
    query_embedding VECTOR(1536),
    match_count INT DEFAULT 5,
    conversation_filter UUID DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    role TEXT,
    similarity FLOAT
)
LANGUAGE SQL STABLE AS $$
    SELECT
        id,
        content,
        role,
        1 - (embedding <=> query_embedding) AS similarity
    FROM messages
    WHERE
        embedding IS NOT NULL
        AND (conversation_filter IS NULL OR conversation_id = conversation_filter)
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;
```

### Complete Code

```python
"""
Exercise 6: pgvector Preview
Demonstrates: storing embeddings, cosine similarity search via Supabase RPC
This is a preview of the RAG patterns covered in Phase 5 (Topic 13).
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
    supabase_service_key: str

    class Config:
        env_file = ".env"


settings = Settings()


async def get_supabase() -> AsyncGenerator[AsyncClient, None]:
    client = await acreate_client(settings.supabase_url, settings.supabase_service_key)
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
    match_count: int = 5,
    client: SupabaseClient = Depends(get_supabase),
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
```

### Test It

```bash
uv run uvicorn 10_6_pgvector_preview:app --reload --port 8006
```

1. First create a conversation to use as `conversation_id`:
   ```sql
   SELECT id FROM conversations LIMIT 1;
   ```
2. `POST /messages/embed` three times with different content (use the same `conversation_id`)
3. `GET /messages/search?query=your+search+term&match_count=3`
4. Verify results are ordered by similarity score (highest first)

### What You Should See

```json
[
  {"id": "uuid", "content": "similar message content", "role": "assistant", "similarity": 0.9234},
  {"id": "uuid", "content": "less similar content", "role": "user", "similarity": 0.7891},
  {"id": "uuid", "content": "least similar", "role": "user", "similarity": 0.6543}
]
```

### Key Takeaway

The `<=>` operator is cosine distance (0 = identical, 2 = opposite). Similarity = `1 - distance`. The `ivfflat` index trades a small amount of recall for large speed gains on tables with millions of rows. This exact setup — `embedding VECTOR(1536)` column + `match_messages` RPC + cosine similarity — is the technical foundation of every RAG system you'll build in Phase 5.

---

## Exercise 7: Redis LLM Response Caching

**Goal:** Implement a cache-aside pattern for LLM responses — cache hit avoids the LLM call, cache miss calls LLM and stores result with TTL.

**File:** `10_7_redis_llm_cache.py`
**Estimated time:** 40 minutes

### Prerequisites

Start local Redis:
```bash
docker run -d -p 6379:6379 --name redis-dev redis:alpine
# OR
brew install redis && redis-server
```

### Steps

1. Create Redis connection from `REDIS_URL` env var
2. Write `make_cache_key(model, messages, temperature)` using SHA256
3. Write `cached_llm_call()` with cache-aside logic
4. Add cache hit/miss logging
5. Build `/chat/cached` endpoint
6. Test: first call logs MISS + takes ~500ms (simulated); second identical call logs HIT + takes < 5ms

### Complete Code

```python
"""
Exercise 7: Redis LLM Response Caching
Demonstrates: cache-aside pattern, deterministic cache keys, TTL strategy, hit/miss logging
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
```

### Test It

```bash
uv run uvicorn 10_7_redis_llm_cache:app --reload --port 8007
```

Send the same request twice:
```bash
# First call — should take ~500ms, logs "Cache MISS"
curl -X POST http://localhost:8007/chat/cached \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3-5-sonnet-20241022", "messages": [{"role": "user", "content": "What is FastAPI?"}]}'

# Second call — should take < 5ms, logs "Cache HIT"
curl -X POST http://localhost:8007/chat/cached \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3-5-sonnet-20241022", "messages": [{"role": "user", "content": "What is FastAPI?"}]}'
```

### What You Should See

First call:
```json
{"content": "...", "tokens": 25, "model": "claude-3-5-sonnet-20241022", "cache_hit": false, "latency_ms": 502}
```

Second identical call:
```json
{"content": "...", "tokens": 25, "model": "claude-3-5-sonnet-20241022", "cache_hit": true, "latency_ms": 2}
```

Server logs:
```
Cache MISS | key=llm:cache:a3f9b2... — calling LLM
Cache HIT  | key=llm:cache:a3f9b2...
```

### Key Takeaway

The cache key must be deterministic from the inputs. Using `sort_keys=True` in `json.dumps` ensures `{"b": 1, "a": 2}` and `{"a": 2, "b": 1}` produce the same hash. The SHA256 digest converts arbitrary-length input into a fixed 64-char key safe for Redis. Only cache `temperature=0` calls — non-deterministic outputs shouldn't be cached.

---

## Exercise 8: Redis Rate Limiting + Context Cache + Upstash Migration

**Goal:** Implement a sliding window rate limiter, a conversation context hot cache, then migrate to Upstash by swapping one environment variable.

**File:** `10_8_redis_rate_limit_context_upstash.py`
**Estimated time:** 60 minutes

### Prerequisites

For the Upstash section, create a free database at https://upstash.com and note the `UPSTASH_REDIS_REST_URL` and token.

### Steps

1. Build sliding window rate limiter using Redis SORTED SET
2. Build conversation context hot cache (avoid DB on every LLM call)
3. Build `/chat/full` endpoint combining rate limiting + context cache + mock LLM
4. Test rate limiting: >5 requests in 60s → 429
5. Swap the Redis client to Upstash: change env vars and one import, run the same tests

### Complete Code

```python
"""
Exercise 8: Redis Rate Limiting + Context Cache + Upstash Migration
Demonstrates: SORTED SET rate limiter, context hot cache, local→Upstash migration
"""

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

import redis.asyncio as aioredis
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# ─── To migrate to Upstash, comment out the block above and uncomment below: ───
# from upstash_redis.asyncio import Redis as UpstashRedis


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    rate_limit_requests: int = 5   # max requests
    rate_limit_window: int = 60    # per N seconds
    context_window_size: int = 10  # messages to keep in hot cache
    context_ttl: int = 1800        # 30 minutes

    class Config:
        env_file = ".env"


settings = Settings()

redis_client: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global redis_client
    if redis_client is None:
        redis_client = aioredis.Redis.from_url(settings.redis_url, decode_responses=True)
    return redis_client

# ─── Upstash equivalent (same API after this line): ─────────────────────────
# async def get_redis() -> UpstashRedis:
#     return UpstashRedis(
#         url=os.environ["UPSTASH_REDIS_REST_URL"],
#         token=os.environ["UPSTASH_REDIS_REST_TOKEN"],
#     )

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
```

### Test It

```bash
uv run uvicorn 10_8_redis_rate_limit_context_upstash:app --reload --port 8008
```

**Test rate limiting** (limit=5, window=60s):
```bash
CONV_ID=$(python -c "import uuid; print(uuid.uuid4())")
for i in {1..7}; do
  echo "Request $i:"
  curl -s -X POST http://localhost:8008/chat/full \
    -H "Content-Type: application/json" \
    -H "X-User-Id: test-user-1" \
    -d "{\"conversation_id\": \"$CONV_ID\", \"message\": \"Hello $i\"}" | python -m json.tool
done
```

**Test context cache**:
```bash
# After 3 messages, check context grows
curl http://localhost:8008/rate-limit/status/test-user-1
```

**Upstash migration** (no code changes — just env + import):

1. Sign up at https://upstash.com → create Redis database → copy credentials
2. Update `.env`:
   ```
   UPSTASH_REDIS_REST_URL=https://your-db.upstash.io
   UPSTASH_REDIS_REST_TOKEN=your_token
   ```
3. In `10_8_redis_rate_limit_context_upstash.py`:
   - Comment out `import redis.asyncio as aioredis`
   - Uncomment the Upstash import and `get_redis()` function
4. Run the same tests — behavior is identical

### What You Should See

Requests 1-5: normal responses
Request 6:
```json
{
  "detail": {
    "error": "rate_limit_exceeded",
    "requests_in_window": 6,
    "limit": 5,
    "window_seconds": 60,
    "retry_after": 60
  }
}
```

Context growing with each message:
```json
{"reply": "...", "context_size": 4, "context_source": "redis_cache", "rate_limit_used": 3, "rate_limit_max": 5}
```

### Key Takeaway

The sliding window rate limiter uses `ZADD` (add with timestamp score) + `ZREMRANGEBYSCORE` (prune old entries) + `ZCOUNT` (count remaining) in a single pipeline. All three operations execute atomically in one round-trip. The Upstash migration is truly one env var + one import: the `upstash-redis` SDK mirrors the `redis-py` API exactly. This is why you design with `REDIS_URL` from day one.

---

## Summary

You've built the complete data layer for a production GenAI backend:

| Exercise | What You Built | Key Concept |
|----------|----------------|-------------|
| 10_1 | Async SQLAlchemy engine + session | `expire_on_commit=False`, `pool_pre_ping` |
| 10_2 | User, Conversation, Message models | `lazy="noload"`, UUID PKs, JSON columns |
| 10_3 | CRUD + cursor pagination | `LIMIT n+1` trick, soft deletes |
| 10_4 | Token usage tracking + budget enforcement | Dependency guards, atomic `UPDATE` |
| 10_5 | Supabase async client + RLS demo | Anon vs service key, silent RLS filtering |
| 10_6 | pgvector embedding + similarity search | `<=>` operator, `ivfflat` index |
| 10_7 | Redis LLM response caching | Cache-aside, TTL tiers, SHA256 keys |
| 10_8 | Rate limiter + context cache + Upstash | SORTED SET, `REDIS_URL` migration pattern |

**Next:** Topic 7 — LLM Integration. You'll replace the mock LLM calls in these exercises with real Anthropic and OpenAI API calls, using the conversation storage and caching layer you just built.
