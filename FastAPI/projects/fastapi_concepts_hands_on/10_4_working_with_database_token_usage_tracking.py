"""
This example demonstrates how to implement token usage tracking for users in a FastAPI application using async SQLAlchemy.

Key concepts:
- How token budgeting works: each user has a token limit and a count of tokens used. Each time they make a request that consumes tokens, we check if they have enough budget before allowing the request, and then increment their usage after the request.
- Before every request to the LLM, what are the key things to verify to ensure the user have the required budget to make the request? (Hint: check the check_token_budget dependency function)
- How to atomically increment token usage in the database to avoid race conditions when multiple requests are made concurrently by the same user.   

Key Takeaway:
- Always check the token budget before the LLM call — not after. 
  If you call the LLM first and then discover the budget is exceeded, you've already spent money and can't refund it. 
  The pre-check is an optimistic guard; the post-check prevents edge cases where the call overshoots the remaining budget.

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
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import (String, Integer, Text, JSON, DateTime, ForeignKey, func)
from datetime import datetime
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from pgvector.sqlalchemy import Vector


class Settings(BaseSettings):
    database_url: str
    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()
engine = create_async_engine(settings.database_url, pool_size=10, pool_pre_ping=True, connect_args={"statement_cache_size": 0},)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# [paste Base + User + Conversation + Message models from Exercise 2]
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
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)

    conversation: Mapped["Conversation"] = relationship(
        back_populates="messages", lazy="noload"
    )



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