"""
This example demonstrates how to implement CRUD operations and cursor-based pagination with async SQLAlchemy in FastAPI.

Key concepts:
- Defining Pydantic models for request validation and response serialization. (UserCreate, ConversationCreate, MessageCreate, MessageResponse, MessagePage)
- Implementing CRUD functions for creating users, conversations, messages, and fetching paginated messages.
- Using cursor-based pagination for efficient retrieval of messages in a conversation. (get_messages_page function)
- Implementing API endpoints for creating users, conversations, messages, listing messages with pagination, and soft-deleting conversations.
- Soft deletion by setting a deleted_at timestamp instead of actually deleting records from the database.

Key Takeaway:
- The LIMIT + 1 trick is the cleanest way to detect "is there a next page?" — fetch one more than requested. 
  If you get limit + 1 rows, there are more results; strip the extra row and use the last row's ID as the cursor. This avoids a separate COUNT(*) query.

"""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Annotated, Literal

from fastapi import FastAPI, Depends, HTTPException, Query
from pydantic import BaseModel, EmailStr
from pydantic_settings import BaseSettings
from sqlalchemy import select
from sqlalchemy import (
    String, Integer, Text, JSON, DateTime, ForeignKey, func,
)
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession, async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from pgvector.sqlalchemy import Vector

# --- (re-import models from Exercise 2 setup — copy Base + models here) ---
# For brevity, assume User, Conversation, Message, Base, engine, AsyncSessionLocal
# are defined as in Exercise 2. In your file, paste all the model definitions.


class Settings(BaseSettings):
    database_url: str
    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()
engine = create_async_engine(settings.database_url, pool_size=10, pool_pre_ping=True, connect_args={"statement_cache_size": 0},)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# [paste Base + User + Conversation + Message models from Exercise 2 here]
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
    role: Literal["user", "assistant", "system"]
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
        .order_by(Message.created_at)
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