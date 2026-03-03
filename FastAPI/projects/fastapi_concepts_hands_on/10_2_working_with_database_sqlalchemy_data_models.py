"""
This example demonstrates how to define async SQLAlchemy data models in FastAPI, and create the corresponding tables in the database on application startup.

Key concepts:
- Defining SQLAlchemy ORM models using the DeclarativeBase class. (User, Conversation, Message)
- Using appropriate column types and relationships to model the data. (e.g. UUID primary keys, ForeignKey relationships, JSON column for provider metadata)
- Using the async context manager in the FastAPI lifespan to create tables on startup. (Base.metadata.create_all)
- Using the inspect function to verify that tables were created successfully.   

Key Takeaway:
- lazy="noload" is mandatory in async SQLAlchemy. 
  Any other lazy loading strategy (select, dynamic) will trigger implicit synchronous I/O on attribute access, crashing with MissingGreenlet. 
  Always use noload and load relationships explicitly when needed (via selectinload or joinedload in your query).

- DDL = How the database stores data
  ORM modeling = How your application THINKS about data
  ORM = Translator between the two

- engine.connect() -> connection object → execute raw SQL, manage transactions (like pencil, you write SQL queries and manage transactions manually)
  engine.begin() -> transaction context → auto-commit or rollback, connection management (like pen, you write SQL queries but transaction and connection management is handled for you. Useful for - create tables, migrations, writes)


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
        extra = "allow"


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