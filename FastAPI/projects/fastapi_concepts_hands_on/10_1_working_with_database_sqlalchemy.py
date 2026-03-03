"""
This example demonstrates how to set up an async SQLAlchemy engine and session in FastAPI, and verify the database connection on startup.

Key concepts:
- Using `pydantic_settings` to load configuration from environment variables. (Clean, typed settings management)
- Using `create_async_engine` to create an async SQLAlchemy engine.
- Using `async_sessionmaker` to create an async session factory.
- Implementing a dependency to provide an async session to route handlers.
- Using `asynccontextmanager` to verify the database connection on application startup and dispose of the engine on shutdown. (lifespan config for FastAPI)
    - Before yield -> controls the startup phase, after yield -> controls the shutdown phase.

Key Takeaway:
- The expire_on_commit=False setting is non-negotiable for async SQLAlchemy. 
  Without it, accessing object attributes after commit() triggers implicit synchronous I/O and raises MissingGreenlet. 
  The pool_pre_ping=True setting prevents "connection was closed" errors after PostgreSQL drops idle connections (Supabase does this after ~10 minutes of inactivity).

-   App starts
    ↓
    Verify DB connection
    ↓
    yield → serve requests
    ↓
    App stops
    ↓
    Dispose DB engine


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
        extra = "allow"


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