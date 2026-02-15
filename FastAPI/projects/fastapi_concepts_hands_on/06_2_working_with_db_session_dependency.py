"""
This module demonstrates how to work with a database session dependency in FastAPI.

This application includes:
1. A database setup using SQLAlchemy's asynchronous engine and session maker.
2. A dependency function `get_db` that provides a database session to endpoint functions and ensures proper cleanup after the request is processed.
3. Endpoints to list conversations, create a new conversation, and retrieve a specific conversation by ID, all of which utilize the database session provided by the dependency.

NOTE: In a real application, you would replace the dummy data and simulated database operations with actual queries and transactions using your database models. The `get_db` dependency ensures that each request gets a fresh database session and that the session is properly closed after the request is completed, preventing potential issues with lingering database connections.

"""

from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from typing import AsyncGenerator

# Create an instance of the FastAPI class
app = FastAPI()

# ====== Database Setup ========
# Use an in-memory SQLite database for demonstration purposes
DATABASE_URL = "sqlite+aiosqlite:///./test.db"
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# ===== Dependency ========
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide a database session with automatic cleanup"""
    print("Dependency: Creating a new database session")
    async with async_session() as session:
        try:
            yield session
        finally:
            print("Dependency: Closing the database session")

# ====== Endpoints using DB session ========            
@app.get("/conversations")
async def list_conversations(db: AsyncSession = Depends(get_db)):
    """This endpoint lists conversations using a database session provided by the dependency."""
    print("Endpoint: Listing conversations using the database session")
    # Simulate a database query (replace with actual query in a real application)
    # result = await db.execute("SELECT id, title FROM conversations")
    # conversations = result.fetchall()
    
    # dummy data for demonstration purposes
    conversations = [
        {"id": 1, "title": "Conversation 1"},
        {"id": 2, "title": "Conversation 2"},
        {"id": 3, "title": "Conversation 3"},
    ]
    return {"conversations": conversations}

@app.post("/conversations")
async def create_conversation(title: str, db: AsyncSession = Depends(get_db)):
    """This endpoint creates a new conversation using a database session provided by the dependency."""
    print(f"Endpoint: Creating a new conversation with title '{title}' using the database session")
    # Simulate a database insert (replace with actual insert in a real application)
    # new_conversation = Conversation(title=title)
    # db.add(new_conversation)
    # await db.commit()
    
    # dummy response for demonstration purposes
    return {"message": f"Conversation '{title}' created successfully!"}

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: int, db: AsyncSession = Depends(get_db)):
    """This endpoint retrieves a specific conversation by ID using a database session provided by the dependency."""
    print(f"Endpoint: Retrieving conversation with ID {conversation_id} using the database session")
    # Simulate a database query (replace with actual query in a real application)
    # result = await db.execute("SELECT id, title FROM conversations WHERE id = :id", {"id": conversation_id})
    # conversation = result.fetchone()
    
    # dummy data for demonstration purposes
    conversation = {"id": conversation_id, "title": f"Conversation {conversation_id}"}
    return {"conversation": conversation}