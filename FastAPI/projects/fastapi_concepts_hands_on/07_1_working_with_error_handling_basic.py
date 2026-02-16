"""
In this example, we will create a simple FastAPI application that manages conversations. We will implement basic error handling to ensure that our API responds appropriately when a conversation is not found.

This application includes:
1. An endpoint to retrieve a conversation by its ID, which returns a 404 Not Found error if the conversation does not exist.
2. An endpoint to list all conversations.
3. An endpoint to delete a conversation by its ID, which also returns a 404 Not Found error if the conversation does not exist.
4. We will use an in-memory database (a dictionary) to store our conversations for simplicity.

Concepts covered:
- HTTPException(status_code, detail)
- Raising exceptions in endpoints
- How FastAPI converts exceptions to JSON responses

Key Takeaway:
- HTTPException is FastAPI's primary way to return error responses. Raise it with the appropriate status code and message, and FastAPI handles the rest. This is perfect for simple errors like "resource not found."

"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Create an instance of the FastAPI application
app = FastAPI()

# ====== Models =====
class Conversation(BaseModel):
    id: int
    title: str
    message_count: int

# ====== In-Memory Database =====
conversations_db = {
    "conv_1": Conversation(id=1, title="General Chat", message_count=100),
    "conv_2": Conversation(id=2, title="Project Discussion", message_count=50),
    "conv_3": Conversation(id=3, title="Random Talk", message_count=20),
}

# ====== Endpoints =====
@app.get("/conversations/{conversation_id}", response_model=Conversation)
def get_conversation(conversation_id: str):
    if conversation_id not in conversations_db:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversations_db[conversation_id]

@app.get("/conversations", response_model=list[Conversation])
def list_conversations():
    return list(conversations_db.values())

@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    if conversation_id not in conversations_db:
        raise HTTPException(status_code=404, detail="Conversation not found")
    del conversations_db[conversation_id]
    return {"message": "Conversation deleted successfully"}