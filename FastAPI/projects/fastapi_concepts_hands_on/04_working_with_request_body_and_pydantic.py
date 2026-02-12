from fastapi import FastAPI
from pydantic import BaseModel, Field

# Create an instance of the FastAPI class
app = FastAPI()

# ====== Models ========
class Message(BaseModel):
    role: str = Field(..., description="The role of the message sender, e.g., 'user' or 'assistant'")
    content: str = Field(..., description="The content of the message")

class ChatRequest(BaseModel):
    model: str 
    messages: list[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature for response generation, between 0.0 and 2.0")
    max_tokens: int = Field(default=1000, gt=1, lt=4096, description="Maximum number of tokens in the generated response, between 1 and 4096")

class ChatResponse(BaseModel):
    id: str
    model: str
    content: str 
    tokens_used: int 


# ====== Endpoints ========

# Endpoint with only request body validation
@app.post("/chat/completions")
def create_chat_completion(request: ChatRequest):
    """Create a chat completion based on the provided request body.""" # For demonstration purposes, we'll return a dummy response. In a real application, you would integrate with a GenAI model here. response = ChatResponse( id="response-123", model=request.model, content="This is a generated response based on your input messages.", tokens_used=50 ) return response
    response = ChatResponse(
        id = "message-123",
        model = request.model,
        content = "This is a generated response based on your input messages.", 
        tokens_used = 50 
    )

    response = "Since there is no response validation, we can return any response object!"

    return response

# Endpoint with both request body and response validation
@app.post("/chat/completions/validated", response_model=ChatResponse)
def create_validated_chat_completion(request: ChatRequest): 
    """Create a chat completion with both request body and response validation.""" # For demonstration purposes, we'll return a dummy response. In a real application, you would integrate with a GenAI model here. response = ChatResponse( id = "message-123", model = request.model, content = "This is a generated response based on your input messages.", tokens_used = 50 ) return response
    response = ChatResponse(id = "message-123", model = request.model, content = "Thisis a generated response based on your input messages.", tokens_used = 50) 

    response = "This will cause a validation error because the response does not conform to the ChatResponse model schema. Comment this line to see a proper response object returned!"
    return response


# LEARNINGS:
# If a parameter is declared in the path string (URL route), it's a path parameter
# If a parameter is NOT in the path and has a simple type (int, str, float, bool, etc.), it's a query parameter
# If a parameter is NOT in the path and is a Pydantic model (or complex type), it's a request body parameter