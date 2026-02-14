"""
This is a mini project to demonstrate the concepts of FastAPI by creating a simple GenAI backend API without any actual LLM integration. The API will have endpoints for health checks, listing available models, getting model details, and creating chat completions based on user input. The responses will be dummy data for demonstration purposes.

Concepts covered in this mini project:
1. Using FastAPI to create a simple API with multiple endpoints
2. Using Pydantic data models for request validation and response modeling
3. Using Enums for better validation and documentation of query parameters
4. Implementing basic error handling with HTTP exceptions
5. Using status codes to indicate the result of API operations
6. Organizing endpoints logically (health/info endpoints, chat endpoints, admin endpoints)

"""

from fastapi import FastAPI, Query
from fastapi import status, HTTPException
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

# Create an instance of the FastAPI class
app = FastAPI(
    title="GenAI Backend - Mini Project (without any LLM integration)",
    version="0.0.1"
)

# ====== Models ========
class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    META = "meta"
    LITELLM = "litellm"

class Message(BaseModel):
    role: str = Field(..., description="The role of the message sender (e.g., user, assistant)")
    content: str = Field(..., description="The content of the message")

class ChatRequest(BaseModel):
    model: str = Field(..., description="The ID of the model to use for generating the response")
    messages: list[Message] = Field(..., description="A list of messages in the conversation history")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature for response generation (between 0.0 and 2.0)")
    max_tokens: Optional[int] = Field(default=1000, gt=1, lt=4096, description="Maximum number of tokens in the generated response (between 1 and 4096)")

class ChatResponse(BaseModel):
    id: str = Field(..., description="The unique identifier for the generated response")
    model: str = Field(..., description="The ID of the model used to generate the response")
    content: str = Field(..., description="The content of the generated response")
    tokens_used: dict = Field(default={"input": 0, "output": 0}, description="The number of tokens used in the generated response")

class ModelInfo(BaseModel):
    id: str = Field(..., description="The unique identifier for the model")
    model_name: str = Field(..., description="The name of the model")
    provider: Provider = Field(..., description="The provider of the model")
    max_tokens: int = Field(..., description="The maximum number of tokens supported by the model")

# ====== Sample Data ========
MODELS = {
    "gpt-4": ModelInfo(id="gpt-4", model_name="GPT-4", provider=Provider.OPENAI, max_tokens=8192),
    "claude-opus-4-5": ModelInfo(id="claude-opus-4-5", model_name="Claude Opus 4.5", provider=Provider.ANTHROPIC, max_tokens=8192),
    "llama-3.1": ModelInfo(id="llama-3.1", model_name="LLaMA 3.1", provider=Provider.META, max_tokens=4096),
    "gpt-5": ModelInfo(id="gpt-5", model_name="GPT-5", provider=Provider.LITELLM, max_tokens=16384),
    "claude-sonnet-4-5": ModelInfo(id="claude-sonnet-4-5", model_name="Claude Sonnet 4.5", provider=Provider.ANTHROPIC, max_tokens=16384),
}


# ====== Endpoints ========
# ====== Health & Info Endpoints ======
@app.get("/health")
def health_check():
    """Health check endpoint to verify that the API is running."""
    return {"status": "Healthy"}

@app.get("/models")
def list_models(
    provider: Optional[Provider] = Query(None), 
    limit: int = Query(10, ge=1, le=100),
):
    """Endpoint to list available models with optional filtering by provider and pagination."""
    models = list(MODELS.values())
    
    if provider:
        models = [model for model in models if model.provider == provider]
    
    return models[:limit]

@app.get("/models/{model_id}")
def get_model(model_id: str):
    """Endpoint to get details of a specific model by its ID."""
    if model_id not in MODELS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {model_id} not found")
    
    return MODELS[model_id]


# ====== Chat Endpoint ======
@app.post("/chat/completions", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
def create_chat_completion(request: ChatRequest):
    """Endpoint to create a chat completion based on the provided request body."""
    if request.model not in MODELS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {request.model} not found")
    
    # For demonstration purposes, we'll return a dummy response. In a real application, you would integrate with a GenAI model here.
    last_message = request.messages[-1].content
    response = ChatResponse(
        id="response-123",
        model=request.model,
        content=f"This is a generated response based on your input: '{last_message}'",
        tokens_used={"input": sum(len(msg.content.split()) for msg in request.messages), "output": 10}
    )

    return response

@app.post("/providers/{provider_id}/chat", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
def create_provider_chat_completion(
    provider_id: Provider, 
    request: ChatRequest
):
    """Endpoint to create a chat completion for a specific provider."""
    provider_models = [model for model in MODELS.values() if model.provider == provider_id]
    
    if not provider_models:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No models found for provider {provider_id}")
    
    if request.model not in [model.id for model in provider_models]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {request.model} not found for provider {provider_id}")
    
    # For demonstration purposes, we'll return a dummy response. In a real application, you would integrate with a GenAI model here.
    last_message = request.messages[-1].content
    response = ChatResponse(
        id="response-456",
        model=request.model,
        content=f"This is a generated response from provider {provider_id} based on your input: '{last_message}'",
        tokens_used={"input": sum(len(msg.content.split()) for msg in request.messages), "output": 15}
    )

    return response

# ===== Admin Endpoints ======
@app.post("/models", status_code=status.HTTP_201_CREATED, response_model=ModelInfo)
def create_model(model_info: ModelInfo):
    """Register a new model"""
    if model_info.id in MODELS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Model with ID {model_info.id} already exists")
    
    MODELS[model_info.id] = model_info
    return model_info 

@app.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(model_id: str):
    """Delete a model by its ID"""

    if model_id not in MODELS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {model_id} not found")
    
    del MODELS[model_id]
