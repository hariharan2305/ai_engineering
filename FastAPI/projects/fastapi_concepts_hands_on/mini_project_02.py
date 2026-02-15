"""
This is a mini project to demonstrate the concepts of FastAPI, similar to mini_project_01 but with the addition of using dependencies to manage common logic such as authentication, settings retrieval, and model validation. The API will have endpoints for health checks, listing available models, getting model details, creating chat completions based on user input, and admin operations for managing models. The responses will be dummy data for demonstration purposes.

Concepts covered in this mini project:
1. Using FastAPI to create a simple API with multiple endpoints
2. Using Pydantic data models for request validation and response modeling
3. Using Enums for better validation and documentation of query parameters
4. Implementing basic error handling with HTTP exceptions
5. Using status codes to indicate the result of API operations
6. Organizing endpoints logically (health/info endpoints, chat endpoints, admin endpoints)
7. Using dependencies to manage common logic (e.g., authentication, settings retrieval, model validation)

This project utilized dependencies to achieve the following:
- Authenticate users based on API keys provided in the request headers
- Retrieve application settings that can be used across multiple endpoints
- Validate model IDs before processing requests that require a valid model
- Validate the user tier for access control to certain features (e.g., max tokens limit for free tier users, admin privileges for model management)
- Validate the user's admin privileges for accessing admin endpoints

Dependencies make endpoints cleaner by:
- Moving validation logic out of endpoints
- Providing pre-validated data to your functions
- Enabling code reuse across multiple endpoints
- Making the dependency graph explicit (readable)
- Your endpoint functions now focus only on business logic, not infrastructure concerns.

"""

from fastapi import FastAPI, Depends, HTTPException, status, Query, Header
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

# Create an instance of the FastAPI class
app = FastAPI(
    title="GenAI Backend - Mini Project 02 (without any LLM integration + Dependency Injection)",
    version="0.0.1"
)

# ====== Models ========
class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    META = "meta"
    LITELLM = "litellm"

class Settings(BaseModel):
    DEFAULT_MODEL: str = "gpt-4"
    TEMPERATURE: Optional[float] = 0.7
    MAX_TOKENS: Optional[int] = 1000
    RATE_LIMIT: Optional[int] = 10

class User(BaseModel):
    id: int
    email: str
    tier: str

class Message(BaseModel):
    role: str = Field(..., description="The role of the message sender (e.g., user, assistant)", pattern="^(user|assistant|system)$")
    content: str = Field(..., description="The content of the message")

class ChatRequest(BaseModel):
    model: Optional[str] = Field(default=None, description="The ID of the model to use for generating the response")
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

USERS = {
    "key-123": User(id=1, email="user1@example.com", tier="free"),
    "key-456": User(id=2, email="user2@example.com", tier="pro"),
    "admin-123": User(id=000, email="admin@example.com", tier="admin"),
}

# ====== Dependency Function ========
def authenticate_user(x_api_key: str = Header(...)) -> User:
    """Authenticate the user based on the provided API key."""
    print(f"Authenticating user with API key: {x_api_key}") 
    
    if x_api_key not in USERS:
        print("Authentication failed: Invalid API key")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    
    print("Authentication successful")
    return USERS[x_api_key]

def get_settings() -> Settings:
    """Return application settings."""
    print("Fetching application settings")
    return Settings()

def validate_model(model_id: str) -> ModelInfo:
    """Validate the requested model ID and return the model information."""
    print(f"Validating model ID: {model_id}")
    
    if model_id not in MODELS:
        print(f"Model validation failed: Model {model_id} not found")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {model_id} not found")
    
    print(f"Model validation successful: Found model {model_id}")
    return MODELS[model_id]

def validate_admin(user: User = Depends(authenticate_user)):
    """Validate that the user has admin privileges."""
    print(f"Validating admin privileges for user: {user.email}")
    
    if user.tier != "admin":
        print(f"Admin validation failed: User {user.email} does not have admin privileges")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")
    
    print(f"Admin validation successful for user: {user.email}")
    return user


# ====== Endpoints ========
# ====== Health, Info and Public Endpoints ======
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/models", response_model=list[ModelInfo])
def list_models(
    provider: Optional[Provider] = Query(None, description="Filter models by provider"),
    limit: int = Query(10, ge=1, le=100, description="Limit the number of models returned (between 1 and 100)"),
    user: User = Depends(authenticate_user) # Require authentication to access this endpoint
):
    """Endpoint to list available models with optional filtering by provider and pagination."""
    print(f"User {user.email} is requesting the list of models with provider filter: {provider} and limit: {limit}")
    
    models = list(MODELS.values())
    
    if provider:
        models = [model for model in models if model.provider == provider]
    
    return models[:limit]

@app.get("/models/{model_id}", response_model=ModelInfo)
def get_model(model_id: str, user: User = Depends(authenticate_user), model: ModelInfo = Depends(validate_model)): # validation of model by dependency function
    """Endpoint to get details of a specific model by its ID."""
    print(f"User {user.email} is requesting details for model ID: {model_id}")
    return model

@app.post("/chat/completions", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
def create_chat_completion(
    request: ChatRequest, 
    user: User = Depends(authenticate_user), # Require authentication to access this endpoint
    settings: Settings = Depends(get_settings), # Use dependency to get application settings    
):
    """Endpoint to create a chat completion based on the provided request body."""
    print(f"User {user.email} is creating a chat completion with model: {request.model}")

    # Validate the requested model
    model_info = validate_model(request.model or settings.DEFAULT_MODEL)

    # Tier-based validation for max tokens
    if user.tier == "free" and request.max_tokens > 1000:
        print(f"User {user.email} has exceeded the max tokens limit for free tier")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Free tier users can only request up to 1000 max tokens")
    
    # For demonstration purposes, we'll return a dummy response. In a real application, you would integrate with a GenAI model here.
    last_message = request.messages[-1].content
    response = ChatResponse(
        id="response-123",
        model=model_info.id,
        content=f"This is a generated response based on your input: '{last_message}'",
        tokens_used={"input": sum(len(msg.content.split()) for msg in request.messages), "output": 10}
    )

    return response

# ===== Admin Endpoint ======
@app.post("/models", response_model=ModelInfo, status_code=status.HTTP_201_CREATED)
def create_model(model: ModelInfo, user: User = Depends(validate_admin)):
    """Admin endpoint to create a new model."""
    print(f"Admin User {user.email} is attempting to create a new model with ID: {model.id}")
    
    if model.id in MODELS:
        print(f"Model creation failed: Model with ID {model.id} already exists")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Model with ID {model.id} already exists")
    
    MODELS[model.id] = model
    print(f"Model with ID {model.id} created successfully by user {user.email}")
    return model

@app.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(model_id: str, user: User = Depends(validate_admin)):
    """Admin endpoint to delete a model by its ID."""
    print(f"Admin User {user.email} is attempting to delete model with ID: {model_id}")
    
    if model_id not in MODELS:
        print(f"Model deletion failed: Model with ID {model_id} does not exist")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model with ID {model_id} does not exist")
    
    del MODELS[model_id]
    print(f"Model with ID {model_id} deleted successfully by user {user.email}")
