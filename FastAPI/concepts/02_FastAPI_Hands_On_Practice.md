# FastAPI Hands-On Practice: Build & Test

This guide takes you through **practical implementation** of FastAPI concepts with progressive complexity.

---

## Exercise 1: Hello GenAI World (10 minutes)

### Goal
Create a working FastAPI server and understand the request-response cycle.

### Steps

**1. Create project structure:**
```bash
mkdir -p ~/projects/genai-backend/{app,tests}
cd ~/projects/genai-backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install fastapi uvicorn pydantic
```

**2. Create `app/main.py`:**
```python
from fastapi import FastAPI

app = FastAPI(
    title="GenAI Backend",
    version="0.1.0"
)

@app.get("/")
def root():
    return {"message": "Welcome to GenAI Backend"}

@app.get("/health")
def health():
    return {"status": "healthy"}
```

**3. Run the server:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**4. Test it:**
```bash
# In another terminal
curl http://localhost:8000/
curl http://localhost:8000/health

# Visit interactive docs
# http://localhost:8000/docs
```

### What You Should See

```json
// GET http://localhost:8000/
{"message": "Welcome to GenAI Backend"}

// GET http://localhost:8000/health
{"status": "healthy"}
```

### Understanding the Flow

```
curl request → FastAPI receives → Matches to @app.get("/")
→ Calls root() → Gets {"message": ...} → Converts to JSON
→ Returns HTTP 200 with JSON body → curl displays it
```

---

## Exercise 2: Path Parameters (15 minutes)

### Goal
Extract dynamic values from URLs.

### Extend `app/main.py`

Add these endpoints:

```python
from fastapi import FastAPI
from enum import Enum

app = FastAPI()

# ===== Models =====
class ModelName(str, Enum):
    GPT4 = "gpt-4"
    CLAUDE_SONNET = "claude-sonnet-4"
    LLAMA = "llama-3"

# ===== Endpoints =====

# Simple path parameter
@app.get("/models/{model_id}")
def get_model(model_id: str):
    return {
        "model_id": model_id,
        "type": "model"
    }

# Type-validated path parameter
@app.get("/completions/{completion_id}")
def get_completion(completion_id: int):
    return {
        "completion_id": completion_id,
        "type": int.__name__
    }

# Enum-validated path parameter
@app.get("/supported-models/{model_name}")
def get_supported_model(model_name: ModelName):
    return {
        "model": model_name,
        "available": True
    }

# Multiple path parameters
@app.get("/users/{user_id}/conversations/{conversation_id}")
def get_conversation(user_id: int, conversation_id: int):
    return {
        "user_id": user_id,
        "conversation_id": conversation_id
    }
```

### Test It

```bash
# String parameter
curl http://localhost:8000/models/gpt-4
# → {"model_id": "gpt-4", "type": "model"}

# Integer parameter (type validation)
curl http://localhost:8000/completions/123
# → {"completion_id": 123, "type": "int"}

# Integer parameter with wrong type
curl http://localhost:8000/completions/abc
# → 422 error (Unprocessable Entity)

# Enum validation
curl http://localhost:8000/supported-models/gpt-4
# → {"model": "gpt-4", "available": true}

# Enum with invalid value
curl http://localhost:8000/supported-models/unknown-model
# → 422 error

# Multiple parameters
curl http://localhost:8000/users/123/conversations/456
# → {"user_id": 123, "conversation_id": 456}
```

### Key Takeaway

FastAPI validates types automatically based on your function signature. You get **free validation** just by using type hints.

---

## Exercise 3: Query Parameters (15 minutes)

### Goal
Add filtering, pagination, and optional parameters.

### Extend `app/main.py`

```python
from typing import Optional

@app.get("/models")
def list_models(
    skip: int = 0,
    limit: int = 10,
    provider: Optional[str] = None
):
    """List models with pagination and optional filtering"""
    all_models = [
        {"id": "gpt-4", "provider": "openai", "max_tokens": 8192},
        {"id": "gpt-3.5", "provider": "openai", "max_tokens": 4096},
        {"id": "claude-sonnet-4", "provider": "anthropic", "max_tokens": 200000},
        {"id": "llama-3", "provider": "meta", "max_tokens": 8192},
    ]

    # Filter by provider if specified
    if provider:
        all_models = [m for m in all_models if m["provider"] == provider]

    # Pagination
    models = all_models[skip:skip+limit]

    return {
        "total": len(all_models),
        "skip": skip,
        "limit": limit,
        "models": models
    }

# Combining path + query parameters
@app.get("/providers/{provider_id}/models")
def list_provider_models(
    provider_id: str,
    limit: int = 5
):
    """Get models for a specific provider"""
    return {
        "provider": provider_id,
        "limit": limit,
        "models": [f"model_{i}" for i in range(limit)]
    }
```

### Test It

```bash
# Get all models (with defaults)
curl http://localhost:8000/models
# → 0 skip, 10 limit, all models

# Pagination
curl http://localhost:8000/models?skip=2&limit=2
# → Skip first 2, get next 2

# Filter by provider
curl http://localhost:8000/models?provider=openai
# → Only OpenAI models

# Combine pagination + filtering
curl http://localhost:8000/models?provider=openai&skip=1&limit=1

# Path + query parameters
curl http://localhost:8000/providers/anthropic/models?limit=3
```

### Key Takeaway

Query parameters are optional by default (they have defaults). Required query parameters have no defaults. Type validation still applies—`limit: int = 5` means if you pass `limit=abc`, you get 422 error.

---

## Exercise 4: Request Bodies with Pydantic (20 minutes)

### Goal
Accept structured JSON data and validate it.

### Extend `app/main.py`

```python
from pydantic import BaseModel, Field

# ===== Pydantic Models =====
class Message(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=4096)

class ChatResponse(BaseModel):
    id: str
    model: str
    content: str
    tokens_used: int

# ===== Endpoints =====

@app.post("/chat/completions")
def create_completion(request: ChatRequest):
    """Create a chat completion"""

    # In reality, you'd call the LLM here
    response = ChatResponse(
        id="msg_123",
        model=request.model,
        content=f"Response to: {request.messages[-1].content}",
        tokens_used=42
    )

    return response

# Request body + path parameter
@app.post("/models/{model_id}/evaluate")
def evaluate_model(
    model_id: str,
    request: ChatRequest
):
    """Evaluate a model with test prompts"""
    return {
        "model_id": model_id,
        "message_count": len(request.messages),
        "temperature": request.temperature
    }
```

### Test It

```bash
# Valid request
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hi there!"}
    ]
  }'

# Missing required field (temperature is optional, but try without messages)
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4"}'
# → 422 error: messages is required

# Invalid type
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": 123}],
    "temperature": "hot"
  }'
# → 422 error: content should be str, temperature should be float

# Constraint violation
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hi"}],
    "temperature": 5.0,
    "max_tokens": 50000
  }'
# → 422 error: temperature > 2.0 (violates ge, le), max_tokens > 4096
```

### Key Takeaway

Pydantic models validate:
- Field presence (required vs optional)
- Field types
- Field constraints (ge, le, etc.)
- Nested structures

All validation happens automatically before your function is called. You never see invalid data.

---

## Exercise 5: Response Status Codes (10 minutes)

### Goal
Use appropriate HTTP status codes for different scenarios.

### Extend `app/main.py`

```python
from fastapi import status, HTTPException

@app.post("/models", status_code=status.HTTP_201_CREATED)
def create_model(name: str):
    """Create a new model (returns 201 Created)"""
    return {
        "id": "new_model_123",
        "name": name
    }

@app.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(model_id: str):
    """Delete a model (returns 204 No Content, no response body)"""
    pass  # Return nothing

@app.get("/models/{model_id}")
def get_model_or_404(model_id: str):
    """Return 404 if model not found"""
    models = {"gpt-4": {"name": "GPT-4"}}

    if model_id not in models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )

    return models[model_id]

@app.post("/validate")
def validate_request(request: ChatRequest):
    """Demonstrate validation errors (422)"""
    # Validation happens automatically
    # Invalid requests return 422
    return {"valid": True}
```

### Test It

```bash
# 201 Created
curl -X POST http://localhost:8000/models?name=my-model -v
# → Status: 201 Created

# 204 No Content
curl -X DELETE http://localhost:8000/models/gpt-4 -v
# → Status: 204 No Content (no body)

# 404 Not Found
curl http://localhost:8000/models/nonexistent -v
# → Status: 404 Not Found
# → Body: {"detail": "Model nonexistent not found"}

# 422 Unprocessable Entity
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4"}' -v
# → Status: 422
# → Body: Details about validation errors
```

### Key Status Codes to Remember

```
201 Created    → POST successfully creates resource
204 No Content → Success but no response body (usually DELETE)
400 Bad Request → Malformed request (bad JSON syntax)
404 Not Found   → Resource doesn't exist
422 Unprocessable Entity → Valid JSON but validation failed
500 Internal Server Error → Your code crashed
```

---

## Exercise 6: Combining Everything (30 minutes)

### Goal
Build a complete mini GenAI API.

### Replace entire `app/main.py`

```python
from fastapi import FastAPI, status, HTTPException, Query
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

# ===== App Setup =====
app = FastAPI(
    title="Mini GenAI API",
    description="Multi-provider LLM chat completions",
    version="1.0.0"
)

# ===== Models =====
class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LITELLM = "litellm"

class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message] = Field(..., min_items=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=4096)

class ChatResponse(BaseModel):
    id: str
    model: str
    content: str
    tokens: dict = {"input": 0, "output": 0}

class ModelInfo(BaseModel):
    id: str
    name: str
    provider: Provider
    max_tokens: int

# ===== Database Mock =====
MODELS = {
    "gpt-4": ModelInfo(id="gpt-4", name="GPT-4", provider=Provider.OPENAI, max_tokens=8192),
    "claude-sonnet": ModelInfo(id="claude-sonnet", name="Claude Sonnet", provider=Provider.ANTHROPIC, max_tokens=200000),
    "llama-3": ModelInfo(id="llama-3", name="Llama 3", provider=Provider.LITELLM, max_tokens=8192),
}

# ===== Health & Info Endpoints =====

@app.get("/health")
def health_check():
    """Server health status"""
    return {"status": "healthy"}

@app.get("/models", response_model=list[ModelInfo])
def list_models(
    provider: Optional[Provider] = Query(None),
    limit: int = Query(10, ge=1, le=100)
):
    """List available models"""
    models = list(MODELS.values())

    if provider:
        models = [m for m in models if m.provider == provider]

    return models[:limit]

@app.get("/models/{model_id}", response_model=ModelInfo)
def get_model(model_id: str):
    """Get model details"""
    if model_id not in MODELS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    return MODELS[model_id]

# ===== Main Chat Endpoints =====

@app.post("/chat/completions", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
def create_chat_completion(request: ChatRequest):
    """Create a chat completion"""

    # Validate model exists
    if request.model not in MODELS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {request.model} not found"
        )

    # Mock response
    last_message = request.messages[-1].content
    response = ChatResponse(
        id="msg_123",
        model=request.model,
        content=f"Response to: '{last_message[:50]}'..." if len(last_message) > 50 else f"Response to: '{last_message}'",
        tokens={"input": len(request.messages) * 10, "output": 50}
    )

    return response

@app.post("/providers/{provider_id}/chat")
def chat_with_provider(
    provider_id: Provider,
    request: ChatRequest
):
    """Chat with a specific provider"""

    # Validate model belongs to provider
    model = MODELS.get(request.model)
    if not model or model.provider != provider_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model {request.model} not available for provider {provider_id}"
        )

    return {
        "provider": provider_id,
        "model": request.model,
        "response": f"Handled by {provider_id}"
    }

# ===== Admin Endpoints =====

@app.post("/models", response_model=ModelInfo, status_code=status.HTTP_201_CREATED)
def register_model(model: ModelInfo):
    """Register a new model"""
    if model.id in MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model {model.id} already exists"
        )
    MODELS[model.id] = model
    return model

@app.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def unregister_model(model_id: str):
    """Unregister a model"""
    if model_id not in MODELS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    del MODELS[model_id]
```

### Test It Comprehensively

```bash
# ===== Health & Models =====
curl http://localhost:8000/health

curl http://localhost:8000/models
curl http://localhost:8000/models?provider=openai
curl http://localhost:8000/models/gpt-4

# ===== Chat Completions =====
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "What is REST?"}
    ]
  }'

# ===== Provider-Specific Chat =====
curl -X POST http://localhost:8000/providers/openai/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hi"}],
    "temperature": 0.5
  }'

# ===== Admin Operations =====
# Register new model
curl -X POST http://localhost:8000/models \
  -H "Content-Type: application/json" \
  -d '{
    "id": "new-model",
    "name": "New Model",
    "provider": "litellm",
    "max_tokens": 2000
  }'

# Delete model
curl -X DELETE http://localhost:8000/models/new-model

# ===== Error Cases =====
# 404: Model not found
curl http://localhost:8000/models/nonexistent

# 422: Missing required field
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4"}'

# 400: Model not for provider
curl -X POST http://localhost:8000/providers/anthropic/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hi"}]
  }'
```

### Visit Interactive Docs

```
http://localhost:8000/docs
```

Explore:
- All endpoints listed
- Click "Try it out" on any endpoint
- Fill in parameters and see responses
- See error examples

---

## Testing Best Practices

### Use a `.http` File for Better Testing

Create `test_api.http`:

```http
### Health check
GET http://localhost:8000/health

### List all models
GET http://localhost:8000/models

### List OpenAI models only
GET http://localhost:8000/models?provider=openai

### Get specific model
GET http://localhost:8000/models/gpt-4

### Create completion (valid request)
POST http://localhost:8000/chat/completions
Content-Type: application/json

{
  "model": "gpt-4",
  "messages": [
    {"role": "user", "content": "Explain FastAPI"}
  ],
  "temperature": 0.8
}

### Create completion (validation error - missing messages)
POST http://localhost:8000/chat/completions
Content-Type: application/json

{
  "model": "gpt-4"
}

### Provider-specific chat
POST http://localhost:8000/providers/openai/chat
Content-Type: application/json

{
  "model": "gpt-4",
  "messages": [{"role": "user", "content": "Hi"}]
}
```

Use with VS Code's REST Client extension (Ctrl+Alt+R to send).

---

## Common Mistakes & How to Avoid Them

### ❌ Mistake 1: Forgetting Type Hints

```python
# Wrong - no validation
@app.get("/models/{model_id}")
def get_model(model_id):  # No type hint
    return {"id": model_id}

# Right - validates type
@app.get("/models/{model_id}")
def get_model(model_id: str):  # Type hint enables validation
    return {"id": model_id}
```

### ❌ Mistake 2: Query Params vs Path Params

```python
# Wrong - trying to treat query param as required
@app.get("/models")
def list_models(limit):  # No default, but query param → HTTPException
    return []

# Right - query params should have defaults
@app.get("/models")
def list_models(limit: int = 10):  # Default makes it optional
    return []
```

### ❌ Mistake 3: Not Defining Response Models

```python
# Wrong - docs don't show response structure
@app.get("/models")
def list_models():
    return [{"id": "gpt-4"}]

# Right - docs show shape
@app.get("/models", response_model=list[ModelInfo])
def list_models():
    return [{"id": "gpt-4", "name": "GPT-4", "provider": "openai", "max_tokens": 8192}]
```

### ❌ Mistake 4: Not Validating at Boundaries

```python
# Wrong - trusting client data
@app.post("/chat")
def chat(request: ChatRequest):
    # What if request.messages is empty? What if role is invalid?
    return process_chat(request)

# Right - validate with constraints
class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_items=1)

class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
```

---

## Next Steps

1. **Complete all 6 exercises** - They take ~1.5 hours total
2. **Modify examples** - Change model names, providers, add fields
3. **Break things** - Try invalid inputs, see how FastAPI responds
4. **Check docs** - Visit `/docs` after each change

After these exercises, you'll understand:
- ✅ How FastAPI receives and parses requests
- ✅ How type hints enable automatic validation
- ✅ How Pydantic models structure data
- ✅ How to use proper HTTP status codes
- ✅ How to combine path/query/body parameters

Then you're ready for error handling, middleware, and async operations.

