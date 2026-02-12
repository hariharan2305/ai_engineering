# FastAPI Core Concepts: Building Your First GenAI Backend

> **Context**: FastAPI is a modern Python framework that makes building REST APIs intuitive and fast. It handles all the HTTP protocol details while you focus on your business logic. Perfect for GenAI backends because it's minimal yet powerful.

---

## Table of Contents

1. [FastAPI vs Flask vs Django](#fastapi-vs-flask-vs-django-why-fastapi)
2. [Application Setup](#application-setup)
3. [Path Operations & Decorators](#path-operations--decorators)
4. [Path Parameters](#path-parameters)
5. [Query Parameters](#query-parameters)
6. [Request Bodies](#request-bodies)
7. [Response Handling](#response-handling)
8. [Complete Working Example](#complete-working-example)
9. [Common Patterns for GenAI](#common-patterns-for-genai-applications)

---

## FastAPI vs Flask vs Django: Why FastAPI?

### Quick Comparison

| Feature | Flask | Django | FastAPI |
|---------|-------|--------|---------|
| **Complexity** | Minimal | Full framework | Minimal + modern |
| **Performance** | Good | Good | Excellent (ASGI) |
| **Type Hints** | Optional | Not encouraged | Required/Built-in |
| **Validation** | Manual | Models | Automatic (Pydantic) |
| **Docs** | Manual | Built-in admin | Auto-generated |
| **Async Support** | Added later | Limited | Native first-class |
| **Learning Curve** | Shallow | Steep | Medium |

### Why FastAPI for GenAI?

```
Your choice: FastAPI

Why?
1. Built-in validation (Pydantic) catches errors before they reach your LLM
2. Native async support (crucial for concurrent API calls to multiple LLM providers)
3. Auto-generated docs (/docs) - useful for testing endpoints during development
4. Type hints make code self-documenting
5. Minimal overhead - you control exactly what happens
6. Perfect for multi-provider setup (Anthropic, OpenAI, LiteLLM, OpenRouter)
```

**Your ML engineer perspective**: FastAPI is like Spark's DataFrame API vs raw RDD operations—higher level abstraction, automatic optimization, but you're not hidden from what's happening.

---

## Application Setup

### 1.2.1 Creating a FastAPI App Instance

#### Installation

```bash
# Install FastAPI and Uvicorn (ASGI server)
pip install fastapi uvicorn[standard]

# Verify
python -c "import fastapi; print(fastapi.__version__)"
```

#### Minimal App

```python
# main.py
from fastapi import FastAPI

# Create application instance
app = FastAPI(
    title="GenAI Backend",
    description="Multi-provider LLM API",
    version="0.1.0"
)

# Define a simple endpoint
@app.get("/")
def read_root():
    return {"message": "Hello, GenAI World!"}

# To run: uvicorn main:app --reload
```

**What's happening:**
- `FastAPI()` creates the application instance (similar to `Flask(__name__)`)
- `@app.get("/")` decorates a function, telling FastAPI this handles GET requests to `/`
- The function returns a dictionary, which FastAPI automatically converts to JSON
- That's it—three lines of logic

### Running with Uvicorn (ASGI Server)

```bash
# Basic startup
uvicorn main:app --reload

# With custom host/port
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Production (no reload, multiple workers)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

**What's `--reload`?**
Watches for file changes and restarts the server. Use in development, disable in production.

**Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
```

Now visit http://localhost:8000 in your browser—you'll see:
```json
{"message": "Hello, GenAI World!"}
```

### Automatic Interactive Docs

One of FastAPI's superpowers: automatic documentation.

```
http://localhost:8000/docs      # Swagger UI (interactive)
http://localhost:8000/redoc     # ReDoc (clean documentation)
http://localhost:8000/openapi.json  # Raw OpenAPI spec
```

**Why this matters:**
- During development, test endpoints without Postman
- For API consumers, they see your API schema automatically
- For CI/CD, the OpenAPI spec enables automated testing

This is generated from your code's type hints and docstrings. Write clean code, get free documentation.

---

## Path Operations & Decorators

### Understanding Decorators in FastAPI

A **decorator** in FastAPI tells the framework "this function handles HTTP requests."

```python
from fastapi import FastAPI

app = FastAPI()

# Decorator syntax: @app.<http_method>(<path>)
@app.get("/")           # GET request to root path
def read_root():
    return {"status": "ok"}

@app.post("/items")     # POST request to /items
def create_item():
    return {"item_id": 1}

@app.put("/items/1")    # PUT request to /items/1
def update_item():
    return {"updated": True}

@app.delete("/items/1") # DELETE request to /items/1
def delete_item():
    return {"deleted": True}
```

**What FastAPI does:**
1. Registers the function with the HTTP method and path
2. When a request arrives, matches the method + path to the right function
3. Calls your function with parsed parameters
4. Converts the return value to JSON

### Available Decorators

```python
@app.get()      # Read data
@app.post()     # Create data
@app.put()      # Replace data
@app.patch()    # Partial update
@app.delete()   # Delete data
@app.options()  # CORS OPTIONS
@app.head()     # Like GET, no body
@app.trace()    # Diagnostic trace
```

For GenAI work, you'll mostly use `@app.get()` and `@app.post()`.

### Multiple Decorators (Same Function, Multiple Paths)

```python
@app.get("/models")
@app.get("/models/list")
def list_models():
    """Handle both /models and /models/list"""
    return {"models": ["gpt-4", "claude-sonnet"]}
```

When a request hits either path, the same function runs.

---

## Path Parameters

### Basic Path Parameters

**Goal**: Make URLs dynamic by extracting parts of the path.

```python
@app.get("/models/{model_id}")
def get_model(model_id: str):
    return {"model_id": model_id}

# Requests:
# GET /models/gpt-4        → returns {"model_id": "gpt-4"}
# GET /models/claude-3     → returns {"model_id": "claude-3"}
```

**How it works:**
1. `{model_id}` in the path is a parameter placeholder
2. The function argument `model_id: str` captures it
3. Type hint `str` tells FastAPI to treat it as a string
4. FastAPI extracts the value from the URL and passes it to your function

### Type Validation with Path Parameters

FastAPI validates types automatically:

```python
@app.get("/messages/{message_id}")
def get_message(message_id: int):
    """message_id must be an integer"""
    return {"message_id": message_id}

# GET /messages/123       → ✅ Works, message_id=123 (int)
# GET /messages/abc       → ❌ Returns 422 error (not an integer)
```

When you request `/messages/abc`, FastAPI returns:
```json
{
  "detail": [
    {
      "type": "value_error.number.not_a_valid_integer",
      "loc": ["path", "message_id"],
      "msg": "value is not a valid integer"
    }
  ]
}
```

**This is powerful**: Your code doesn't need to validate. FastAPI does it based on type hints.

### Predefined Values with Enums

```python
from enum import Enum
from fastapi import FastAPI

class ModelName(str, Enum):
    GPT4 = "gpt-4"
    CLAUDE = "claude-sonnet-4"
    LLAMA = "llama-3"

app = FastAPI()

@app.get("/models/{model_name}")
def get_model(model_name: ModelName):
    """Only accept predefined model names"""
    return {"model_name": model_name}

# GET /models/gpt-4           → ✅ Works
# GET /models/claude-sonnet-4 → ✅ Works
# GET /models/unknown         → ❌ 422 error
```

**Why use Enums?**
1. Type safety (only valid values accepted)
2. Auto-generated docs show allowed values
3. IDE autocomplete works

### Path Parameters with Regex

```python
@app.get("/files/{file_path:path}")
def read_file(file_path: str):
    """
    The :path tells FastAPI to match everything in the path,
    including forward slashes
    """
    return {"file_path": file_path}

# GET /files/documents/2024/report.pdf
# → file_path = "documents/2024/report.pdf"
```

**Without `:path`:**
```python
@app.get("/files/{file_path}")  # Won't work with subdirectories
# GET /files/documents/report.pdf → FastAPI matches only "documents"
```

### Multiple Path Parameters

```python
@app.get("/users/{user_id}/conversations/{conversation_id}")
def get_conversation(user_id: int, conversation_id: int):
    return {
        "user_id": user_id,
        "conversation_id": conversation_id
    }

# GET /users/123/conversations/456
# → user_id=123, conversation_id=456
```

**Order matters**: Parameters appear in order in the URL, and function arguments must match.

---

## Query Parameters

### Basic Query Parameters

**Goal**: Add optional filters and configuration to requests.

```python
@app.get("/models")
def list_models(skip: int = 0, limit: int = 10):
    """
    skip: how many to skip (default 0)
    limit: how many to return (default 10)
    """
    return {
        "skip": skip,
        "limit": limit,
        "models": ["model1", "model2", "model3"]
    }

# Requests:
# GET /models                 → skip=0, limit=10 (defaults)
# GET /models?skip=5          → skip=5, limit=10
# GET /models?limit=20        → skip=0, limit=20
# GET /models?skip=5&limit=20 → skip=5, limit=20
```

**How it works:**
1. Parameters NOT in the path `{}` are query parameters
2. Default values make them optional
3. QueryString passed after `?` in the URL

### Required vs Optional Query Parameters

```python
@app.get("/search")
def search(
    query: str,              # Required (no default)
    provider: str = "openai" # Optional (has default)
):
    return {"query": query, "provider": provider}

# GET /search?query=hello                    → ✅ Works
# GET /search?query=hello&provider=anthropic → ✅ Works
# GET /search                                → ❌ 422 error (query required)
```

### Optional with None

```python
from typing import Optional

@app.get("/completions")
def get_completions(
    model: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
):
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

# GET /completions?model=gpt-4              → temperature=None
# GET /completions?model=gpt-4&temperature=0.7
```

### Type Validation for Query Parameters

```python
@app.get("/items")
def get_items(skip: int = 0, limit: int = 10):
    """Both must be integers"""
    return {"skip": skip, "limit": limit}

# GET /items?skip=5&limit=10   → ✅ Works
# GET /items?skip=abc          → ❌ 422 error (not an integer)
```

### Query Parameters with Lists

```python
@app.get("/users")
def get_users(tags: list[str] = []):
    """Get users with specific tags"""
    return {"tags": tags}

# GET /users?tags=admin&tags=user
# → tags = ["admin", "user"]

# GET /users
# → tags = []
```

### Combining Path and Query Parameters

```python
@app.get("/users/{user_id}/messages")
def get_user_messages(
    user_id: int,           # Path parameter
    skip: int = 0,          # Query parameter
    limit: int = 10,        # Query parameter
    sort: str = "recent"    # Query parameter
):
    return {
        "user_id": user_id,
        "skip": skip,
        "limit": limit,
        "sort": sort
    }

# GET /users/123/messages?skip=0&limit=5&sort=oldest
# → user_id=123 (from path), skip=0, limit=5, sort="oldest" (from query)
```

**Mental model:**
- **Path parameters** (`{...}`) = Which resource (WHERE)
- **Query parameters** (`?...`) = How to process it (FILTER, SORT, PAGINATE)

---

## Request Bodies

### Accepting JSON Request Bodies

**Goal**: Accept structured data from the client (POST/PUT requests).

```python
from pydantic import BaseModel

class Message(BaseModel):
    role: str      # "user" or "assistant"
    content: str   # The actual message

app = FastAPI()

@app.post("/chat")
def create_message(message: Message):
    return {
        "received": message,
        "id": 123
    }
```

**What happens:**
1. Client sends JSON: `{"role": "user", "content": "Hello"}`
2. FastAPI parses it into a `Message` object
3. Your function receives the parsed object
4. You access fields: `message.role`, `message.content`

**Request/Response Example:**

```bash
# Client request
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"role": "user", "content": "Explain REST"}'

# Server response
{
  "received": {
    "role": "user",
    "content": "Explain REST"
  },
  "id": 123
}
```

### Pydantic Models: The Foundation

Pydantic models define the **shape and validation** of your data:

```python
from pydantic import BaseModel, Field

class LLMRequest(BaseModel):
    model: str          # Required string
    prompt: str         # Required string
    max_tokens: int = 100  # Optional, default 100
    temperature: float = 0.7  # Optional, default 0.7

@app.post("/completions")
def create_completion(request: LLMRequest):
    return {"model": request.model, "max_tokens": request.max_tokens}

# Valid request:
# {"model": "gpt-4", "prompt": "Hello"}

# Invalid request (missing required field):
# {"model": "gpt-4"}  → 422 error

# Invalid request (wrong type):
# {"model": "gpt-4", "prompt": "Hello", "temperature": "hot"}
# → 422 error (temperature should be float)
```

### Field Validation with Constraints

```python
from pydantic import BaseModel, Field

class ChatCompletion(BaseModel):
    model: str
    messages: list[dict]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=4096)

# Constraints:
# ge=0.0 → greater than or equal to
# le=2.0 → less than or equal to
# gt, lt for strict greater/less than
```

When you try to create with invalid values:
```python
ChatCompletion(
    model="gpt-4",
    messages=[],
    temperature=5.0  # > 2.0, violates le=2.0
)
# Raises validation error
```

FastAPI automatically returns 422 with error details to the client.

### Nested Models (Complex Structures)

```python
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str

class ConversationRequest(BaseModel):
    model: str
    system_prompt: str
    messages: list[Message]  # Nested list of Messages

@app.post("/conversation")
def handle_conversation(request: ConversationRequest):
    return {
        "model": request.model,
        "message_count": len(request.messages)
    }

# Request:
{
  "model": "claude-sonnet",
  "system_prompt": "You are helpful",
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
  ]
}
```

FastAPI validates the entire nested structure. If any message is missing `role` or `content`, you get a 422 error.

### Multiple Body Parameters

```python
class User(BaseModel):
    username: str
    email: str

class Settings(BaseModel):
    notifications_enabled: bool
    theme: str

@app.put("/users/{user_id}")
def update_user(user_id: int, user: User, settings: Settings):
    return {
        "user_id": user_id,
        "user": user,
        "settings": settings
    }

# Request:
{
  "user": {"username": "john", "email": "john@example.com"},
  "settings": {"notifications_enabled": true, "theme": "dark"}
}
```

FastAPI nests multiple body parameters under their field names in the JSON.

### Request Body + Path + Query Parameters

```python
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 100

@app.post("/models/{model_id}/completions")
def create_completion(
    model_id: str,                          # Path
    request: CompletionRequest,             # Body
    stream: bool = False                    # Query
):
    return {
        "model_id": model_id,
        "stream": stream,
        "max_tokens": request.max_tokens
    }

# Request:
# POST /models/gpt-4/completions?stream=true
# Body: {"prompt": "Hello", "max_tokens": 200}
```

FastAPI intelligently routes parameters to the right place based on location.

---

## Response Handling

### Basic Responses (Auto JSON Conversion)

```python
@app.get("/models")
def list_models():
    return {"models": ["gpt-4", "claude-3"]}  # Dict → JSON
```

FastAPI converts dictionaries to JSON automatically. Response:
```json
{"models": ["gpt-4", "claude-3"]}
```

### Returning Pydantic Models

```python
from pydantic import BaseModel

class Model(BaseModel):
    id: str
    name: str
    provider: str

@app.get("/models/{model_id}", response_model=Model)
def get_model(model_id: str):
    return Model(
        id=model_id,
        name="Claude Sonnet",
        provider="Anthropic"
    )
```

**Why `response_model`?**
1. Validates your response matches the schema
2. Auto-generates docs showing response structure
3. Serializes model to JSON

### Setting HTTP Status Codes

```python
from fastapi import status

@app.post("/models", status_code=status.HTTP_201_CREATED)
def create_model(model_name: str):
    return {"id": 1, "name": model_name}

# Returns 201 Created instead of default 200

@app.get("/models/{model_id}", status_code=status.HTTP_200_OK)
def get_model(model_id: str):
    return {"id": model_id}

# Explicitly return 200 (default anyway)
```

**Common status codes:**
```python
status.HTTP_200_OK              # ✅ Success
status.HTTP_201_CREATED         # ✅ Created
status.HTTP_204_NO_CONTENT      # ✅ Success, no content
status.HTTP_400_BAD_REQUEST     # ❌ Client error
status.HTTP_401_UNAUTHORIZED    # ❌ Auth required
status.HTTP_403_FORBIDDEN       # ❌ Permission denied
status.HTTP_404_NOT_FOUND       # ❌ Not found
status.HTTP_422_UNPROCESSABLE_ENTITY  # ❌ Validation failed
status.HTTP_500_INTERNAL_SERVER_ERROR # 💥 Server error
```

### Custom Response Types

#### JSONResponse

```python
from fastapi.responses import JSONResponse

@app.get("/custom")
def custom_response():
    return JSONResponse(
        status_code=200,
        content={"message": "Custom"},
        headers={"X-Custom-Header": "value"}
    )
```

#### FileResponse (Download Files)

```python
from fastapi.responses import FileResponse

@app.get("/download/report")
def download_report():
    return FileResponse(
        path="/path/to/report.pdf",
        filename="report.pdf"
    )
```

#### StreamingResponse (Large Data)

```python
from fastapi.responses import StreamingResponse
import json

@app.get("/stream")
def stream_data():
    def generate():
        for i in range(1000):
            yield json.dumps({"index": i}) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson"
    )
```

**IMPORTANT for GenAI**: Streaming is crucial for LLM responses.

```python
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        # Stream token by token from LLM
        async for token in llm_client.stream(request.prompt):
            yield f"data: {json.dumps({'token': token})}\n\n"

    return StreamingResponse(generate())
```

### Response with Custom Headers

```python
from fastapi import Response

@app.get("/models")
def list_models():
    content = {"models": [...]}
    return Response(
        content=json.dumps(content),
        status_code=200,
        headers={"X-Total-Count": "5"}
    )
```

Or use `response_model` + inject headers:

```python
@app.get("/models")
def list_models(response: Response):
    response.headers["X-Total-Count"] = "5"
    return {"models": [...]}
```

---

## Complete Working Example

Here's a minimal but complete GenAI backend that uses all concepts:

```python
from fastapi import FastAPI, status
from pydantic import BaseModel, Field
from typing import Optional
import json

# Initialize FastAPI app
app = FastAPI(
    title="GenAI Multi-Provider API",
    description="Chat completions with multiple LLM providers",
    version="1.0.0"
)

# ============== MODELS ==============

class Message(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    model: str = Field(..., description="Model ID")
    messages: list[Message] = Field(..., description="Conversation history")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=4096)
    stream: bool = Field(default=False, description="Stream response?")

class ChatResponse(BaseModel):
    id: str
    model: str
    content: str
    usage: dict = {"input_tokens": 0, "output_tokens": 0}

class Model(BaseModel):
    id: str
    name: str
    provider: str
    max_tokens: int

# ============== ENDPOINTS ==============

# GET: List all available models
@app.get("/models", response_model=list[Model])
def list_models(provider: Optional[str] = None):
    """List available models, optionally filtered by provider"""
    all_models = [
        Model(id="gpt-4", name="GPT-4", provider="openai", max_tokens=8192),
        Model(id="claude-sonnet-4", name="Claude Sonnet 4", provider="anthropic", max_tokens=200000),
        Model(id="llama-3", name="Llama 3", provider="meta", max_tokens=8192),
    ]

    if provider:
        return [m for m in all_models if m.provider == provider]
    return all_models

# GET: Get specific model details
@app.get("/models/{model_id}", response_model=Model)
def get_model(model_id: str):
    """Get details for a specific model"""
    models = {
        "gpt-4": Model(id="gpt-4", name="GPT-4", provider="openai", max_tokens=8192),
        "claude-sonnet-4": Model(id="claude-sonnet-4", name="Claude Sonnet 4", provider="anthropic", max_tokens=200000),
    }

    if model_id not in models:
        return {"error": "Model not found"}  # Would be proper error handling

    return models[model_id]

# POST: Create a chat completion
@app.post("/chat/completions", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
def create_completion(request: ChatRequest):
    """Create a chat completion with the specified model"""

    # In real implementation, call actual LLM API
    # This is a mock response
    response = ChatResponse(
        id="msg_123",
        model=request.model,
        content="This is a mock response from " + request.model,
        usage={"input_tokens": 10, "output_tokens": 15}
    )

    return response

# POST: Chat with streaming (Server-Sent Events)
@app.post("/chat/completions/stream")
def stream_completion(request: ChatRequest):
    """Stream a chat completion token by token"""
    from fastapi.responses import StreamingResponse

    async def generate():
        # Mock streaming
        response_text = "This is a mock streaming response from " + request.model
        for token in response_text.split():
            yield f"data: {json.dumps({'token': token + ' '})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# GET: Health check
@app.get("/health")
def health_check():
    """Check if the service is running"""
    return {"status": "healthy"}

# ============== RUN ==============
# Command: uvicorn filename:app --reload
# Visit: http://localhost:8000/docs
```

**Test it:**

```bash
# Start server
uvicorn main:app --reload

# List models
curl http://localhost:8000/models

# Get specific model
curl http://localhost:8000/models/gpt-4

# Create completion
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 500
  }'

# Visit interactive docs
# http://localhost:8000/docs
```

---

## Common Patterns for GenAI Applications

### Pattern 1: Multi-Provider Dispatcher

```python
from enum import Enum

class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LITELLM = "litellm"

@app.post("/completions/{provider}")
def create_completion(
    provider: Provider,
    request: ChatRequest
):
    """Route to the appropriate provider"""

    if provider == Provider.OPENAI:
        return call_openai(request)
    elif provider == Provider.ANTHROPIC:
        return call_anthropic(request)
    elif provider == Provider.LITELLM:
        return call_litellm(request)
```

### Pattern 2: Async for Concurrent Calls

```python
from fastapi import FastAPI
import asyncio

app = FastAPI()

@app.post("/compare-providers")
async def compare_providers(request: ChatRequest):
    """Call multiple providers concurrently"""

    # Async calls happen in parallel
    results = await asyncio.gather(
        call_openai_async(request),
        call_anthropic_async(request),
        call_litellm_async(request)
    )

    return {"results": results}
```

### Pattern 3: Response Caching

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_model_config(model_id: str):
    """Cache model configurations"""
    return load_model_config(model_id)

@app.get("/models/{model_id}/config")
def get_config(model_id: str):
    return get_model_config(model_id)
```

### Pattern 4: Pagination

```python
class PaginatedResponse(BaseModel):
    items: list
    total: int
    skip: int
    limit: int

@app.get("/completions", response_model=PaginatedResponse)
def list_completions(skip: int = 0, limit: int = 10):
    """Get paginated list of completions"""
    all_items = load_all_completions()
    total = len(all_items)
    items = all_items[skip:skip+limit]

    return PaginatedResponse(
        items=items,
        total=total,
        skip=skip,
        limit=limit
    )
```

### Pattern 5: Error Handling (Preview)

```python
from fastapi import HTTPException

@app.get("/models/{model_id}")
def get_model(model_id: str):
    try:
        return fetch_model(model_id)
    except ModelNotFound:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_id} not found"
        )
    except ProviderError:
        raise HTTPException(
            status_code=503,
            detail="Provider temporarily unavailable"
        )
```

---

## Key Insights for Your Learning

`★ Insight ─────────────────────────────────────`

1. **Type hints are validation rules**: When you write `temperature: float = Field(..., ge=0.0, le=2.0)`, FastAPI doesn't just check that it's a float—it validates the constraints too. This is like Spark's DataFrame schema enforcement, but for HTTP.

2. **Path parameters identify resources, query parameters filter them**: This maps directly to your SQL experience. `GET /users/123/messages?sort=recent` is like `SELECT * FROM messages WHERE user_id=123 ORDER BY created_at DESC`. The path identifies WHAT, the query specifies HOW.

3. **Async is critical for multi-provider setups**: When calling multiple LLM providers (Anthropic, OpenAI, etc.), you want concurrent HTTP calls. Async allows your single FastAPI instance to handle hundreds of requests by not blocking while waiting for external APIs. This is like Spark's lazy evaluation—you don't block on each task.

4. **Pydantic models are your single source of truth**: Define the shape of your data once in a Pydantic model, and FastAPI handles validation, documentation, and serialization. Change the model, your docs auto-update. This beats manual JSON schema validation by far.

`─────────────────────────────────────────────────`

---

## Next Steps

Your learning path:

1. ✅ HTTP & REST API Basics (completed)
2. ✅ FastAPI Core Concepts (current)
3. **Next**: Error Handling & Custom Responses
4. **Then**: Middleware & Request Lifecycle
5. **Then**: Async FastAPI for Concurrent LLM Calls
6. **Then**: Authentication & API Keys
7. **Then**: Integrating Real LLM Providers

Ready to practice, or dive deeper into any concept?

