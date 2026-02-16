# Error Handling Hands-On Practice: Build & Test

This guide takes you through **practical implementation** of error handling patterns with progressive complexity. By the end, you'll have production-ready error handling patterns you can use in real GenAI applications.

Each exercise builds on the previous one. Follow them in order.

---

## Exercise 1: HTTPException for Missing Resources (20 minutes)

### Goal

Use HTTPException to return proper 404 errors when resources don't exist. Understand how FastAPI converts exceptions to HTTP responses.

### Steps

**1. Create `projects/fastapi_concepts_hands_on/07_1_basic_http_exception.py`:**

```python
"""
Exercise 1: HTTPException for Missing Resources

This script demonstrates the basic error handling pattern:
- Return 404 when a resource doesn't exist
- Use HTTPException to control the HTTP status code
- Provide helpful error messages

The model here: "conversations" are chat histories.
When a user asks for a conversation that doesn't exist, we return 404.

Concepts covered:
- HTTPException(status_code, detail)
- Raising exceptions in endpoints
- How FastAPI converts exceptions to JSON responses
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# ===== DATA MODELS =====

class Conversation(BaseModel):
    id: str
    title: str
    message_count: int


# ===== FAKE DATABASE =====
# In real apps, this would be PostgreSQL
conversations_db = {
    "conv_1": {"id": "conv_1", "title": "First chat", "message_count": 5},
    "conv_2": {"id": "conv_2", "title": "Second chat", "message_count": 12},
    "conv_3": {"id": "conv_3", "title": "Third chat", "message_count": 3},
}


# ===== ENDPOINTS =====

@app.get("/conversations/{conversation_id}", response_model=Conversation)
def get_conversation(conversation_id: str):
    """
    Get a conversation by ID.

    Returns 404 if the conversation doesn't exist.
    """
    if conversation_id not in conversations_db:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation '{conversation_id}' not found"
        )

    return conversations_db[conversation_id]


@app.get("/conversations")
def list_conversations():
    """List all conversations"""
    return {"conversations": list(conversations_db.values())}


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    """
    Delete a conversation.

    Returns 404 if the conversation doesn't exist.
    Returns 200 if deletion was successful.
    """
    if conversation_id not in conversations_db:
        raise HTTPException(
            status_code=404,
            detail=f"Cannot delete: conversation '{conversation_id}' not found"
        )

    # Delete it
    del conversations_db[conversation_id]

    return {"message": "Conversation deleted", "id": conversation_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**2. Run the server:**

```bash
cd /path/to/projects/fastapi_concepts_hands_on
python 07_1_basic_http_exception.py
```

### Test It

```bash
# ✅ SUCCESS: Get existing conversation (returns 200 + data)
curl http://localhost:8000/conversations/conv_1

# ❌ ERROR: Get non-existent conversation (returns 404)
curl http://localhost:8000/conversations/unknown

# ✅ SUCCESS: List all
curl http://localhost:8000/conversations

# ✅ SUCCESS: Delete existing
curl -X DELETE http://localhost:8000/conversations/conv_1

# ❌ ERROR: Delete non-existent (returns 404)
curl -X DELETE http://localhost:8000/conversations/unknown
```

### What You Should See

```json
// GET /conversations/conv_1
HTTP 200 OK
{
  "id": "conv_1",
  "title": "First chat",
  "message_count": 5
}

// GET /conversations/unknown
HTTP 404 Not Found
{
  "detail": "Conversation 'unknown' not found"
}

// DELETE /conversations/conv_1
HTTP 200 OK
{
  "message": "Conversation deleted",
  "id": "conv_1"
}

// DELETE /conversations/unknown
HTTP 404 Not Found
{
  "detail": "Cannot delete: conversation 'unknown' not found"
}
```

### Understanding the Pattern

```
User requests GET /conversations/unknown
                    ↓
Endpoint checks: is "unknown" in conversations_db?
                    ↓
No, so raise HTTPException(status_code=404, detail="...")
                    ↓
FastAPI catches the HTTPException
                    ↓
Converts to JSON response:
{
  "status": 404,
  "body": {"detail": "..."},
  "headers": {...}
}
                    ↓
Client receives HTTP 404 + JSON
```

### Key Takeaway

HTTPException is FastAPI's primary way to return error responses. Raise it with the appropriate status code and message, and FastAPI handles the rest. This is perfect for simple errors like "resource not found."

---

## Exercise 2: Custom TokenBudgetExceeded Exception (25 minutes)

### Goal

Create a custom exception class that stores context (tokens used, limit, tokens needed). Understand why custom exceptions are better than HTTPException for complex scenarios.

### Steps

**1. Create `projects/fastapi_concepts_hands_on/07_2_custom_exception_class.py`:**

```python
"""
Exercise 2: Custom Exception Classes

This script demonstrates:
- Creating exception classes that store context
- Raising custom exceptions in business logic
- Why custom exceptions are better than HTTPException for complex errors

The model here: Users have monthly token budgets for LLM API calls.
When they request a chat that would exceed their budget, we raise an exception.

Concepts covered:
- Creating exception classes with __init__
- Storing context as attributes (user_id, tokens_used, tokens_limit)
- Raising exceptions in business logic (not just in endpoints)
- Exception as a communication mechanism between layers
"""

from typing import Optional

# ===== CUSTOM EXCEPTION =====

class GenAIException(Exception):
    """Base exception for all GenAI-specific errors"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class TokenBudgetExceededError(GenAIException):
    """
    Raised when a user would exceed their monthly token budget.

    Attributes:
        user_id: The user who exceeded budget
        tokens_used: Tokens already used this month
        tokens_limit: User's monthly budget
        tokens_needed: Tokens required for current request
    """

    def __init__(
        self,
        user_id: str,
        tokens_used: int,
        tokens_limit: int,
        tokens_needed: Optional[int] = None
    ):
        self.user_id = user_id
        self.tokens_used = tokens_used
        self.tokens_limit = tokens_limit
        self.tokens_needed = tokens_needed

        # Build user-friendly message
        message = (
            f"Token budget exceeded. "
            f"Used {tokens_used}/{tokens_limit} tokens this month. "
        )
        if tokens_needed:
            remaining = tokens_limit - tokens_used
            message += f"Current request needs {tokens_needed}, but only {remaining} remaining."

        super().__init__(message)


# ===== BUSINESS LOGIC =====

def estimate_tokens(message: str) -> int:
    """
    Rough token estimation.
    In reality, use tiktoken library for accurate counts.
    """
    # Very rough: 4 characters ≈ 1 token
    return len(message) // 4


def validate_token_budget(user_id: str, tokens_used: int, tokens_limit: int, request_message: str) -> None:
    """
    Check if user can make this request without exceeding budget.

    Raises:
        TokenBudgetExceededError: If request would exceed budget
    """
    tokens_needed = estimate_tokens(request_message)
    tokens_remaining = tokens_limit - tokens_used

    if tokens_needed > tokens_remaining:
        raise TokenBudgetExceededError(
            user_id=user_id,
            tokens_used=tokens_used,
            tokens_limit=tokens_limit,
            tokens_needed=tokens_needed
        )


# ===== USAGE EXAMPLES =====

def test_valid_request():
    """User has budget - no exception"""
    try:
        # User has used 5000/10000 tokens, needs 100 more
        validate_token_budget(
            user_id="alice",
            tokens_used=5000,
            tokens_limit=10000,
            request_message="Hello" * 50  # ~200 tokens
        )
        print("✅ Valid request: User has sufficient budget")
    except TokenBudgetExceededError as e:
        print(f"❌ {e}")


def test_budget_exceeded():
    """User exceeds budget - exception raised"""
    try:
        # User has used 9500/10000 tokens, needs 500 more (exceeds limit)
        validate_token_budget(
            user_id="bob",
            tokens_used=9500,
            tokens_limit=10000,
            request_message="Hello" * 500  # ~2000 tokens
        )
        print("Request succeeded")
    except TokenBudgetExceededError as e:
        print(f"✅ Exception caught: {e.message}")
        print(f"   User: {e.user_id}")
        print(f"   Tokens used: {e.tokens_used}/{e.tokens_limit}")
        print(f"   Tokens needed: {e.tokens_needed}")


def test_exact_budget():
    """User has exactly enough budget"""
    try:
        # User has used 9900/10000, needs exactly 100
        validate_token_budget(
            user_id="charlie",
            tokens_used=9900,
            tokens_limit=10000,
            request_message="Hello" * 25  # ~100 tokens
        )
        print("✅ Valid request: User has exactly enough budget")
    except TokenBudgetExceededError as e:
        print(f"❌ {e}")


if __name__ == "__main__":
    print("Testing custom exceptions...\n")
    test_valid_request()
    print()
    test_budget_exceeded()
    print()
    test_exact_budget()
```

**2. Run it:**

```bash
python 07_2_custom_exception_class.py
```

### Test It

```
Testing custom exceptions...

✅ Valid request: User has sufficient budget

✅ Exception caught: Token budget exceeded. Used 9500/10000 tokens this month. Current request needs 2000, but only 500 remaining.
   User: bob
   Tokens used: 9500
   Tokens limit: 10000
   Tokens needed: 2000

✅ Valid request: User has exactly enough budget
```

### Understanding the Pattern

```
Business logic: validate_token_budget()
    ↓
Checks: tokens_needed > tokens_remaining?
    ↓
If yes, raise TokenBudgetExceededError(user_id, tokens_used, tokens_limit, tokens_needed)
    ↓
Exception bubbles up to caller
    ↓
Caller can catch and handle (convert to HTTP 429, log, etc)
```

**Benefits of custom exceptions:**

1. **Context stored as attributes**: `exc.user_id`, `exc.tokens_used`, etc.
2. **Type-specific catching**: `except TokenBudgetExceededError` vs generic `except Exception`
3. **Business logic stays clean**: Validation function doesn't know about HTTP
4. **Separation of concerns**: Business logic (check budget) separate from API layer (convert to HTTP)

### Key Takeaway

Custom exceptions let you carry context through your code. The business logic checks budgets and raises exceptions. Later, exception handlers convert those exceptions to HTTP responses. This separation makes code cleaner and more testable.

---

## Exercise 3: Global Exception Handlers (30 minutes)

### Goal

Create global exception handlers that convert custom exceptions to JSON responses with proper status codes and headers. See how handlers centralize error response formatting.

### Steps

**1. Create `projects/fastapi_concepts_hands_on/07_3_global_exception_handlers.py`:**

```python
"""
Exercise 3: Global Exception Handlers

This script demonstrates:
- Defining custom exceptions with stored context
- @app.exception_handler to catch exceptions globally
- Converting exceptions to JSON responses
- Adding headers (Retry-After for rate limits)
- Multiple handlers for different exception types

The model here: A complete chat API with multiple error types.

Concepts covered:
- Exception classes (TokenBudgetExceededError, RateLimitExceededError, etc)
- Global handlers with @app.exception_handler
- JSONResponse for custom error responses
- Status codes and headers
- Handler precedence (specific → general)
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# ===== CUSTOM EXCEPTIONS =====

class GenAIException(Exception):
    """Base for all GenAI errors"""
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class TokenBudgetExceededError(GenAIException):
    """User exceeded token budget"""
    def __init__(self, user_id: str, used: int, limit: int, needed: Optional[int] = None):
        self.user_id = user_id
        self.tokens_used = used
        self.tokens_limit = limit
        self.tokens_needed = needed
        super().__init__(f"Token budget exceeded: {used}/{limit}")


class RateLimitExceededError(GenAIException):
    """User exceeded request rate limit"""
    def __init__(self, limit: int, reset_in: int):
        self.limit = limit
        self.reset_in = reset_in
        super().__init__(f"Rate limit: {limit} requests per minute")


class ProviderError(GenAIException):
    """Error from external LLM provider"""
    def __init__(self, provider: str, status_code: int, message: str, retry_after: int = 60):
        self.provider = provider
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(f"{provider} error: {message}")


class ModelNotFoundError(GenAIException):
    """Requested model doesn't exist"""
    def __init__(self, model_name: str, available: list):
        self.model_name = model_name
        self.available_models = available
        super().__init__(f"Model {model_name} not found")


# ===== EXCEPTION HANDLERS =====

@app.exception_handler(TokenBudgetExceededError)
async def handle_token_budget(request: Request, exc: TokenBudgetExceededError):
    """Convert TokenBudgetExceededError to 429 response"""
    return JSONResponse(
        status_code=429,
        content={
            "error": "token_budget_exceeded",
            "message": "You've used your monthly token budget",
            "tokens_used": exc.tokens_used,
            "tokens_limit": exc.tokens_limit,
            "upgrade_url": "https://app.com/upgrade"
        },
        headers={"Retry-After": "3600"}  # Retry in 1 hour
    )


@app.exception_handler(RateLimitExceededError)
async def handle_rate_limit(request: Request, exc: RateLimitExceededError):
    """Convert RateLimitExceededError to 429 response"""
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": f"Rate limit: {exc.limit} requests per minute",
            "reset_in_seconds": exc.reset_in
        },
        headers={"Retry-After": str(exc.reset_in)}
    )


@app.exception_handler(ProviderError)
async def handle_provider(request: Request, exc: ProviderError):
    """Convert ProviderError to 503 response"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"{exc.provider.lower()}_error",
            "message": f"{exc.provider} API is temporarily unavailable"
        },
        headers={"Retry-After": str(exc.retry_after)}
    )


@app.exception_handler(ModelNotFoundError)
async def handle_model_not_found(request: Request, exc: ModelNotFoundError):
    """Convert ModelNotFoundError to 404 response"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "model_not_found",
            "message": f"Model '{exc.model_name}' not found",
            "available_models": exc.available_models
        }
    )


# ===== DATA MODELS =====

class ChatRequest(BaseModel):
    model: str
    message: str


# ===== ENDPOINTS =====

SUPPORTED_MODELS = ["claude-3", "gpt-4"]
request_count = 0


@app.post("/chat")
def chat(request: ChatRequest):
    """
    Chat endpoint that demonstrates all exception types.

    Raises:
    - ModelNotFoundError: If model doesn't exist
    - TokenBudgetExceededError: If budget exceeded
    - RateLimitExceededError: If rate limited
    - ProviderError: If provider is down
    """
    global request_count

    # 1. Check model exists
    if request.model not in SUPPORTED_MODELS:
        raise ModelNotFoundError(request.model, SUPPORTED_MODELS)

    # 2. Check rate limit (demo: fail after 5 requests)
    request_count += 1
    if request_count > 5:
        raise RateLimitExceededError(limit=5, reset_in=60)

    # 3. Check token budget (demo: fail for long messages)
    if len(request.message) > 100:
        raise TokenBudgetExceededError(
            user_id="user_123",
            used=9500,
            limit=10000,
            needed=len(request.message) // 4
        )

    # 4. Simulate provider error (demo: every 3rd request)
    if request_count % 3 == 0:
        raise ProviderError(
            provider="Anthropic",
            status_code=429,
            message="Rate limited by provider",
            retry_after=60
        )

    # Success
    return {
        "response": f"Response to: {request.message[:50]}...",
        "model": request.model,
        "tokens_used": 100
    }


@app.get("/models")
def list_models():
    """List available models"""
    return {"models": SUPPORTED_MODELS}


@app.get("/health")
def health():
    """Health check"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**2. Run the server:**

```bash
python 07_3_global_exception_handlers.py
```

### Test It

```bash
# ✅ SUCCESS: First chat works
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3", "message": "Hello"}'

# ❌ ERROR: Model not found
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "unknown", "message": "Hi"}'

# ❌ ERROR: Token budget exceeded (long message)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3", "message": "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua"}'

# ❌ ERROR: Rate limit exceeded (after 5 requests)
# Run the first request 5+ times

# ❌ ERROR: Provider error (every 3rd request)
# Run requests 3, 6, 9, etc
```

### What You Should See

```json
// ✅ Success
HTTP 200 OK
{
  "response": "Response to: Hello",
  "model": "claude-3",
  "tokens_used": 100
}

// ❌ Model not found
HTTP 404 Not Found
{
  "error": "model_not_found",
  "message": "Model 'unknown' not found",
  "available_models": ["claude-3", "gpt-4"]
}

// ❌ Token budget exceeded
HTTP 429 Too Many Requests
Retry-After: 3600
{
  "error": "token_budget_exceeded",
  "message": "You've used your monthly token budget",
  "tokens_used": 9500,
  "tokens_limit": 10000,
  "upgrade_url": "https://app.com/upgrade"
}

// ❌ Rate limit exceeded
HTTP 429 Too Many Requests
Retry-After: 60
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit: 5 requests per minute",
  "reset_in_seconds": 60
}

// ❌ Provider error
HTTP 429 Too Many Requests
Retry-After: 60
{
  "error": "anthropic_error",
  "message": "Anthropic API is temporarily unavailable"
}
```

### Understanding the Pattern

```
Endpoint raises TokenBudgetExceededError
                    ↓
FastAPI catches it
                    ↓
Finds @app.exception_handler(TokenBudgetExceededError)
                    ↓
Calls handler(request, exc)
                    ↓
Handler returns JSONResponse(status_code=429, content={...}, headers={...})
                    ↓
Client receives HTTP 429 + custom JSON + Retry-After header
```

**Key observations:**

1. **Handlers centralize error formatting**: All TokenBudgetExceededError responses are formatted the same way
2. **Context matters**: Handlers access `exc.tokens_used`, `exc.tokens_limit`, etc. for rich responses
3. **Headers are important**: `Retry-After` tells clients when to retry
4. **Different exceptions → different responses**: Model not found is 404, budget exceeded is 429

### Key Takeaway

Global exception handlers transform scattered error handling into a centralized system. One handler per exception type. Complex business logic doesn't need to know about HTTP—it just raises the right exception, and handlers take care of the rest.

---

## Exercise 4: Error Logging with Request Context (30 minutes)

### Goal

Add logging to your exception handlers. Include request IDs so you can trace requests through logs. Demonstrate the difference between logs (detailed) and client responses (helpful but generic).

### Steps

**1. Create `projects/fastapi_concepts_hands_on/07_4_error_logging_with_context.py`:**

```python
"""
Exercise 4: Error Logging with Request Context

This script demonstrates:
- Request ID middleware for tracing requests
- Structured logging with extra context
- Logging in exception handlers
- Difference between what you log vs what you show clients
- Using request.state to pass data between middleware and endpoints

The model here: A chat API where every request gets a unique ID.
When something goes wrong, logs include that ID so support can find it.

Concepts covered:
- BaseHTTPMiddleware for adding request IDs
- request.state for storing request-scoped data
- Structured logging with extra={...}
- logger.error() with context
- Separating user-facing messages from internal details
"""

import logging
import uuid
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional

# ===== LOGGING SETUP =====

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# ===== MIDDLEWARE =====

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Add a unique request ID to each request.
    This allows tracing requests through logs.
    """
    async def dispatch(self, request: Request, call_next):
        # Get or create request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Store in request.state (accessible in endpoints and handlers)
        request.state.request_id = request_id
        request.state.user_id = "user_123"  # From auth in real apps
        request.state.start_time = datetime.now()

        # Log incoming request
        logger.info(
            "incoming_request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query": str(request.query_params),
                "client_ip": request.client.host if request.client else "unknown"
            }
        )

        # Call endpoint
        response = await call_next(request)

        # Add request ID to response
        response.headers["X-Request-ID"] = request_id

        # Log response
        elapsed_ms = (datetime.now() - request.state.start_time).total_seconds() * 1000
        logger.info(
            "response_sent",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "elapsed_ms": elapsed_ms,
                "path": request.url.path
            }
        )

        return response


app.add_middleware(RequestIDMiddleware)

# ===== CUSTOM EXCEPTIONS =====

class GenAIException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class TokenBudgetExceededError(GenAIException):
    def __init__(self, user_id: str, used: int, limit: int, needed: Optional[int] = None):
        self.user_id = user_id
        self.tokens_used = used
        self.tokens_limit = limit
        self.tokens_needed = needed
        super().__init__(f"Token budget exceeded: {used}/{limit}")


class ProviderError(GenAIException):
    def __init__(self, provider: str, status_code: int, message: str):
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"{provider} error: {message}")


# ===== EXCEPTION HANDLERS =====

@app.exception_handler(TokenBudgetExceededError)
async def handle_token_budget(request: Request, exc: TokenBudgetExceededError):
    """
    Handle token budget exceeded errors.
    Log detailed information, return generic response to user.
    """
    request_id = request.state.request_id

    # DETAILED LOG (for debugging)
    logger.error(
        "token_budget_exceeded",
        extra={
            "request_id": request_id,
            "user_id": exc.user_id,
            "tokens_used": exc.tokens_used,
            "tokens_limit": exc.tokens_limit,
            "tokens_needed": exc.tokens_needed,
            "endpoint": request.url.path,
            "timestamp": datetime.now().isoformat()
        }
    )

    # GENERIC RESPONSE (for user)
    return JSONResponse(
        status_code=429,
        content={
            "error": "token_budget_exceeded",
            "message": "You've used your monthly token budget",
            "request_id": request_id
        },
        headers={"Retry-After": "3600"}
    )


@app.exception_handler(ProviderError)
async def handle_provider(request: Request, exc: ProviderError):
    """
    Handle provider errors.
    Log details including provider and status code.
    Don't expose provider details to user.
    """
    request_id = request.state.request_id

    # DETAILED LOG
    logger.error(
        f"{exc.provider.lower()}_error",
        extra={
            "request_id": request_id,
            "provider": exc.provider,
            "provider_status_code": exc.status_code,
            "endpoint": request.url.path,
            "user_id": request.state.user_id,
            "timestamp": datetime.now().isoformat()
        }
    )

    # GENERIC RESPONSE
    return JSONResponse(
        status_code=503,
        content={
            "error": "service_unavailable",
            "message": "External service is temporarily unavailable",
            "request_id": request_id
        },
        headers={"Retry-After": "60"}
    )


@app.exception_handler(Exception)
async def handle_unexpected(request: Request, exc: Exception):
    """
    Catch-all for unexpected exceptions.
    Log everything, return generic response.
    """
    request_id = request.state.request_id

    # DETAILED LOG with full traceback
    logger.error(
        "unhandled_exception",
        extra={
            "request_id": request_id,
            "endpoint": request.url.path,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "user_id": request.state.user_id,
            "timestamp": datetime.now().isoformat()
        },
        exc_info=True  # Include full traceback
    )

    # GENERIC RESPONSE
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred",
            "request_id": request_id,
            "support": "Include the request_id above when contacting support"
        }
    )


# ===== DATA MODELS =====

class ChatRequest(BaseModel):
    message: str


# ===== ENDPOINTS =====

@app.post("/chat")
def chat(request: ChatRequest):
    """Chat endpoint that demonstrates logging"""

    # Simulate different errors
    if len(request.message) > 200:
        raise TokenBudgetExceededError(
            user_id="user_123",
            used=9500,
            limit=10000,
            needed=len(request.message) // 4
        )

    if "provider_error" in request.message.lower():
        raise ProviderError(
            provider="Anthropic",
            status_code=429,
            message="Rate limited"
        )

    if "crash" in request.message.lower():
        raise Exception("Intentional error for testing")

    logger.info(
        "chat_request_processed",
        extra={
            "request_id": request.state.request_id,
            "user_id": request.state.user_id,
            "message_length": len(request.message)
        }
    )

    return {
        "response": "Chat response",
        "request_id": request.state.request_id
    }


@app.get("/health")
def health(request: Request):
    """Health check"""
    return {
        "status": "healthy",
        "request_id": request.state.request_id
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**2. Run the server:**

```bash
python 07_4_error_logging_with_context.py
```

### Test It

```bash
# ✅ SUCCESS: Normal request
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello Claude"}'

# ❌ ERROR: Token budget exceeded (long message)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Lorem ipsum dolor sit amet consectetur adipiscing elit."}'

# ❌ ERROR: Provider error
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "This triggers provider_error"}'

# ❌ ERROR: Unexpected exception
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "This will crash"}'

# With custom request ID
curl -X POST http://localhost:8000/chat \
  -H "X-Request-ID: debug-123" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

### What You Should See

**Response (what user sees):**

```json
// ✅ Success
HTTP 200 OK
X-Request-ID: 1a2b3c4d-5e6f-7g8h

{
  "response": "Chat response",
  "request_id": "1a2b3c4d-5e6f-7g8h"
}

// ❌ Token budget
HTTP 429 Too Many Requests
X-Request-ID: 2b3c4d5e-6f7g-8h9i
Retry-After: 3600

{
  "error": "token_budget_exceeded",
  "message": "You've used your monthly token budget",
  "request_id": "2b3c4d5e-6f7g-8h9i"
}

// ❌ Unexpected error
HTTP 500 Internal Server Error
X-Request-ID: 3c4d5e6f-7g8h-9i0j

{
  "error": "internal_error",
  "message": "An unexpected error occurred",
  "request_id": "3c4d5e6f-7g8h-9i0j",
  "support": "Include the request_id above when contacting support"
}
```

**Logs (what you see - detailed):**

```
2024-01-15 10:30:45 - __main__ - INFO - incoming_request - request_id: 1a2b3c4d, method: POST, path: /chat
2024-01-15 10:30:45 - __main__ - INFO - chat_request_processed - request_id: 1a2b3c4d, message_length: 12
2024-01-15 10:30:45 - __main__ - INFO - response_sent - request_id: 1a2b3c4d, status_code: 200, elapsed_ms: 5

2024-01-15 10:30:46 - __main__ - INFO - incoming_request - request_id: 2b3c4d5e, method: POST, path: /chat
2024-01-15 10:30:46 - __main__ - ERROR - token_budget_exceeded - request_id: 2b3c4d5e, user_id: user_123, tokens_used: 9500, tokens_limit: 10000
2024-01-15 10:30:46 - __main__ - INFO - response_sent - request_id: 2b3c4d5e, status_code: 429, elapsed_ms: 3
```

### Understanding the Pattern

```
Request arrives with or without X-Request-ID
                    ↓
RequestIDMiddleware creates unique ID if missing
                    ↓
Stores in request.state.request_id
                    ↓
Endpoint uses request.state.request_id for logging
                    ↓
Handler uses request.state.request_id in logs and response
                    ↓
User can reference request_id in support emails
                    ↓
Support finds request_id in logs and sees full context
```

**Key differences:**

| What You Log | What You Show User |
|---|---|
| user_id | request_id only |
| tokens_used, tokens_limit | Generic "budget exceeded" |
| Full exception traceback | "An error occurred" |
| Provider name and status | "Service unavailable" |
| Endpoint path | Nothing (they know where they clicked) |
| Timestamp | Nothing |

### Key Takeaway

Request IDs are your debugging superpower. Every request gets a unique ID. When something goes wrong, you log that ID with full context. User gets the ID in the response. When they contact support ("My request failed"), you search logs for that ID and see exactly what went wrong.

---

## Exercise 5: Comprehensive Error Response Testing (15 minutes)

### Goal

Write a test script that verifies all your error handling works correctly. Test that errors return proper status codes, headers, and response structure.

### Steps

**1. Create `projects/fastapi_concepts_hands_on/07_5_comprehensive_error_testing.py`:**

```python
"""
Exercise 5: Comprehensive Error Response Testing

This script tests error handling by:
- Making requests that trigger different errors
- Verifying response status codes
- Checking response structure
- Validating headers (like Retry-After)

This is a standalone test script (not a FastAPI app).
It tests the exercise 3 server (07_3_global_exception_handlers.py).

Concepts covered:
- Testing HTTP responses
- Asserting status codes
- Checking response JSON structure
- Verifying headers
- requests library basics
"""

import requests
import json
from typing import Dict, Any

# URL of the server to test
BASE_URL = "http://localhost:8000"


def print_test_header(test_name: str):
    """Print test header for readability"""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")


def print_result(passed: bool, message: str):
    """Print test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: {message}")


def test_health_check():
    """Test that health endpoint works"""
    print_test_header("Health Check")

    response = requests.get(f"{BASE_URL}/health")

    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

    print_result(response.status_code == 200, "Health check returns 200")
    print_result("status" in response.json(), "Response contains 'status' field")


def test_list_models():
    """Test listing available models"""
    print_test_header("List Models")

    response = requests.get(f"{BASE_URL}/models")

    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

    print_result(response.status_code == 200, "List models returns 200")
    print_result("models" in response.json(), "Response contains 'models' field")


def test_valid_chat():
    """Test successful chat request"""
    print_test_header("Valid Chat Request")

    payload = {
        "model": "claude-3",
        "message": "Hello Claude"
    }

    response = requests.post(f"{BASE_URL}/chat", json=payload)

    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

    print_result(response.status_code == 200, "Valid chat returns 200")
    print_result("response" in response.json(), "Response contains 'response' field")
    print_result("model" in response.json(), "Response contains 'model' field")


def test_model_not_found():
    """Test requesting non-existent model"""
    print_test_header("Model Not Found (404)")

    payload = {
        "model": "unknown-model",
        "message": "Hi"
    }

    response = requests.post(f"{BASE_URL}/chat", json=payload)

    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print(f"Headers: {dict(response.headers)}")

    print_result(response.status_code == 404, "Model not found returns 404")
    print_result("error" in response.json(), "Response contains 'error' field")
    print_result(response.json()["error"] == "model_not_found", "Error code is correct")
    print_result("available_models" in response.json(), "Response lists available models")


def test_token_budget_exceeded():
    """Test token budget exceeded error"""
    print_test_header("Token Budget Exceeded (429)")

    # Long message to trigger token budget exceeded
    long_message = "Hello " * 50  # ~300 characters

    payload = {
        "model": "claude-3",
        "message": long_message
    }

    response = requests.post(f"{BASE_URL}/chat", json=payload)

    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print(f"Headers: {dict(response.headers)}")

    print_result(response.status_code == 429, "Token budget exceeded returns 429")
    print_result("error" in response.json(), "Response contains 'error' field")
    print_result(response.json()["error"] == "token_budget_exceeded", "Error code is correct")
    print_result("Retry-After" in response.headers, "Response includes Retry-After header")

    retry_after = response.headers.get("Retry-After")
    print(f"   Retry-After: {retry_after} seconds")


def test_rate_limit_exceeded():
    """Test rate limit exceeded (by making many requests)"""
    print_test_header("Rate Limit Exceeded (429)")

    # Make 6 requests (server limits to 5)
    payload = {
        "model": "claude-3",
        "message": "Hi"
    }

    last_response = None
    for i in range(7):
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        if response.status_code == 429:
            print(f"Got 429 on request {i+1}")
            last_response = response
            break

    if last_response:
        print(f"Status: {last_response.status_code}")
        print(f"Response: {last_response.json()}")

        print_result(last_response.status_code == 429, "Rate limit returns 429")
        print_result("error" in last_response.json(), "Response contains 'error' field")
        print_result(last_response.json()["error"] == "rate_limit_exceeded", "Error code is correct")
        print_result("Retry-After" in last_response.headers, "Response includes Retry-After header")
    else:
        print_result(False, "Could not trigger rate limit (server didn't enforce it)")


def test_provider_error():
    """Test provider error handling"""
    print_test_header("Provider Error (503)")

    payload = {
        "model": "claude-3",
        "message": "Hi"
    }

    # Make requests until we get a provider error (happens every 3rd request)
    for i in range(10):
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        if response.status_code == 429 and "anthropic_error" in response.json().get("error", ""):
            print(f"Got provider error on request {i+1}")
            break

    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

    print_result(response.status_code in [429, 503], "Provider error returns 429 or 503")
    print_result("error" in response.json(), "Response contains 'error' field")
    print_result("Retry-After" in response.headers, "Response includes Retry-After header")


def test_error_response_structure():
    """Test that error responses have consistent structure"""
    print_test_header("Error Response Structure")

    payload = {
        "model": "unknown",
        "message": "Hi"
    }

    response = requests.post(f"{BASE_URL}/chat", json=payload)

    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    body = response.json()

    print_result("error" in body, "Error response contains 'error' field")
    print_result("message" in body, "Error response contains 'message' field")
    print_result(isinstance(body["error"], str), "'error' is a string")
    print_result(isinstance(body["message"], str), "'message' is a string")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("COMPREHENSIVE ERROR HANDLING TEST SUITE")
    print("="*60)
    print(f"Testing server at: {BASE_URL}")
    print("Make sure to run: python 07_3_global_exception_handlers.py")
    print("in another terminal first!\n")

    try:
        # Happy path
        test_health_check()
        test_list_models()
        test_valid_chat()

        # Error cases
        test_model_not_found()
        test_token_budget_exceeded()
        test_error_response_structure()

        # Rate limit and provider (optional, may not trigger)
        print("\n" + "="*60)
        print("OPTIONAL TESTS (may not trigger)")
        print("="*60)
        test_rate_limit_exceeded()
        test_provider_error()

        print("\n" + "="*60)
        print("TEST SUITE COMPLETE")
        print("="*60)

    except requests.ConnectionError:
        print(f"\n❌ Could not connect to {BASE_URL}")
        print("Make sure the server is running!")
        print("Run: python 07_3_global_exception_handlers.py")


if __name__ == "__main__":
    run_all_tests()
```

**2. Run the tests:**

First, make sure the server is running:

```bash
# Terminal 1: Start the server
python 07_3_global_exception_handlers.py

# Terminal 2: Run the tests
pip install requests
python 07_5_comprehensive_error_testing.py
```

### What You Should See

```
============================================================
COMPREHENSIVE ERROR HANDLING TEST SUITE
============================================================
Testing server at: http://localhost:8000
Make sure to run: python 07_3_global_exception_handlers.py in another terminal first!

============================================================
TEST: Health Check
============================================================
Status: 200
Response: {'status': 'healthy'}
✅ PASS: Health check returns 200
✅ PASS: Response contains 'status' field

============================================================
TEST: Valid Chat Request
============================================================
Status: 200
Response: {'response': 'Response to: Hello Claude', 'model': 'claude-3', 'tokens_used': 100}
✅ PASS: Valid chat returns 200
✅ PASS: Response contains 'response' field
✅ PASS: Response contains 'model' field

============================================================
TEST: Model Not Found (404)
============================================================
Status: 404
Response: {'error': 'model_not_found', 'message': "Model 'unknown-model' not found", 'available_models': ['claude-3', 'gpt-4']}
✅ PASS: Model not found returns 404
✅ PASS: Response contains 'error' field
✅ PASS: Error code is correct
✅ PASS: Response lists available models

============================================================
TEST: Token Budget Exceeded (429)
============================================================
Status: 429
Response: {'error': 'token_budget_exceeded', 'message': "You've used your monthly token budget", 'upgrade_url': 'https://app.com/upgrade'}
✅ PASS: Token budget exceeded returns 429
✅ PASS: Response contains 'error' field
✅ PASS: Error code is correct
✅ PASS: Response includes Retry-After header
   Retry-After: 3600 seconds

============================================================
TEST: Error Response Structure
============================================================
Status: 404
Response: {
  "error": "model_not_found",
  "message": "Model 'unknown-model' not found",
  "available_models": ["claude-3", "gpt-4"]
}
✅ PASS: Error response contains 'error' field
✅ PASS: Error response contains 'message' field
✅ PASS: 'error' is a string
✅ PASS: 'message' is a string

============================================================
TEST SUITE COMPLETE
============================================================
```

### Understanding the Pattern

```
Each test:
1. Makes an HTTP request
2. Gets the response
3. Asserts status code is correct
4. Asserts response structure is correct
5. Optionally checks headers
```

**Why test error cases?**

- **Verify status codes**: 404 for not found, 429 for rate limit, 503 for unavailable
- **Check response structure**: Every error has "error" and "message" fields
- **Validate headers**: Retry-After header is present for rate limiting
- **Catch regressions**: When you change code, tests ensure errors still work

### Key Takeaway

Testing error handling is just as important as testing happy paths. Create tests for each exception type. Verify status codes, response structure, and headers. This catches bugs early and gives you confidence your error handling works.

---

## Summary: What You've Built

After completing all 5 exercises, you have:

1. **Exercise 1**: Basic HTTP error responses (404, etc)
2. **Exercise 2**: Custom exception classes with stored context
3. **Exercise 3**: Global exception handlers that convert exceptions to HTTP responses
4. **Exercise 4**: Request ID middleware + structured logging
5. **Exercise 5**: Comprehensive tests that verify error handling works

**This is production-ready error handling.**

### Common Patterns Demonstrated

```python
# 1. Custom exceptions for business logic
try:
    validate_token_budget(user_id, tokens_needed)
except TokenBudgetExceededError as e:
    # Handle it

# 2. Global handlers for HTTP responses
@app.exception_handler(TokenBudgetExceededError)
async def handle(request, exc):
    return JSONResponse(status_code=429, content={...})

# 3. Request IDs for debugging
request.state.request_id = uuid.uuid4()

# 4. Structured logging with context
logger.error("event_name", extra={"request_id": id, ...})

# 5. Testing error responses
response = requests.post("/chat", json=payload)
assert response.status_code == 429
assert response.json()["error"] == "token_budget_exceeded"
```

### Next Steps

1. **Integrate into your chat API**: Add error handling to your existing endpoints
2. **Add more exception types**: Create exceptions for your specific errors
3. **Build the middleware**: Add RequestIDMiddleware to all your FastAPI apps
4. **Test thoroughly**: Write tests for each error scenario

**The pattern is now clear:** Define exceptions → Create handlers → Test everything. This is how you build robust GenAI APIs.

---

*Ready to move to Topic 4: Middleware & Request Lifecycle?*
