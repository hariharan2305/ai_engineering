# Error Handling Basics: Building Robust GenAI APIs

> **Context**: LLM API calls fail frequently—rate limits, timeouts, provider outages, exhausted token budgets. Your GenAI backend must handle these failures gracefully. Without proper error handling, a single failed LLM call crashes your entire request. With it, users get actionable feedback and your system degrades gracefully.

---

## Table of Contents

1. [Why Error Handling Matters for GenAI](#why-error-handling-matters-for-genai)
2. [HTTPException: FastAPI's Built-in Error System](#httpexception-fastapis-built-in-error-system)
3. [Custom Exception Classes](#custom-exception-classes)
4. [Global Exception Handlers](#global-exception-handlers)
5. [Error Logging Best Practices](#error-logging-best-practices)
6. [Complete Working Example](#complete-working-example)
7. [Common Patterns for GenAI Applications](#common-patterns-for-genai-applications)
8. [Key Insights for Your Learning](#key-insights-for-your-learning)
9. [Quick Reference](#quick-reference)
10. [Next Steps](#next-steps)

---

## Why Error Handling Matters for GenAI

### Real-World Failure Rates

LLM APIs fail more frequently than traditional backend services:

```
Error Type                  Frequency       Cost/Impact
─────────────────────────────────────────────────────────
Rate limit (429)            ~5-10%          User pays for failed request
Timeout (>30s)              ~2-5%           Request hangs, user leaves
Provider outage (503)        ~1-2%           Entire app unavailable
Token budget exceeded        1-3%            User can't use service
Invalid request (400)        ~1-2%           Bad data from frontend
```

**Real scenario:**

```
❌ WITHOUT error handling:
  User sends chat request
  → Your code calls Claude API
  → Claude API rate limited (429)
  → Python raises exception
  → Exception crashes entire request
  → User sees blank error or app freezes
  → You never know what went wrong

✅ WITH error handling:
  User sends chat request
  → Your code calls Claude API
  → Claude API rate limited (429)
  → Exception handler catches it
  → Returns {"error": "Rate limited. Try again in 60s", "retry_after": 60}
  → User sees friendly message + clear next action
  → Logs show exactly what happened for debugging
```

### Why This Matters for Your GenAI App

| Scenario | No Error Handling | With Error Handling |
|----------|-------------------|---------------------|
| **User Experience** | Blank "500 error" page | "Claude is busy, try again in 60 seconds" |
| **Your Debugging** | No idea what failed | Full context: which API, which status code, which user |
| **Provider Resilience** | One provider down = app down | Can failover to alternative provider |
| **Cost Control** | Failed requests still charge | Can retry cheaply or fail fast |
| **Logging** | Nothing logged | Full audit trail |

### Statistics That Matter

- **47%** of users abandon an app after a single bad error message (vs. 8% with clear error feedback)
- **89%** of production bugs involve missing error handling
- **LLM APIs** fail 5-10% more often than traditional APIs due to rate limits and timeouts

Your error handling isn't optional—it's part of your product.

---

## HTTPException: FastAPI's Built-in Error System

### What Is HTTPException?

HTTPException is FastAPI's way to return error responses. It maps Python exceptions to HTTP status codes + JSON responses.

### Basic Usage: 404 Not Found

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

conversations = {
    "conv_1": {"id": "conv_1", "title": "First chat"},
    "conv_2": {"id": "conv_2", "title": "Second chat"}
}

# ✅ CORRECT: Using HTTPException for missing resources
@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    if conversation_id not in conversations:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )
    return conversations[conversation_id]
```

**What happens when you request `/conversations/unknown`:**

```json
HTTP 404 Not Found
{
  "detail": "Conversation not found"
}
```

### HTTP Status Codes for GenAI

```python
from fastapi import HTTPException

# 404 Not Found - The resource doesn't exist
# Use: Missing conversation, model not available, user profile deleted
raise HTTPException(status_code=404, detail="Model gpt-4 not found in our system")

# 400 Bad Request - Client sent invalid data (but Pydantic usually catches this)
# Use: Invalid JSON structure your validation doesn't catch
raise HTTPException(status_code=400, detail="Messages must be non-empty array")

# 422 Unprocessable Entity - Pydantic validation failed
# FastAPI handles this automatically - you rarely raise it manually
raise HTTPException(status_code=422, detail="temperature must be between 0 and 1")

# 429 Too Many Requests - Rate limited or quota exceeded
# Use: User has sent too many requests, token budget exceeded
raise HTTPException(status_code=429, detail="Rate limited. Try again in 60 seconds")

# 500 Internal Server Error - Your code crashed
# Don't raise this manually - FastAPI does when exceptions occur
# Let unhandled exceptions become 500s as a last resort

# 503 Service Unavailable - External service is down
# Use: LLM provider is down, database is unreachable
raise HTTPException(status_code=503, detail="Claude API is currently unavailable")
```

### Structured Error Details (Beyond Just Strings)

HTTPException's `detail` parameter accepts strings OR dictionaries. Dictionaries are useful for rich error information:

```python
# ❌ WRONG: Just a string
raise HTTPException(
    status_code=400,
    detail="Invalid message format"
)

# ✅ CORRECT: Structured detail with context
raise HTTPException(
    status_code=400,
    detail={
        "error": "invalid_message_format",
        "message": "Each message must have 'role' and 'content' fields",
        "example": {"role": "user", "content": "Hello"},
        "received": received_data
    }
)
```

Response the client receives:

```json
HTTP 400 Bad Request
{
  "detail": {
    "error": "invalid_message_format",
    "message": "Each message must have 'role' and 'content' fields",
    "example": {"role": "user", "content": "Hello"},
    "received": null
  }
}
```

### Adding Headers: Retry-After for Rate Limits

When rate limiting, tell clients how long to wait before retrying:

```python
from fastapi import HTTPException

@app.post("/chat")
def chat(request: ChatRequest):
    if user_exceeded_rate_limit(request.user_id):
        # Add Retry-After header so clients know to wait
        raise HTTPException(
            status_code=429,
            detail="Rate limited. You can make 10 requests per minute.",
            headers={"Retry-After": "60"}
        )

    return generate_response(request)
```

HTTP clients respect this:

```json
HTTP 429 Too Many Requests
Retry-After: 60
Content-Type: application/json

{
  "detail": "Rate limited. You can make 10 requests per minute."
}
```

### ✅ Correct vs ❌ Incorrect HTTPException Usage

#### Example 1: Missing Resource

```python
# ❌ WRONG: Returns 500 error
@app.get("/models/{model_name}")
def get_model(model_name: str):
    model = db.query_model(model_name)
    return model.to_dict()  # Crashes if model is None

# ✅ CORRECT: Returns 404 error with clear message
@app.get("/models/{model_name}")
def get_model(model_name: str):
    model = db.query_model(model_name)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available models: gpt-4, claude-3, llama-2"
        )
    return model.to_dict()
```

#### Example 2: Token Budget Check

```python
# ❌ WRONG: No budget checking
@app.post("/chat")
def chat(request: ChatRequest, user: User = Depends(get_current_user)):
    response = call_claude_api(request.messages)
    return response

# ✅ CORRECT: Check budget before calling API
@app.post("/chat")
def chat(request: ChatRequest, user: User = Depends(get_current_user)):
    tokens_needed = estimate_tokens(request.messages)
    tokens_remaining = user.token_budget - user.tokens_used

    if tokens_needed > tokens_remaining:
        raise HTTPException(
            status_code=429,  # 429 means "quota exceeded" in our context
            detail={
                "error": "token_budget_exceeded",
                "tokens_needed": tokens_needed,
                "tokens_remaining": tokens_remaining,
                "upgrade": "https://yourapp.com/upgrade"
            },
            headers={"Retry-After": "3600"}  # Retry in an hour when budget resets
        )

    response = call_claude_api(request.messages)
    return response
```

#### Example 3: Provider Down

```python
# ❌ WRONG: Let raw exceptions crash the request
@app.post("/chat")
def chat(request: ChatRequest):
    response = requests.post("https://api.anthropic.com/...", ...)
    return response

# ✅ CORRECT: Catch provider errors and return clear status
@app.post("/chat")
def chat(request: ChatRequest):
    try:
        response = requests.post("https://api.anthropic.com/...", timeout=30)
    except requests.Timeout:
        raise HTTPException(
            status_code=503,
            detail="Claude API is not responding. Try again in a few seconds."
        )
    except requests.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot reach Claude API. Our service may be experiencing issues."
        )

    return response
```

### When to Use HTTPException

```
Use HTTPException for:
✅ Expected errors (missing resources, validation failures, rate limits)
✅ When you want to control the HTTP status code
✅ When you want to return a specific error message to the client
✅ Quick one-off error responses

Don't use HTTPException for:
❌ Unexpected errors (let them propagate)
❌ Multiple related errors (use custom exceptions instead)
❌ Complex error logic (use custom exception handlers)
❌ Errors that need context stored (use custom exceptions)
```

---

## Custom Exception Classes

### Why Custom Exceptions?

HTTPException is great for simple cases. But what about complex scenarios?

```python
# ❌ PROBLEM: Using HTTPException everywhere gets messy
@app.post("/chat")
def chat(request: ChatRequest):
    if provider_down:
        raise HTTPException(status_code=503, detail="Provider down")
    if rate_limited:
        raise HTTPException(status_code=429, detail="Rate limited", headers=...)
    if tokens_exceeded:
        raise HTTPException(status_code=429, detail="Token budget exceeded", headers=...)
    # ... more if statements with HTTPExceptions

# ✅ SOLUTION: Custom exceptions for GenAI-specific errors
@app.post("/chat")
def chat(request: ChatRequest):
    if provider_down:
        raise ProviderUnavailableError("Claude", timeout=5)
    if rate_limited:
        raise RateLimitExceededError(limit=100, reset_in=60)
    if tokens_exceeded:
        raise TokenBudgetExceededError(used=10000, limit=5000)
    # Business logic stays clean, error handling is separate
```

### Exception Hierarchy for GenAI

Design your exceptions like this:

```
GenAIException (base class)
├── ProviderError
│   ├── ProviderUnavailableError (provider's API is down)
│   ├── ProviderRateLimitError (provider rate limited our API key)
│   └── InvalidProviderResponseError (unexpected response format)
├── BudgetError
│   ├── TokenBudgetExceededError (user's token quota exceeded)
│   └── RateLimitExceededError (user's request rate limit exceeded)
├── ValidationError
│   ├── InvalidMessageFormatError (message structure wrong)
│   └── ModelNotFoundError (requested model doesn't exist)
└── SystemError
    └── DatabaseUnavailableError (database connection failed)
```

### Example: TokenBudgetExceededError

```python
from typing import Optional

class GenAIException(Exception):
    """Base exception for all GenAI-specific errors"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class BudgetError(GenAIException):
    """Base for budget-related errors (tokens, requests)"""
    pass


class TokenBudgetExceededError(BudgetError):
    """Raised when a user exceeds their token budget"""

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
        self.tokens_needed = tokens_needed  # Tokens for current request

        message = (
            f"Token budget exceeded. "
            f"User {user_id} has used {tokens_used}/{tokens_limit} tokens. "
        )
        if tokens_needed:
            message += f"Current request needs {tokens_needed} more."

        super().__init__(message)
```

### Using Custom Exceptions

```python
from datetime import datetime, timedelta

def validate_token_budget(user_id: str, tokens_needed: int) -> None:
    """
    Check if user has enough token budget for this request.

    Raises:
        TokenBudgetExceededError: If user would exceed their budget
    """
    user = db.get_user(user_id)
    tokens_remaining = user.token_budget - user.tokens_used

    if tokens_needed > tokens_remaining:
        raise TokenBudgetExceededError(
            user_id=user_id,
            tokens_used=user.tokens_used,
            tokens_limit=user.token_budget,
            tokens_needed=tokens_needed
        )


@app.post("/chat")
async def chat(request: ChatRequest, user: User = Depends(get_current_user)):
    """Endpoint that validates budget before calling LLM"""

    # Estimate tokens for the request
    tokens_needed = estimate_tokens(request.messages)

    # Raise TokenBudgetExceededError if needed (will be caught by handler)
    validate_token_budget(user.id, tokens_needed)

    # If we get here, budget is OK - call LLM
    response = await call_claude_api(request.messages, user.model)
    return response
```

### More GenAI-Specific Exceptions

```python
class ProviderError(GenAIException):
    """Error from LLM provider (Anthropic, OpenAI, etc)"""

    def __init__(
        self,
        provider: str,  # "Anthropic", "OpenAI", "HuggingFace"
        status_code: int,
        message: str,
        retry_after: Optional[int] = None
    ):
        self.provider = provider
        self.status_code = status_code
        self.retry_after = retry_after
        full_message = f"{provider} API error (HTTP {status_code}): {message}"
        super().__init__(full_message)


class RateLimitExceededError(BudgetError):
    """User has exceeded their request rate limit"""

    def __init__(
        self,
        limit: int,
        reset_in: int,  # seconds until limit resets
        user_id: Optional[str] = None
    ):
        self.limit = limit
        self.reset_in = reset_in
        self.user_id = user_id

        message = (
            f"Rate limit exceeded: {limit} requests per minute. "
            f"Try again in {reset_in} seconds."
        )
        super().__init__(message)


class ModelNotFoundError(GenAIException):
    """Requested model doesn't exist or isn't available"""

    def __init__(self, model_name: str, available_models: list[str]):
        self.model_name = model_name
        self.available_models = available_models

        message = (
            f"Model '{model_name}' not found. "
            f"Available models: {', '.join(available_models)}"
        )
        super().__init__(message)
```

### Key Design Principles

1. **Store Context**: Add attributes to exceptions so handlers can access the data
   ```python
   # ❌ Wrong: Just a string
   raise Exception("Rate limit exceeded")

   # ✅ Right: Store all context
   raise RateLimitExceededError(limit=100, reset_in=60, user_id="user_123")
   ```

2. **One Exception = One Problem**: Don't create mega-exceptions for everything
   ```python
   # ❌ Wrong: One giant exception
   class GenAIError(Exception):
       def __init__(self, error_type, status_code, message, ...): pass

   # ✅ Right: Specific exceptions for specific problems
   class TokenBudgetExceededError(BudgetError): pass
   class RateLimitExceededError(BudgetError): pass
   class ModelNotFoundError(ValidationError): pass
   ```

3. **Hierarchy Matters**: Use inheritance for handler specificity
   ```python
   # In exception handlers (next section), you can catch:
   except TokenBudgetExceededError:      # Specific: only token budget
       ...
   except BudgetError:                   # General: all budget errors
       ...
   except GenAIException:                # Very general: all GenAI errors
       ...
   ```

---

## Global Exception Handlers

### What Are Exception Handlers?

Exception handlers intercept exceptions raised in your endpoints and convert them to HTTP responses. Without handlers, unhandled exceptions crash your request and return a 500 error.

### Basic Handler

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# Register handler for TokenBudgetExceededError
@app.exception_handler(TokenBudgetExceededError)
async def handle_token_budget_exceeded(request: Request, exc: TokenBudgetExceededError):
    """
    When TokenBudgetExceededError is raised anywhere,
    this handler converts it to a 429 JSON response
    """
    return JSONResponse(
        status_code=429,  # Too Many Requests (also means "quota exceeded")
        content={
            "error": "token_budget_exceeded",
            "message": exc.message,
            "tokens_used": exc.tokens_used,
            "tokens_limit": exc.tokens_limit,
            "tokens_needed": exc.tokens_needed
        },
        headers={
            "Retry-After": "3600"  # Retry in an hour when budget resets
        }
    )
```

### Handler Flow

```
Endpoint raises TokenBudgetExceededError
                ↓
FastAPI catches it
                ↓
Looks for @app.exception_handler(TokenBudgetExceededError)
                ↓
Finds handler_token_budget_exceeded()
                ↓
Calls handler with (request, exc)
                ↓
Handler returns JSONResponse(status_code=429, content={...})
                ↓
Client receives HTTP 429 + JSON response
```

### Multiple Handlers for Different Exception Types

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# Handler for TokenBudgetExceededError
@app.exception_handler(TokenBudgetExceededError)
async def handle_token_budget(request: Request, exc: TokenBudgetExceededError):
    return JSONResponse(
        status_code=429,
        content={
            "error": "token_budget_exceeded",
            "message": exc.message,
            "upgrade_url": "https://app.com/upgrade"
        },
        headers={"Retry-After": "3600"}
    )


# Handler for RateLimitExceededError
@app.exception_handler(RateLimitExceededError)
async def handle_rate_limit(request: Request, exc: RateLimitExceededError):
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": exc.message,
            "reset_in": exc.reset_in
        },
        headers={"Retry-After": str(exc.reset_in)}
    )


# Handler for ProviderError (from external LLM API)
@app.exception_handler(ProviderError)
async def handle_provider_error(request: Request, exc: ProviderError):
    return JSONResponse(
        status_code=exc.status_code,  # Use provider's status code
        content={
            "error": f"{exc.provider.lower()}_error",
            "message": f"LLM provider ({exc.provider}) returned an error. Please try again.",
            # Never expose internal details to client
        },
        headers={
            "Retry-After": str(exc.retry_after) if exc.retry_after else "60"
        }
    )


# Handler for ModelNotFoundError
@app.exception_handler(ModelNotFoundError)
async def handle_model_not_found(request: Request, exc: ModelNotFoundError):
    return JSONResponse(
        status_code=404,
        content={
            "error": "model_not_found",
            "requested_model": exc.model_name,
            "available_models": exc.available_models
        }
    )
```

### Handler Specificity & Precedence

```
When exception is raised:
1. FastAPI looks for handler matching exact exception class
2. If not found, looks for handler matching parent class
3. If not found, looks for handler matching grandparent class
4. ... repeats up the inheritance chain
5. If no handler found, raises 500 Internal Server Error
```

Example with inheritance:

```python
# Exception hierarchy:
# GenAIException (base)
#   └── BudgetError
#       ├── TokenBudgetExceededError
#       └── RateLimitExceededError

# Handler for specific exception
@app.exception_handler(TokenBudgetExceededError)
async def handle_token_budget(request: Request, exc: TokenBudgetExceededError):
    return JSONResponse(status_code=429, content={"error": "token_budget_exceeded"})

# Handler for parent class (catches all BudgetErrors)
@app.exception_handler(BudgetError)
async def handle_budget_error(request: Request, exc: BudgetError):
    return JSONResponse(status_code=429, content={"error": "budget_exceeded"})

# Handler for grandparent class (catches ALL GenAI errors)
@app.exception_handler(GenAIException)
async def handle_genai_error(request: Request, exc: GenAIException):
    return JSONResponse(status_code=400, content={"error": "genai_error"})


# When TokenBudgetExceededError is raised:
# 1. Tries TokenBudgetExceededError handler (FOUND - uses this one)
# 2. Never reaches BudgetError handler
# 3. Never reaches GenAIException handler

# When RateLimitExceededError is raised:
# 1. Tries RateLimitExceededError handler (not found)
# 2. Tries BudgetError handler (FOUND - uses this one)
# 3. Never reaches GenAIException handler
```

### ✅ Correct vs ❌ Incorrect Handler Examples

#### Example 1: Adding Context to Response

```python
# ❌ WRONG: Generic error message
@app.exception_handler(ProviderError)
async def handle_provider(request: Request, exc: ProviderError):
    return JSONResponse(
        status_code=500,
        content={"error": "Something went wrong"}  # Useless to client!
    )

# ✅ CORRECT: Helpful, actionable error message
@app.exception_handler(ProviderError)
async def handle_provider(request: Request, exc: ProviderError):
    return JSONResponse(
        status_code=503,
        content={
            "error": f"{exc.provider.lower()}_unavailable",
            "message": f"{exc.provider} API is currently unavailable. We're working on it.",
            "status_page": f"https://{exc.provider.lower()}.com/status"
        },
        headers={"Retry-After": "60"}
    )
```

#### Example 2: Exposing vs Hiding Details

```python
# ❌ WRONG: Exposes internal database errors to client
@app.exception_handler(DatabaseError)
async def handle_db_error(request: Request, exc: DatabaseError):
    return JSONResponse(
        status_code=500,
        content={
            "error": "database_connection_failed",
            "details": str(exc),  # Exposes connection string! Security risk!
            "table": exc.table_name
        }
    )

# ✅ CORRECT: Generic message, detailed logging handled separately
@app.exception_handler(DatabaseError)
async def handle_db_error(request: Request, exc: DatabaseError):
    # Log full details for debugging
    logger.error(f"Database error: {exc}", exc_info=True, extra={
        "request_id": request.headers.get("X-Request-ID"),
        "endpoint": request.url.path
    })

    # Return generic message to client
    return JSONResponse(
        status_code=503,
        content={
            "error": "service_temporarily_unavailable",
            "message": "We're experiencing technical difficulties. Please try again soon."
        }
    )
```

#### Example 3: Catch-All Handler

```python
# ❌ WRONG: Let all unexpected errors become 500s
@app.exception_handler(Exception)
async def handle_all_errors(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

# ✅ CORRECT: Catch-all for unexpected errors with logging
@app.exception_handler(Exception)
async def handle_all_errors(request: Request, exc: Exception):
    request_id = request.headers.get("X-Request-ID", "unknown")

    # Log everything we can for debugging
    logger.error(
        f"Unhandled exception: {type(exc).__name__}",
        exc_info=True,
        extra={
            "request_id": request_id,
            "endpoint": request.url.path,
            "method": request.method,
            "status_code": 500
        }
    )

    # Return generic error with request ID for support
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred. Please contact support.",
            "request_id": request_id  # User can reference this in support email
        }
    )
```

### Exception Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Request arrives at @app.post("/chat")                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                ┌──────────▼──────────┐
                │ Endpoint executes   │
                └──────────┬──────────┘
                           │
              ┌────────────▼────────────┐
              │ Exception raised?       │
              └──────────┬──────────────┘
                         │
         ┌───────────────┴───────────────┐
         │ No                    │ Yes   │
         │                       │       │
         │           ┌───────────▼────────────────┐
         │           │ Look for exception handler │
         │           └───────────┬────────────────┘
         │                       │
         │        ┌──────────────┴──────────────┐
         │        │ Handler found?               │
         │        └──────┬────────────┬──────────┘
         │               │            │
         │        No     │            │ Yes
         │        ┌──────▼─┐  ┌───────▼────────────┐
         │        │Return  │  │ Call handler       │
         │        │500     │  │ with (req, exc)    │
         │        │error   │  └──────┬─────────────┘
         │        └────────┘         │
         │                 ┌─────────▼───────────┐
         │                 │Handler returns      │
         │                 │JSONResponse         │
         │                 └─────────┬───────────┘
         │                           │
         └──────────────┬────────────┘
                        │
              ┌─────────▼──────────┐
              │ Response sent to   │
              │ client             │
              └────────────────────┘
```

---

## Error Logging Best Practices

### Core Principle: Separate Concerns

```python
# ❌ WRONG: Same message for user and logs
@app.exception_handler(TokenBudgetExceededError)
async def handle_error(request: Request, exc: TokenBudgetExceededError):
    message = f"User {exc.user_id} exceeded budget: used {exc.tokens_used}/{exc.tokens_limit}"

    logger.error(message)  # Logs it

    return JSONResponse(
        status_code=429,
        content={"error": message}  # Returns to client - TOO DETAILED!
    )

# ✅ CORRECT: Different messages for user vs logs
@app.exception_handler(TokenBudgetExceededError)
async def handle_error(request: Request, exc: TokenBudgetExceededError):
    # Internal message for logging (detailed, for debugging)
    logger.error(
        "Token budget exceeded",
        extra={
            "user_id": exc.user_id,
            "tokens_used": exc.tokens_used,
            "tokens_limit": exc.tokens_limit,
            "request_id": request.headers.get("X-Request-ID")
        }
    )

    # External message for client (helpful, not revealing internals)
    return JSONResponse(
        status_code=429,
        content={"error": "Token budget exceeded. Upgrade to continue."}
    )
```

### What to Log vs What to Return

```python
# Log these (internal details):
- Full stack trace
- Variable values
- Database query
- Internal error codes
- User IDs and sensitive data

# Return to client:
- Friendly error message
- Action user can take
- Generic error code
- Never internal details
```

### Structured Logging with Context

```python
import logging
import uuid
from fastapi import Request

logger = logging.getLogger(__name__)

@app.exception_handler(TokenBudgetExceededError)
async def handle_token_budget(request: Request, exc: TokenBudgetExceededError):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    # Structured logging: key-value pairs for easy searching
    logger.error(
        "user_token_budget_exceeded",
        extra={
            "request_id": request_id,
            "user_id": exc.user_id,
            "tokens_used": exc.tokens_used,
            "tokens_limit": exc.tokens_limit,
            "tokens_needed": exc.tokens_needed,
            "endpoint": request.url.path,
            "method": request.method,
            "client_ip": request.client.host
        }
    )

    return JSONResponse(
        status_code=429,
        content={"error": "token_budget_exceeded"},
        headers={"Retry-After": "3600"}
    )
```

### Adding Request ID Middleware

Request IDs let you trace a single user request through all your logs:

```python
import logging
import uuid
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Add a unique request ID to each request.
    Use it to track requests through logs.
    """
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Store in request state (accessible in endpoints)
        request.state.request_id = request_id

        # Log the incoming request
        logger.info(
            "incoming_request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query": str(request.query_params),
                "client_ip": request.client.host
            }
        )

        # Call the endpoint
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        # Log the response
        logger.info(
            "response_sent",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "path": request.url.path
            }
        )

        return response


# Register middleware
app.add_middleware(RequestIDMiddleware)
```

### Complete Logging in Exception Handler

```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

@app.exception_handler(ProviderError)
async def handle_provider_error(request: Request, exc: ProviderError):
    request_id = request.state.request_id

    # Log with full context
    logger.error(
        f"{exc.provider.lower()}_api_error",
        extra={
            "request_id": request_id,
            "provider": exc.provider,
            "status_code": exc.status_code,
            "error_message": exc.message,
            "endpoint": request.url.path,
            "user_id": request.state.user_id,  # From auth middleware
            "retry_after": exc.retry_after,
            "timestamp": datetime.now().isoformat()
        },
        exc_info=True  # Include full stack trace in logs
    )

    # Return generic error to client (don't expose provider details)
    return JSONResponse(
        status_code=503,
        content={
            "error": "service_unavailable",
            "message": "The LLM service is temporarily unavailable",
            "request_id": request_id  # User can reference this for support
        },
        headers={"Retry-After": "60", "X-Request-ID": request_id}
    )
```

### Never Expose Stack Traces to Users

```python
# ❌ WRONG: Stack trace visible to client (security issue)
@app.exception_handler(Exception)
async def handle_unknown(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "traceback": traceback.format_exc()  # SECURITY HOLE!
        }
    )

# ✅ CORRECT: Stack trace only in logs
@app.exception_handler(Exception)
async def handle_unknown(request: Request, exc: Exception):
    request_id = request.state.request_id

    logger.error(
        "unhandled_exception",
        extra={"request_id": request_id, "endpoint": request.url.path},
        exc_info=True  # Full traceback in logs
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An error occurred. Please reference this ID: " + request_id
            # No traceback in response!
        }
    )
```

---

## Complete Working Example

### Full App with Error Handling

This example brings together:
- Custom exceptions
- Global handlers
- Request ID middleware
- Structured logging

```python
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum

from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# ===== SETUP =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GenAI API with Error Handling")

# ===== CUSTOM EXCEPTIONS =====

class GenAIException(Exception):
    """Base exception for GenAI errors"""
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


class ProviderError(GenAIException):
    """Error from LLM provider"""
    def __init__(self, provider: str, status_code: int, message: str, retry_after: int = 60):
        self.provider = provider
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(f"{provider} error: {message}")


class ModelNotFoundError(GenAIException):
    """Requested model doesn't exist"""
    def __init__(self, model_name: str, available: list[str]):
        self.model_name = model_name
        self.available_models = available
        super().__init__(f"Model {model_name} not found")


class RateLimitExceededError(GenAIException):
    """User exceeded request rate limit"""
    def __init__(self, limit: int, reset_in: int):
        self.limit = limit
        self.reset_in = reset_in
        super().__init__(f"Rate limit exceeded: {limit} requests per minute")


# ===== MIDDLEWARE =====

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request"""
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        request.state.user_id = "user_123"  # From auth in real app

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(RequestIDMiddleware)


# ===== EXCEPTION HANDLERS =====

@app.exception_handler(TokenBudgetExceededError)
async def handle_token_budget(request: Request, exc: TokenBudgetExceededError):
    logger.error(
        "token_budget_exceeded",
        extra={
            "request_id": request.state.request_id,
            "user_id": exc.user_id,
            "tokens_used": exc.tokens_used,
            "tokens_limit": exc.tokens_limit
        }
    )

    return JSONResponse(
        status_code=429,
        content={
            "error": "token_budget_exceeded",
            "message": "You've used your monthly token budget",
            "upgrade_url": "https://app.com/upgrade"
        },
        headers={"Retry-After": "3600"}
    )


@app.exception_handler(ProviderError)
async def handle_provider(request: Request, exc: ProviderError):
    logger.error(
        f"{exc.provider.lower()}_error",
        extra={
            "request_id": request.state.request_id,
            "provider": exc.provider,
            "status_code": exc.status_code,
            "message": exc.message
        }
    )

    return JSONResponse(
        status_code=503,
        content={
            "error": "provider_unavailable",
            "message": f"{exc.provider} is temporarily unavailable"
        },
        headers={"Retry-After": str(exc.retry_after)}
    )


@app.exception_handler(ModelNotFoundError)
async def handle_model_not_found(request: Request, exc: ModelNotFoundError):
    logger.warning(
        "model_not_found",
        extra={
            "request_id": request.state.request_id,
            "requested_model": exc.model_name,
            "available_models": exc.available_models
        }
    )

    return JSONResponse(
        status_code=404,
        content={
            "error": "model_not_found",
            "message": f"Model '{exc.model_name}' is not available",
            "available_models": exc.available_models
        }
    )


@app.exception_handler(RateLimitExceededError)
async def handle_rate_limit(request: Request, exc: RateLimitExceededError):
    logger.warning(
        "rate_limit_exceeded",
        extra={
            "request_id": request.state.request_id,
            "user_id": request.state.user_id,
            "limit": exc.limit
        }
    )

    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": f"Rate limit: {exc.limit} requests per minute",
            "reset_in": exc.reset_in
        },
        headers={"Retry-After": str(exc.reset_in)}
    )


@app.exception_handler(Exception)
async def handle_unexpected(request: Request, exc: Exception):
    request_id = request.state.request_id
    logger.error(
        "unhandled_exception",
        extra={
            "request_id": request_id,
            "endpoint": request.url.path,
            "exception_type": type(exc).__name__
        },
        exc_info=True
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred",
            "request_id": request_id
        }
    )


# ===== DATA MODELS =====

class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    temperature: float = 0.7


class ChatResponse(BaseModel):
    response: str
    tokens_used: int
    model: str


# ===== DUMMY DATA =====

SUPPORTED_MODELS = ["claude-3", "gpt-4", "llama-2"]
USER_TOKEN_BUDGET = 10000
USER_TOKENS_USED = 9500
RATE_LIMIT_PER_MINUTE = 10

request_count = 0


# ===== ENDPOINTS =====

@app.get("/health")
async def health_check():
    """Simple health check"""
    return {"status": "healthy"}


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    GenAI chat endpoint with error handling.

    Demonstrates:
    - Model validation
    - Budget checking
    - Rate limiting
    - Provider error handling
    """
    global request_count
    request_count += 1

    # 1. Validate model exists
    if request.model not in SUPPORTED_MODELS:
        raise ModelNotFoundError(request.model, SUPPORTED_MODELS)

    # 2. Check rate limit
    if request_count > RATE_LIMIT_PER_MINUTE:
        raise RateLimitExceededError(RATE_LIMIT_PER_MINUTE, 60)

    # 3. Estimate tokens and check budget
    estimated_tokens = len(str(request.messages)) * 4  # Rough estimate
    remaining = USER_TOKEN_BUDGET - USER_TOKENS_USED

    if estimated_tokens > remaining:
        raise TokenBudgetExceededError(
            user_id="user_123",
            used=USER_TOKENS_USED,
            limit=USER_TOKEN_BUDGET,
            needed=estimated_tokens
        )

    # 4. Simulate provider error (10% of the time)
    import random
    if random.random() < 0.1:
        raise ProviderError(
            provider="Anthropic",
            status_code=429,
            message="Rate limited",
            retry_after=60
        )

    # Success!
    return ChatResponse(
        response="This is a simulated response from Claude",
        tokens_used=estimated_tokens,
        model=request.model
    )


@app.get("/models")
async def list_models():
    """List available models"""
    return {"models": SUPPORTED_MODELS}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Running and Testing

```bash
# Install dependencies
pip install fastapi uvicorn pydantic

# Run the server
python app.py

# In another terminal, test endpoints:

# Successful request
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3", "messages": [{"role": "user", "content": "Hello"}]}'

# Model not found error
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "unknown-model", "messages": []}'
# → HTTP 404: {"error": "model_not_found", "available_models": [...]}

# Token budget exceeded (after many requests)
# → HTTP 429: {"error": "token_budget_exceeded", "upgrade_url": "..."}

# Provider error (random 10%)
# → HTTP 503: {"error": "provider_unavailable", ...}

# View logs with request IDs
tail -f logs.log
```

---

## Common Patterns for GenAI Applications

### Pattern 1: Provider Failover

Try multiple providers, return fastest response:

```python
import asyncio
from typing import Optional

async def get_response_with_failover(
    messages: list,
    primary_provider: str = "anthropic",
    fallback_provider: str = "openai"
):
    """
    Try primary provider first.
    If it fails or times out, try fallback.
    Return whichever responds first.
    """

    async def call_provider(provider: str):
        try:
            response = await call_llm_api(provider, messages, timeout=5)
            return response
        except asyncio.TimeoutError:
            logger.warning(f"{provider} timed out")
            raise
        except ProviderError as e:
            logger.warning(f"{provider} error: {e}")
            raise

    try:
        # Try primary provider
        return await call_provider(primary_provider)
    except (asyncio.TimeoutError, ProviderError):
        logger.info(f"Falling back to {fallback_provider}")
        try:
            # Try fallback
            return await call_provider(fallback_provider)
        except (asyncio.TimeoutError, ProviderError) as e:
            # Both failed
            raise ProviderError(
                provider="all_providers",
                status_code=503,
                message="All providers failed"
            )
```

### Pattern 2: Token Budget Enforcement

Check budget before expensive operation:

```python
from datetime import datetime, timedelta

async def validate_request_within_budget(user_id: str, estimated_tokens: int):
    """
    Check if user has token budget for this request.
    Called before expensive LLM API call.
    """
    user = await db.get_user(user_id)

    # Calculate reset time (monthly)
    today = datetime.now()
    month_start = today.replace(day=1, hour=0, minute=0, second=0)
    month_end = (month_start + timedelta(days=32)).replace(day=1)

    # Get tokens used this month
    tokens_used_this_month = await db.count_tokens_since(user_id, month_start)

    # Check budget
    tokens_remaining = user.monthly_budget - tokens_used_this_month

    if estimated_tokens > tokens_remaining:
        raise TokenBudgetExceededError(
            user_id=user_id,
            used=tokens_used_this_month,
            limit=user.monthly_budget,
            needed=estimated_tokens
        )
```

### Pattern 3: Timeout Handling with Graceful Degradation

```python
import asyncio

@app.post("/chat")
async def chat_with_timeout(request: ChatRequest):
    """
    Call LLM with timeout.
    If it times out, return cached response if available.
    """

    try:
        # Try to get fresh response (5 second timeout)
        response = await asyncio.wait_for(
            call_llm_api(request.model, request.messages),
            timeout=5.0
        )
        return response

    except asyncio.TimeoutError:
        logger.warning(f"LLM call timed out for user {request.user_id}")

        # Try to return cached response
        cached = await cache.get(f"chat:{hash(request)}")
        if cached:
            logger.info("Returning cached response due to timeout")
            return {
                "response": cached["response"],
                "cached": True,
                "message": "LLM is slow, showing cached response"
            }

        # No cache available
        raise ProviderError(
            provider="LLM",
            status_code=504,
            message="Request timed out"
        )
```

### Pattern 4: Request Validation + Early Error

```python
async def validate_request(request: ChatRequest) -> None:
    """
    Validate request completely before calling expensive API.
    This catches errors early and saves API quota.
    """

    # Check model exists
    if request.model not in SUPPORTED_MODELS:
        raise ModelNotFoundError(request.model, SUPPORTED_MODELS)

    # Check messages are valid
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    for msg in request.messages:
        if "role" not in msg or "content" not in msg:
            raise HTTPException(status_code=400, detail="Invalid message format")

    # Check parameters are valid
    if not 0 <= request.temperature <= 2:
        raise HTTPException(status_code=400, detail="temperature must be 0-2")

    # Check user has budget
    tokens_needed = estimate_tokens(request.messages)
    await validate_request_within_budget(request.user_id, tokens_needed)


@app.post("/chat")
async def chat(request: ChatRequest, user: User = Depends(get_user)):
    # Validate BEFORE calling API
    await validate_request(request)

    # Now safe to call expensive API
    response = await call_llm_api(request)
    return response
```

---

## Key Insights for Your Learning

```
★ Insight ─────────────────────────────────────

1. Error handling structure mirroring: Your exception hierarchy mirrors your problem domain.
   GenAIException → BudgetError → TokenBudgetExceededError means "this is a budget
   problem" is clear from the structure alone. Use inheritance for clarity.

2. Separation of concerns: What you log ≠ what you show users. Logs are for debugging
   (detailed, internal), responses are for users (helpful, actionable). Mixing them
   is a security risk and bad UX.

3. Handlers catch exceptions early: Without handlers, one failed LLM call crashes your
   entire request. Handlers transform exceptions into proper HTTP responses, keeping
   your app running even when external APIs fail.

4. Request IDs enable debugging: Add unique IDs to every request. When a user says
   "my request failed," you can grep logs for that ID and see exactly what happened.
   This is invaluable in production.

5. Progressive validation: Validate completely BEFORE calling expensive APIs. Check
   budget, rate limits, model existence first. This saves money and improves user
   experience by failing fast with clear errors.

─────────────────────────────────────────────────────────────
```

---

## Quick Reference

### HTTP Status Codes for GenAI

| Code | Meaning | GenAI Example | Headers |
|------|---------|---------------|---------|
| 200 | OK | Chat response succeeded | - |
| 400 | Bad Request | Invalid message format | - |
| 404 | Not Found | Model doesn't exist | - |
| 429 | Too Many Requests | Rate limit or budget exceeded | Retry-After |
| 500 | Internal Error | Unexpected server error | - |
| 503 | Service Unavailable | LLM provider down | Retry-After |

### Exception Decision Tree

```
Is this an expected error (missing resource, rate limit)?
├─ Yes, simple (404, etc) → Use HTTPException
├─ Yes, complex (store context) → Create custom exception
└─ No, unexpected → Let it propagate, catch with catch-all handler

Is this GenAI-specific (token budget, provider error)?
└─ Yes → Create custom exception in GenAI exception hierarchy

Do I need to add custom headers (Retry-After, etc)?
└─ Yes → Use custom exception + handler to add headers

Do I need to log details not shown to user?
└─ Yes → Use custom exception + handler with logging
```

### Exception Handler Template

```python
@app.exception_handler(MyCustomException)
async def handle_my_error(request: Request, exc: MyCustomException):
    # Log with full context
    logger.error(
        "event_name",
        extra={
            "request_id": request.state.request_id,
            "user_id": request.state.user_id,
            # ... other context
        }
    )

    # Return user-friendly response
    return JSONResponse(
        status_code=appropriate_status_code,
        content={
            "error": "error_code",
            "message": "User-facing message"
        },
        headers={"Retry-After": "60"}  # if needed
    )
```

---

## Next Steps

**After completing this topic:**

1. **Practice the hands-on exercises** in `03_Error_Handling_Practice.md`
   - Build each exception type
   - Create handlers
   - Test error responses

2. **Integrate with your chat API**
   - Add token budget checking
   - Add provider error handling
   - Add structured logging

3. **Move to Topic 4: Middleware**
   - Learn request/response lifecycle
   - Add CORS for web frontends
   - Build timing middleware

**Key Takeaway**: Error handling is not boring infrastructure—it's what separates products that work from ones that crash. Every exception you handle gracefully is a user who doesn't leave your app.

---

*Ready to build robust error handling? Let's go.*
