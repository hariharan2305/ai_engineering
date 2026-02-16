"""
In this example, we demonstrate advanced error handling in FastAPI with global exception handlers that have access to request context.

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

Pattern:
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

Key Takeaway:
- Request IDs are your debugging superpower. Every request gets a unique ID. When something goes wrong, you log that ID with full context. User gets the ID in the response. When they contact support ("My request failed"), you search logs for that ID and see exactly what went wrong.

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


# Create an instance of the FastAPI application
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

@app.get("/health")
def health(request: Request):
    """Health check"""
    return {
        "status": "healthy",
        "request_id": request.state.request_id
    }


@app.post("/chat")
def chat(chat_request: ChatRequest, request: Request):
    """Chat endpoint that demonstrates logging"""

    # Simulate different errors
    if len(chat_request.message) > 200:
        raise TokenBudgetExceededError(
            user_id="user_123",
            used=9500,
            limit=10000,
            needed=len(chat_request.message) // 4
        )

    if "provider_error" in chat_request.message.lower():
        raise ProviderError(
            provider="Anthropic",
            status_code=429,
            message="Rate limited"
        )

    if "crash" in chat_request.message.lower():
        raise Exception("Intentional error for testing")

    logger.info(
        "chat_request_processed",
        extra={
            "request_id": request.state.request_id,
            "user_id": request.state.user_id,
            "message_length": len(chat_request.message)
        }
    )

    return {
        "response": "Chat response",
        "request_id": request.state.request_id
    }

