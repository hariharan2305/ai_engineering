"""
In this example, we define custom exception classes for handling specific error scenarios related to GenAI usage.

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

Pattern:
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

Key observations:
- Handlers centralize error formatting: All TokenBudgetExceededError responses are formatted the same way
- Context matters: Handlers access exc.tokens_used, exc.tokens_limit, etc. for rich responses
- Headers are important: Retry-After tells clients when to retry
- Different exceptions → different responses: Model not found is 404, budget exceeded is 429

Key Takeaway:
- Global exception handlers transform scattered error handling into a centralized system. One handler per exception type. Complex business logic doesn't need to know about HTTP—it just raises the right exception, and handlers take care of the rest.

"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

# Create an instance of the FastAPI application
app = FastAPI()

# ===== Custom Exception Class =====
class GenAIException(Exception):
    """Custom exception class for handling GenAI-related errors."""
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

class RateLimitExceededError(GenAIException):
    """
    Raised when a user exceeds their rate limit.

    Attributes:
        user_id: The user who exceeded rate limit
        requests_made: Requests made in the current time window
        requests_limit: Allowed requests in the time window
    """

    def __init__(
        self,
        user_id: str,
        requests_made: int,
        requests_limit: int
    ):
        self.user_id = user_id
        self.requests_made = requests_made
        self.requests_limit = requests_limit

        message = (
            f"Rate limit exceeded. "
            f"Made {requests_made}/{requests_limit} requests in the current time window."
        )

        super().__init__(message)

class ProvderError(GenAIException):
    """
    Raised when an error occurs with the GenAI provider.

    Attributes:
        provider_name: Name of the GenAI provider
        error_code: Error code returned by the provider
        error_message: Detailed error message from the provider
    """

    def __init__(
        self,
        provider_name: str,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        self.provider_name = provider_name
        self.error_code = error_code
        self.error_message = error_message

        message = f"Error with provider {provider_name}."
        if error_code:
            message += f" Error code: {error_code}."
        if error_message:
            message += f" Message: {error_message}"

        super().__init__(message)

class ModelNotFoundError(GenAIException):
    """
    Raised when a requested GenAI model is not found.

    Attributes:
        model_name: Name of the model that was not found
        available_models: List of available models
    """

    def __init__(self, model_name: str, available_models: Optional[list[str]] = None):
        self.model_name = model_name
        self.available_models = available_models
        message = f"Model '{model_name}' not found."
        if available_models:
            message += f" Available models: {', '.join(available_models)}"
        super().__init__(message)


# ====== Global Exception Handlers =====
@app.exception_handler(TokenBudgetExceededError)
async def handle_token_budget(request: Request, exc: TokenBudgetExceededError):
    """Convert TokenBudgetExceededError to a JSON response with status code 429."""
    return JSONResponse(
        status_code=429,
        content={"error": "token_budget_exceeded",
                 "message": exc.message,
                 "user_id": exc.user_id,
                 "tokens_used": exc.tokens_used,
                 "tokens_limit": exc.tokens_limit,
                 "tokens_needed": exc.tokens_needed,
                 "upgrade_url": "https://example.com/upgrade"
                },
        headers={"X-RateLimit-Retry-After": "3600"}  # Suggest retry after 1 hour
    )

@app.exception_handler(RateLimitExceededError)
async def handle_rate_limit(request: Request, exc: RateLimitExceededError):
    """Convert RateLimitExceededError to a JSON response with status code 429."""
    return JSONResponse(
        status_code=429,
        content={"error": "rate_limit_exceeded",
                 "message": exc.message,
                 "user_id": exc.user_id,
                 "requests_made": exc.requests_made,
                 "requests_limit": exc.requests_limit,
                 "retry_after_seconds": 60  # Suggest retry after 60 seconds
                },
        headers={"X-RateLimit-Retry-After": "60"}
    )

@app.exception_handler(ProvderError)
async def handle_provider_error(request: Request, exc: ProvderError):
    """Convert ProvderError to a JSON response with status code 503."""
    return JSONResponse(
        status_code=503,
        content={"error": "provider_error",
                 "message": exc.message,
                 "provider_name": exc.provider_name,
                 "error_code": exc.error_code,
                 "error_message": exc.error_message
                },
        headers={"X-RateLimit-Retry-After": "3600"}  # Suggest retry after 1 hour
    )

@app.exception_handler(ModelNotFoundError)
async def handle_model_not_found(request: Request, exc: ModelNotFoundError):
    """Convert ModelNotFoundError to a JSON response with status code 404."""
    return JSONResponse(
        status_code=404,
        content={"error": "model_not_found",
                 "message": exc.message,
                 "model_name": exc.model_name,
                 "available_models": exc.available_models
                }
    )

# ====== Data Models =====
class ChatRequest(BaseModel):
    user_id: str
    model_name: str
    message: str

SUPPORTED_MODELS = ["claude-4", "gpt-4", "claude-opus-4-6"]
request_count = 0  # Simulate request count for rate limiting

# ====== Endpoints =====
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/models")
async def list_models():
    # Simulate fetching available models from a provider
    return {"available_models": SUPPORTED_MODELS}

@app.post("/chat")
async def chat(request: ChatRequest):
    global request_count    

    # 1. Check if model is supported
    if request.model_name not in SUPPORTED_MODELS:
        raise ModelNotFoundError(
            model_name=request.model_name,
            available_models=SUPPORTED_MODELS
        )
    
    # 2. Simulate rate limiting (e.g., max 5 requests per minute)
    if request_count > 5:
        request_count += 1
        raise RateLimitExceededError(
            user_id=request.user_id,
            requests_made=request_count,
            requests_limit=5
        )
    
    # 3. Simulate token estimation (e.g., 1 token per 4 characters)
    tokens_needed = len(request.message) // 4
    if tokens_needed > 10:
        raise TokenBudgetExceededError(
            user_id=request.user_id,
            tokens_used=8,  # Simulate no tokens used for simplicity
            tokens_limit=10,
            tokens_needed=tokens_needed
        )

    # 4. Simulate Provider Error (e.g., 10% chance of provider failure)
    if request_count %3 == 0:  # Simulate provider error every 3rd request
        raise ProvderError(
            provider_name="ExampleGenAIProvider",
            error_code="provider_timeout",
            error_message="The provider did not respond in time."
        )    

    # Simulate successful chat response
    return {"response": f"Simulated response from {request.model_name} for message: {request.message}"}