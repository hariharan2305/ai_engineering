"""
In this example, we implement a custom middleware to assign a unique request ID to every incoming request.

This script demonstrates creating custom middleware with BaseHTTPMiddleware:
- Generating unique request IDs (UUID v4)
- Accepting client-provided IDs via X-Request-ID header
- Storing data in request.state for use in endpoints
- Adding custom headers to responses
- Using request IDs for log correlation

GenAI Context:
When a user's chat request fails in production, request IDs let you trace
that exact request through your logs — from incoming request, through
LLM provider call, to the error response. Without them, debugging is
searching through millions of log lines blind.

Concepts covered:
- BaseHTTPMiddleware pattern
- dispatch(self, request, call_next)
- request.state for passing data across the lifecycle
- Response header modification

Understanding the flow:
curl sends request
    │
    ▼
RequestIDMiddleware.dispatch()
    │
    ├── 1. Read X-Request-ID header (or generate UUID)
    ├── 2. Store in request.state.request_id
    ├── 3. Log: "request_start | id=abc-123"
    │
    ├── call_next(request) ──→ Endpoint executes
    │                            ├── Reads request.state.request_id
    │                            ├── Uses ID in its own logging
    │                            └── Returns response
    │
    ├── 4. Add X-Request-ID to response headers
    ├── 5. Log: "request_end | id=abc-123"
    │
    └── return response

key takeaways:
- Request ID middleware is the most valuable middleware for production GenAI apps. 
  It creates a correlation ID that ties together all logs for a single request — from the initial HTTP call, through your auth layer, through the LLM provider call, to the final response. 
  When things go wrong (and with LLM APIs, they will), this is how you debug.    
    
"""
import uuid
import logging

from fastapi import FastAPI, Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="GenAI API with Request ID Tracking")


# ===== CUSTOM MIDDLEWARE =====

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Assigns a unique ID to every request.

    - Generates UUID v4 if client doesn't provide one
    - Accepts client-provided X-Request-ID header
    - Stores ID in request.state for endpoints/handlers
    - Adds ID to response header X-Request-ID
    """

    async def dispatch(self, request: Request, call_next):
        # Accept client ID or generate new one
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Store in request.state (accessible everywhere downstream)
        request.state.request_id = request_id

        # Log incoming request with ID
        logger.info(
            "request_start | id=%s method=%s path=%s",
            request_id, request.method, request.url.path
        )

        # Call the endpoint (or next middleware)
        response = await call_next(request)

        # Add ID to response headers
        response.headers["X-Request-ID"] = request_id

        # Log response with same ID
        logger.info(
            "request_end | id=%s status=%d",
            request_id, response.status_code
        )

        return response


# Register middleware
app.add_middleware(RequestIDMiddleware)


# ===== DATA MODELS =====

class ChatRequest(BaseModel):
    model: str = "claude-3"
    messages: list[dict]


# ===== MOCK DATA =====

MODELS = ["claude-3", "gpt-4"]

conversations = {
    "conv_1": {"id": "conv_1", "title": "First chat", "messages": 5},
    "conv_2": {"id": "conv_2", "title": "Second chat", "messages": 12},
}


# ===== ENDPOINTS =====

@app.get("/health")
async def health(request: Request):
    """Health check — shows request ID in response"""
    return {
        "status": "healthy",
        "request_id": request.state.request_id
    }


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, request: Request):
    """
    Get conversation — uses request ID for logging.
    Returns 404 with request ID for debugging.
    """
    logger.info(
        "fetching_conversation | id=%s conv=%s",
        request.state.request_id, conversation_id
    )

    if conversation_id not in conversations:
        logger.warning(
            "conversation_not_found | id=%s conv=%s",
            request.state.request_id, conversation_id
        )
        raise HTTPException(
            status_code=404,
            detail={
                "error": "conversation_not_found",
                "conversation_id": conversation_id,
                "request_id": request.state.request_id,
                "message": f"Conversation '{conversation_id}' not found. Check your conversation ID."
            }
        )

    return conversations[conversation_id]


@app.post("/chat")
async def chat(body: ChatRequest, request: Request):
    """
    Chat endpoint — logs request ID with simulated LLM call.
    In production, the request ID would be passed to the LLM provider.
    """
    logger.info(
        "chat_request | id=%s model=%s msg_count=%d",
        request.state.request_id, body.model, len(body.messages)
    )

    if body.model not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{body.model}' not found. Available: {MODELS}"
        )

    # Simulate LLM call
    logger.info(
        "llm_call | id=%s provider=anthropic model=%s",
        request.state.request_id, body.model
    )

    return {
        "response": f"Simulated response from {body.model}",
        "model": body.model,
        "request_id": request.state.request_id  # Return for debugging
    }
