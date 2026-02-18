"""
This example demonstrates a complete middleware stack in FastAPI, combining CORS handling, request ID generation, timing, and logging. 

This script demonstrates combining all middleware patterns:
- CORS for frontend access
- Request ID for debug tracing
- Timing for performance monitoring
- Logging for request/response audit

The KEY learning here is MIDDLEWARE ORDER:
- Last added = first to execute (outermost)
- Registration order determines the execution stack
- Wrong order = broken features (e.g., logging without request ID)

GenAI Context:
This middleware stack is what every production GenAI API needs.
CORS allows your chat UI to call the API, request IDs trace LLM calls,
timing catches slow completions, and logging provides the audit trail
for debugging.

Concepts covered:
- Multiple middleware working together
- Correct registration order (critical!)
- request.state sharing between middleware
- Complete request lifecycle with all middleware
- Production-ready patterns

Understanding the Full Middleware Stack:
curl request
    │
    ▼
┌─── CORS Middleware (1st to execute — added last) ─────────────┐
│ Check origin, prepare CORS headers                            │
│                                                               │
│ ┌─── Request ID Middleware (2nd) ──────────────────────────┐  │
│ │ Generate UUID, store in request.state.request_id         │  │
│ │                                                          │  │
│ │ ┌─── Timing Middleware (3rd) ──────────────────────────┐ │  │
│ │ │ Record start_time                                    │ │  │
│ │ │                                                      │ │  │
│ │ │ ┌─── Logging Middleware (4th — innermost) ─────────┐ │ │  │
│ │ │ │ Log: → REQUEST | id=abc-123 POST /chat           │ │ │  │
│ │ │ │                                                  │ │ │  │
│ │ │ │ ┌────────────────────────────────────────────┐   │ │ │  │
│ │ │ │ │ ENDPOINT: /chat                            │   │ │ │  │
│ │ │ │ │  - Validate model                          │   │ │ │  │
│ │ │ │ │  - Simulate LLM call (2s)                  │   │ │ │  │
│ │ │ │ │  - Return response                         │   │ │ │  │
│ │ │ │ └────────────────────────────────────────────┘   │ │ │  │
│ │ │ │                                                  │ │ │  │
│ │ │ │ Log: ← RESPONSE | id=abc-123 status=200 2.0s ✓  │ │ │  │
│ │ │ └──────────────────────────────────────────────────┘ │ │  │
│ │ │                                                      │ │  │
│ │ │ Calculate duration, add X-Process-Time: 2.004s       │ │  │
│ │ └──────────────────────────────────────────────────────┘ │  │
│ │                                                          │  │
│ │ Add X-Request-ID: abc-123 header                         │  │
│ └──────────────────────────────────────────────────────────┘  │
│                                                               │
│ Add Access-Control-Allow-Origin header                        │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
Response with all headers → client

Key takeaways:
- A complete middleware stack gives you production-ready infrastructure for free on every request. 
  The critical lesson is ordering: middleware must be registered in the correct sequence so each layer has access to what it needs. 
  Request IDs must exist before logging can use them. Timing must wrap everything to get accurate durations. 
  CORS must be outermost to handle preflight requests before anything else. 
  This stack (CORS + Request ID + Timing + Logging) is the foundation of every production GenAI API.

"""

import logging
import time
import uuid
import asyncio

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

# ===== LOGGING SETUP =====

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s"
)
logger = logging.getLogger("genai_api")

app = FastAPI(title="GenAI API — Complete Middleware Stack")


# ===== CUSTOM MIDDLEWARE CLASSES =====

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generate or accept request IDs for tracing"""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Measure request duration and flag slow requests"""

    def __init__(self, app, slow_threshold: float = 5.0):
        super().__init__(app)
        self.slow_threshold = slow_threshold

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        # Store for other middleware to use
        request.state.process_time = duration
        response.headers["X-Process-Time"] = f"{duration:.3f}s"

        # Flag slow requests
        if duration > self.slow_threshold:
            request_id = getattr(request.state, "request_id", "unknown")
            logger.warning(
                "SLOW REQUEST | id=%s path=%s duration=%.3fs (threshold: %.1fs)",
                request_id, request.url.path, duration, self.slow_threshold
            )

        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log every request and response with context from other middleware"""

    async def dispatch(self, request: Request, call_next):
        # Read data set by RequestIDMiddleware (runs before this)
        request_id = getattr(request.state, "request_id", "unknown")

        # Log incoming request
        logger.info(
            "→ REQUEST  | id=%s %s %s client=%s",
            request_id,
            request.method,
            request.url.path,
            request.client.host if request.client else "unknown"
        )

        # Execute endpoint
        response = await call_next(request)

        # Read data set by TimingMiddleware (ran around this)
        duration = getattr(request.state, "process_time", 0)

        # Choose log level based on status code
        if response.status_code >= 500:
            log_func = logger.error
            emoji = "✗"
        elif response.status_code >= 400:
            log_func = logger.warning
            emoji = "⚠"
        else:
            log_func = logger.info
            emoji = "✓"

        # Log outgoing response
        log_func(
            "← RESPONSE | id=%s %s %s status=%d duration=%.3fs %s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration,
            emoji
        )

        return response


# ===== REGISTER MIDDLEWARE (ORDER IS CRITICAL!) =====
#
# Remember: LAST added = FIRST to execute on request (outermost layer)
#
# Execution order for REQUESTS (top to bottom):
#   1. CORS         → handles preflight, adds CORS headers
#   2. Request ID   → generates ID for tracing
#   3. Timing       → starts timer
#   4. Logging      → logs incoming request (has request_id from step 2)
#   5. Endpoint     → your business logic
#
# Execution order for RESPONSES (bottom to top):
#   5. Endpoint     → returns response
#   4. Logging      → logs response (has timing from step 3)
#   3. Timing       → calculates duration, adds X-Process-Time
#   2. Request ID   → adds X-Request-ID header
#   1. CORS         → adds CORS headers

# 4. Logging (innermost — runs closest to endpoint)
app.add_middleware(LoggingMiddleware)

# 3. Timing (wraps logging + endpoint)
app.add_middleware(TimingMiddleware, slow_threshold=3.0)

# 2. Request ID (generates ID for all downstream middleware)
app.add_middleware(RequestIDMiddleware)

# 1. CORS (outermost — handles preflight before anything else)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-ID"],
    expose_headers=["X-Request-ID", "X-Process-Time", "Retry-After"],
)


# ===== DATA MODELS =====

class ChatRequest(BaseModel):
    model: str = "claude-3"
    messages: list[dict]
    temperature: float = 0.7


# ===== MOCK DATA =====

SUPPORTED_MODELS = {
    "claude-3": {"name": "Claude 3", "provider": "Anthropic"},
    "gpt-4": {"name": "GPT-4", "provider": "OpenAI"},
}


# ===== ENDPOINTS =====

@app.get("/health")
async def health():
    """Fast health check — all middleware still processes this"""
    return {"status": "healthy"}


@app.get("/models")
async def list_models():
    """List available models — fast endpoint"""
    return {"models": list(SUPPORTED_MODELS.values())}


@app.post("/chat")
async def chat(body: ChatRequest, request: Request):
    """
    Chat endpoint — simulates LLM call with 2-second delay.
    Demonstrates middleware context flowing into the endpoint.
    """
    if body.model not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{body.model}' not found. Available: {list(SUPPORTED_MODELS.keys())}"
        )

    # Simulate LLM API call
    await asyncio.sleep(2)

    logger.info(
        "LLM CALL   | id=%s model=%s provider=%s",
        request.state.request_id,
        body.model,
        SUPPORTED_MODELS[body.model]["provider"]
    )

    return {
        "response": f"Simulated response from {body.model}",
        "model": body.model,
        "tokens_used": 42,
        "request_id": request.state.request_id
    }


@app.post("/chat/slow")
async def chat_slow(request: Request):
    """
    Deliberately slow endpoint (~5s) — triggers timing warning.
    Demonstrates slow request detection.
    """
    await asyncio.sleep(5)

    return {
        "response": "This was a complex analysis...",
        "simulated_latency": "5s",
        "request_id": request.state.request_id
    }
