# Middleware Hands-On Practice: Build & Test

This guide takes you through **practical implementation** of middleware patterns with progressive complexity. By the end, you'll have a production-ready middleware stack for your GenAI API.

Each exercise builds on the previous one. Follow them in order.

---

## Exercise 1: Configure CORS for a React/Next.js Frontend (25 minutes)

### Goal

Set up CORS middleware so a React frontend at `http://localhost:3000` can call your FastAPI API. Understand what CORS is and why browsers enforce it.

### Steps

**1. Create `projects/fastapi_concepts_hands_on/08_1_cors_middleware.py`:**

```python
"""
Exercise 1: CORS Middleware for Web Frontends

This script demonstrates CORS (Cross-Origin Resource Sharing) configuration:
- Why browsers block cross-origin requests
- How CORSMiddleware solves this
- Configuring allowed origins, methods, and headers
- Testing CORS with curl (simulating a browser)

GenAI Context:
When you build a React/Next.js chat UI (http://localhost:3000)
that calls your FastAPI backend (http://localhost:8000), the browser
blocks the requests unless your API explicitly allows them via CORS.

Concepts covered:
- CORSMiddleware configuration
- allow_origins (specific domains, not wildcards in production)
- allow_methods (which HTTP methods are permitted)
- allow_headers (which request headers are allowed)
- expose_headers (which response headers the browser can read)
- Preflight OPTIONS requests
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="GenAI API with CORS")


# ===== CORS CONFIGURATION =====

# Environment-aware origins
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "development":
    # Allow common development frontend ports
    ALLOWED_ORIGINS = [
        "http://localhost:3000",    # React (Create React App / Next.js)
        "http://localhost:5173",    # Vite
        "http://127.0.0.1:3000",   # Alternative localhost
    ]
else:
    # Production: only your deployed frontend
    ALLOWED_ORIGINS = [
        "https://your-app.vercel.app",
        "https://your-custom-domain.com",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,  # Allow Authorization/Cookie headers
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type",      # JSON body
        "Authorization",     # JWT Bearer token
        "X-API-Key",         # API key authentication
        "X-Request-ID",      # Client-provided request ID
    ],
    expose_headers=[
        "X-Request-ID",      # Let frontend read request ID from response
        "X-Process-Time",    # Let frontend show response time
        "Retry-After",       # Let frontend handle rate limits
    ],
)


# ===== DATA MODELS =====

class ChatRequest(BaseModel):
    model: str = "claude-3"
    messages: list[dict]


class ChatResponse(BaseModel):
    response: str
    model: str


# ===== MOCK DATA =====

MODELS = {
    "claude-3": "Anthropic Claude 3",
    "gpt-4": "OpenAI GPT-4",
}


# ===== ENDPOINTS =====

@app.get("/health")
def health():
    """Health check"""
    return {"status": "healthy"}


@app.get("/models")
def list_models():
    """List available LLM models"""
    return {"models": list(MODELS.keys())}


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    """
    Chat endpoint.
    In a real app, this would call the LLM provider.
    """
    if body.model not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{body.model}' not found. Available: {list(MODELS.keys())}"
        )

    return ChatResponse(
        response=f"Simulated response from {MODELS[body.model]}",
        model=body.model,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**2. Run the server:**

```bash
cd /path/to/projects/fastapi_concepts_hands_on
python 08_1_cors_middleware.py
```

### Test It

```bash
# ── Test 1: Simple request (no CORS needed for curl, but shows endpoint works) ──
curl http://localhost:8000/health
curl http://localhost:8000/models

# ── Test 2: Simulate browser preflight OPTIONS request ──
curl -v -X OPTIONS http://localhost:8000/chat \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type, Authorization"
# Look for these response headers:
#   access-control-allow-origin: http://localhost:3000
#   access-control-allow-methods: ...
#   access-control-allow-headers: ...

# ── Test 3: Simulate browser actual request with Origin header ──
curl -v -X POST http://localhost:8000/chat \
  -H "Origin: http://localhost:3000" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3", "messages": [{"role": "user", "content": "Hello"}]}'
# Look for: access-control-allow-origin: http://localhost:3000

# ── Test 4: Test blocked origin ──
curl -v -X POST http://localhost:8000/chat \
  -H "Origin: http://evil-site.com" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3", "messages": [{"role": "user", "content": "Hello"}]}'
# access-control-allow-origin header should NOT be present
# (Browser would block this request)
```

### What You Should See

```bash
# Test 2: Preflight response (HTTP 200, no body)
HTTP/1.1 200 OK
access-control-allow-origin: http://localhost:3000
access-control-allow-methods: GET, POST, PUT, DELETE, OPTIONS
access-control-allow-headers: Content-Type, Authorization, X-API-Key, X-Request-ID
access-control-allow-credentials: true

# Test 3: Actual request response
HTTP/1.1 200 OK
access-control-allow-origin: http://localhost:3000
access-control-allow-credentials: true
access-control-expose-headers: X-Request-ID, X-Process-Time, Retry-After
content-type: application/json
{"response":"Simulated response from Anthropic Claude 3","model":"claude-3"}

# Test 4: Blocked origin — no CORS headers in response
HTTP/1.1 200 OK
content-type: application/json
# Notice: NO access-control-allow-origin header!
# Browser would block the frontend from reading this response
```

### Understanding the Flow

```
Browser at http://localhost:3000
    │
    │  1. fetch("http://localhost:8000/chat", {method: "POST", ...})
    │
    │  2. Browser checks: "Different origin! Send preflight first."
    │
    │  OPTIONS /chat
    │  Origin: http://localhost:3000
    │  Access-Control-Request-Method: POST
    │  Access-Control-Request-Headers: Content-Type
    │─────────────────────────────────────────────→ FastAPI
    │                                                │
    │  3. CORSMiddleware checks:                     │
    │     "Is localhost:3000 in allow_origins?"       │
    │     YES → return CORS headers                  │
    │                                                │
    │  HTTP 200                                       │
    │  access-control-allow-origin: localhost:3000    │
    │←─────────────────────────────────────────────── │
    │                                                │
    │  4. Browser: "Server says OK, proceed."        │
    │                                                │
    │  POST /chat (actual request)                   │
    │─────────────────────────────────────────────→   │
    │                                                │
    │  Response + CORS headers                       │
    │←─────────────────────────────────────────────── │
```

### Key Takeaway

CORS is a browser security feature, not a server-side concern. Your API must explicitly allow cross-origin requests by including CORS response headers. Without `CORSMiddleware`, any React/Next.js frontend on a different port will be blocked by the browser. The API response still works (curl proves it), but the browser refuses to pass it to JavaScript.

---

## Exercise 2: Create Request ID Middleware (30 minutes)

### Goal

Build custom middleware that assigns a unique ID to every request. This ID flows through the entire request lifecycle and appears in response headers for debugging.

### Steps

**1. Create `projects/fastapi_concepts_hands_on/08_2_request_id_middleware.py`:**

```python
"""
Exercise 2: Request ID Middleware

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**2. Run the server:**

```bash
cd /path/to/projects/fastapi_concepts_hands_on
python 08_2_request_id_middleware.py
```

### Test It

```bash
# ── Test 1: Auto-generated request ID ──
curl -v http://localhost:8000/health
# Look for: X-Request-ID: <auto-generated-uuid> in response headers
# Response body also includes the request_id

# ── Test 2: Custom request ID (client-provided) ──
curl -v http://localhost:8000/health \
  -H "X-Request-ID: my-debug-id-123"
# Look for: X-Request-ID: my-debug-id-123 in response headers

# ── Test 3: Request ID in error responses ──
curl -v http://localhost:8000/conversations/nonexistent
# Even 404 responses have X-Request-ID header
# Response body includes request_id for support reference

# ── Test 4: Request ID in chat flow ──
curl -v -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: trace-abc-123" \
  -d '{"model": "claude-3", "messages": [{"role": "user", "content": "Hello"}]}'
# All logs for this request show: id=trace-abc-123
```

### What You Should See

```bash
# Test 1: Auto-generated ID
HTTP/1.1 200 OK
x-request-id: a1b2c3d4-e5f6-7890-abcd-ef1234567890
{"status":"healthy","request_id":"a1b2c3d4-e5f6-7890-abcd-ef1234567890"}

# Test 2: Custom ID
HTTP/1.1 200 OK
x-request-id: my-debug-id-123
{"status":"healthy","request_id":"my-debug-id-123"}

# Test 3: Error with request ID
HTTP/1.1 404 Not Found
x-request-id: <uuid>
{"detail":{"error":"conversation_not_found","conversation_id":"nonexistent","request_id":"<uuid>","message":"Conversation 'nonexistent' not found. Check your conversation ID."}}

# Server logs (all correlated by request ID):
INFO  request_start | id=trace-abc-123 method=POST path=/chat
INFO  chat_request | id=trace-abc-123 model=claude-3 msg_count=1
INFO  llm_call | id=trace-abc-123 provider=anthropic model=claude-3
INFO  request_end | id=trace-abc-123 status=200
```

### Understanding the Pattern

```
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
```

### Key Takeaway

Request ID middleware is the most valuable middleware for production GenAI apps. It creates a correlation ID that ties together all logs for a single request — from the initial HTTP call, through your auth layer, through the LLM provider call, to the final response. When things go wrong (and with LLM APIs, they will), this is how you debug.

---

## Exercise 3: Add Timing Middleware That Logs Slow Requests (30 minutes)

### Goal

Build timing middleware that measures how long each request takes, adds timing to response headers, and logs warnings when endpoints exceed a threshold. This is critical for GenAI APIs where LLM calls can take 1-30 seconds.

### Steps

**1. Create `projects/fastapi_concepts_hands_on/08_3_timing_middleware.py`:**

```python
"""
Exercise 3: Timing Middleware for Slow Request Detection

This script demonstrates timing middleware:
- Measuring request processing duration
- Adding X-Process-Time header to responses
- Logging warnings for slow requests
- Simulating slow LLM calls with asyncio.sleep

GenAI Context:
LLM API calls are MUCH slower than typical backend operations (1-30 seconds
vs 1-100ms). Timing middleware helps you:
- Know which endpoints are consistently slow
- Monitor LLM provider latency
- Alert when responses degrade
- Show users the response time (so they know it's working, not hung)

Concepts covered:
- Timing with time.time()
- Threshold-based warning logging
- X-Process-Time response header
- Configurable middleware (constructor parameters)
- asyncio.sleep for simulating LLM latency
"""

import time
import logging
import asyncio

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="GenAI API with Timing")


# ===== TIMING MIDDLEWARE =====

class TimingMiddleware(BaseHTTPMiddleware):
    """
    Measure and log request processing time.

    Features:
    - Records start/end time around endpoint execution
    - Adds X-Process-Time header to every response
    - Logs WARNING for requests exceeding the threshold
    - Configurable threshold via constructor
    """

    def __init__(self, app, slow_threshold: float = 5.0):
        super().__init__(app)
        self.slow_threshold = slow_threshold

    async def dispatch(self, request: Request, call_next):
        # Record start time
        start_time = time.time()

        # Execute endpoint
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Add timing header
        response.headers["X-Process-Time"] = f"{duration:.3f}s"

        # Log based on duration
        if duration > self.slow_threshold:
            logger.warning(
                "🐢 SLOW REQUEST | path=%s method=%s duration=%.3fs threshold=%.1fs",
                request.url.path, request.method, duration, self.slow_threshold
            )
        elif duration > 1.0:
            logger.info(
                "⏱️  moderate_request | path=%s duration=%.3fs",
                request.url.path, duration
            )
        else:
            logger.info(
                "⚡ fast_request | path=%s duration=%.3fs",
                request.url.path, duration
            )

        return response


# Register with 3-second threshold
app.add_middleware(TimingMiddleware, slow_threshold=3.0)


# ===== DATA MODELS =====

class ChatRequest(BaseModel):
    model: str = "claude-3"
    messages: list[dict]


# ===== ENDPOINTS =====

@app.get("/health")
async def health():
    """
    Fast endpoint (~instant).
    Timing middleware will log: ⚡ fast_request
    """
    return {"status": "healthy"}


@app.get("/models")
async def list_models():
    """
    Fast endpoint — just returns a list.
    Expected: < 10ms
    """
    return {
        "models": [
            {"id": "claude-3", "provider": "Anthropic", "latency": "fast"},
            {"id": "gpt-4", "provider": "OpenAI", "latency": "moderate"},
        ]
    }


@app.post("/chat")
async def chat(body: ChatRequest):
    """
    Moderate endpoint — simulates a typical LLM call (~2 seconds).
    Timing middleware will log: ⏱️ moderate_request
    """
    # Simulate LLM API latency (2 seconds)
    await asyncio.sleep(2)

    return {
        "response": f"Response from {body.model}: Hello!",
        "model": body.model,
        "simulated_latency": "2s"
    }


@app.post("/chat/complex")
async def chat_complex(body: ChatRequest):
    """
    Slow endpoint — simulates a complex LLM call (~5 seconds).
    Timing middleware will log: 🐢 SLOW REQUEST WARNING
    """
    # Simulate complex LLM generation (5 seconds)
    await asyncio.sleep(5)

    return {
        "response": f"Complex response from {body.model}: [long analysis...]",
        "model": body.model,
        "simulated_latency": "5s"
    }


@app.post("/chat/timeout")
async def chat_timeout():
    """
    Very slow endpoint — simulates a near-timeout scenario (~8 seconds).
    Timing middleware will log: 🐢 SLOW REQUEST WARNING (well above threshold)
    """
    # Simulate extremely slow LLM call
    await asyncio.sleep(8)

    return {
        "response": "Finally done!",
        "simulated_latency": "8s"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**2. Run the server:**

```bash
cd /path/to/projects/fastapi_concepts_hands_on
python 08_3_timing_middleware.py
```

### Test It

```bash
# ── Test 1: Fast endpoint (health check, < 10ms) ──
curl -v http://localhost:8000/health
# Look for: X-Process-Time: 0.001s
# Server log: ⚡ fast_request | path=/health duration=0.001s

# ── Test 2: Model listing (fast, < 10ms) ──
curl -v http://localhost:8000/models
# X-Process-Time: 0.001s

# ── Test 3: Normal chat (moderate, ~2s) ──
curl -v -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3", "messages": [{"role": "user", "content": "Hi"}]}'
# X-Process-Time: 2.003s
# Server log: ⏱️ moderate_request | path=/chat duration=2.003s

# ── Test 4: Complex chat (slow, ~5s — triggers WARNING) ──
curl -v -X POST http://localhost:8000/chat/complex \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Analyze this"}]}'
# X-Process-Time: 5.002s
# Server log: 🐢 SLOW REQUEST | path=/chat/complex duration=5.002s threshold=3.0s

# ── Test 5: Timeout-like scenario (very slow, ~8s) ──
curl -v -X POST http://localhost:8000/chat/timeout
# X-Process-Time: 8.003s
# Server log: 🐢 SLOW REQUEST | path=/chat/timeout duration=8.003s threshold=3.0s
```

### What You Should See

```bash
# Response headers from Test 3:
HTTP/1.1 200 OK
x-process-time: 2.003s
content-type: application/json
{"response":"Response from claude-3: Hello!","model":"claude-3","simulated_latency":"2s"}

# Server logs (all 5 tests):
INFO  ⚡ fast_request | path=/health duration=0.001s
INFO  ⚡ fast_request | path=/models duration=0.001s
INFO  ⏱️  moderate_request | path=/chat duration=2.003s
WARNING 🐢 SLOW REQUEST | path=/chat/complex duration=5.002s threshold=3.0s
WARNING 🐢 SLOW REQUEST | path=/chat/timeout duration=8.003s threshold=3.0s
```

### Key Takeaway

Timing middleware gives you automatic visibility into endpoint performance without adding timing code to every endpoint. For GenAI APIs, this is essential because LLM calls are orders of magnitude slower than typical operations. The `X-Process-Time` header lets frontends show "Response took 2.3s" to users, and the slow request warnings alert you to degraded performance in production.

---

## Exercise 4: Build Complete Middleware Stack with Proper Ordering (35 minutes)

### Goal

Combine all middleware (CORS + Request ID + Timing + Logging) into a production-ready stack with **correct execution order**. This is the culmination of the middleware topic.

### Steps

**1. Create `projects/fastapi_concepts_hands_on/08_4_complete_middleware_stack.py`:**

```python
"""
Exercise 4: Complete Production-Ready Middleware Stack

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**2. Run the server:**

```bash
cd /path/to/projects/fastapi_concepts_hands_on
python 08_4_complete_middleware_stack.py
```

### Test It

```bash
# ── Test 1: Health check — all middleware runs, fast response ──
curl -v http://localhost:8000/health

# ── Test 2: Normal chat — moderate latency (~2s) ──
curl -v -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3", "messages": [{"role": "user", "content": "Hello!"}]}'

# ── Test 3: Slow chat — triggers timing warning ──
curl -v -X POST http://localhost:8000/chat/slow

# ── Test 4: Error — 404 still gets all middleware headers ──
curl -v -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "nonexistent", "messages": []}'

# ── Test 5: CORS preflight ──
curl -v -X OPTIONS http://localhost:8000/chat \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type"

# ── Test 6: Custom request ID with CORS ──
curl -v -X POST http://localhost:8000/chat \
  -H "Origin: http://localhost:3000" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: my-trace-id-456" \
  -d '{"model": "claude-3", "messages": [{"role": "user", "content": "Hi"}]}'
# Verify: Both X-Request-ID and CORS headers present in response
```

### What You Should See

```bash
# Test 1: Health check response headers
HTTP/1.1 200 OK
x-request-id: abc-123-auto-generated
x-process-time: 0.001s
content-type: application/json
{"status":"healthy"}

# Test 2: Chat response headers
HTTP/1.1 200 OK
x-request-id: def-456-auto-generated
x-process-time: 2.004s
content-type: application/json
{"response":"Simulated response from claude-3","model":"claude-3","tokens_used":42,"request_id":"def-456-auto-generated"}

# Test 4: Error response — still has middleware headers!
HTTP/1.1 404 Not Found
x-request-id: ghi-789-auto-generated
x-process-time: 0.002s
content-type: application/json
{"detail":"Model 'nonexistent' not found. Available: ['claude-3', 'gpt-4']"}

# Test 6: Custom ID with CORS
HTTP/1.1 200 OK
x-request-id: my-trace-id-456
x-process-time: 2.003s
access-control-allow-origin: http://localhost:3000
access-control-allow-credentials: true
access-control-expose-headers: X-Request-ID, X-Process-Time, Retry-After
```

### Server Logs (All Tests Combined)

```
# Test 1: Health check
INFO  → REQUEST  | id=abc-123 GET /health client=127.0.0.1
INFO  ← RESPONSE | id=abc-123 GET /health status=200 duration=0.001s ✓

# Test 2: Normal chat (~2s)
INFO  → REQUEST  | id=def-456 POST /chat client=127.0.0.1
INFO  LLM CALL   | id=def-456 model=claude-3 provider=Anthropic
INFO  ← RESPONSE | id=def-456 POST /chat status=200 duration=2.004s ✓

# Test 3: Slow chat (~5s, triggers warning)
INFO  → REQUEST  | id=ghi-789 POST /chat/slow client=127.0.0.1
WARNING SLOW REQUEST | id=ghi-789 path=/chat/slow duration=5.002s (threshold: 3.0s)
INFO  ← RESPONSE | id=ghi-789 POST /chat/slow status=200 duration=5.002s ✓

# Test 4: Error (404)
INFO  → REQUEST  | id=jkl-012 POST /chat client=127.0.0.1
WARNING ← RESPONSE | id=jkl-012 POST /chat status=404 duration=0.002s ⚠

# Test 6: Custom ID
INFO  → REQUEST  | id=my-trace-id-456 POST /chat client=127.0.0.1
INFO  LLM CALL   | id=my-trace-id-456 model=claude-3 provider=Anthropic
INFO  ← RESPONSE | id=my-trace-id-456 POST /chat status=200 duration=2.003s ✓
```

### Understanding the Full Stack

```
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
```

### Key Takeaway

A complete middleware stack gives you production-ready infrastructure for free on every request. The critical lesson is **ordering**: middleware must be registered in the correct sequence so each layer has access to what it needs. Request IDs must exist before logging can use them. Timing must wrap everything to get accurate durations. CORS must be outermost to handle preflight requests before anything else. This stack (CORS + Request ID + Timing + Logging) is the foundation of every production GenAI API.

---

## What's Next?

After completing these exercises, you have a **production-ready middleware stack**. The next topic is:

**Topic 5: Async FastAPI (Critical for LLMs)** — Understanding why async is essential for GenAI apps where LLM calls take 1-30 seconds. You'll learn `async/await`, `asyncio.gather()` for parallel LLM calls, timeouts, and background tasks.

Your middleware stack from this topic will continue to be used in all future exercises!
