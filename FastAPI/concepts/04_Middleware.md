# Middleware & Request Lifecycle: Cross-Cutting Concerns for GenAI APIs

> **Context**: Every request to your GenAI API needs things beyond business logic—CORS headers for your React frontend, request IDs for debugging failed LLM calls, timing to catch slow completions, logging for production visibility. Middleware handles these cross-cutting concerns in one place instead of repeating them in every endpoint. Think of middleware as the infrastructure layer that wraps every request—your endpoints focus on GenAI logic while middleware handles everything else.

---

## Table of Contents

1. [What Is Middleware and Why It Matters](#what-is-middleware-and-why-it-matters)
2. [Request/Response Lifecycle in FastAPI](#requestresponse-lifecycle-in-fastapi)
3. [CORS Middleware: Enabling Web Frontends](#cors-middleware-enabling-web-frontends)
4. [Custom Middleware: The BaseHTTPMiddleware Pattern](#custom-middleware-the-basehttpmiddleware-pattern)
5. [Request ID Middleware](#request-id-middleware)
6. [Timing Middleware: Finding Slow LLM Calls](#timing-middleware-finding-slow-llm-calls)
7. [Logging Middleware: Debugging in Production](#logging-middleware-debugging-in-production)
8. [Middleware Execution Order](#middleware-execution-order)
9. [Middleware vs Dependencies: When to Use Which](#middleware-vs-dependencies-when-to-use-which)
10. [Complete Working Example](#complete-working-example)
11. [Key Insights for Your Learning](#key-insights-for-your-learning)
12. [Quick Reference](#quick-reference)
13. [Next Steps](#next-steps)

---

## What Is Middleware and Why It Matters

### The Core Concept

Middleware is code that runs on **every single request and response**, regardless of which endpoint is called. It wraps around your endpoint logic like layers of an onion.

```
Without middleware (BAD):
┌─────────────────────────────────────┐
│ Every endpoint repeats:             │
│  - Generate request ID              │
│  - Log incoming request             │
│  - Measure execution time           │
│  - Add CORS headers                 │
│  - Log outgoing response            │
│                                     │
│ @app.get("/chat")                   │
│   request_id = uuid4()  # repeated  │
│   start = time.time()   # repeated  │
│   log(request)          # repeated  │
│   ...actual logic...                │
│   log(response)         # repeated  │
│                                     │
│ @app.get("/models")                 │
│   request_id = uuid4()  # repeated  │
│   start = time.time()   # repeated  │
│   log(request)          # repeated  │
│   ...actual logic...                │
│   log(response)         # repeated  │
└─────────────────────────────────────┘

With middleware (GOOD):
┌─────────────────────────────────────┐
│ Middleware handles once:            │
│  ┌──── Request ID Middleware ────┐  │
│  │ ┌──── Timing Middleware ────┐ │  │
│  │ │ ┌──── Logging MW ──────┐ │ │  │
│  │ │ │                      │ │ │  │
│  │ │ │  @app.get("/chat")   │ │ │  │
│  │ │ │  ...just logic...    │ │ │  │
│  │ │ │                      │ │ │  │
│  │ │ │  @app.get("/models") │ │ │  │
│  │ │ │  ...just logic...    │ │ │  │
│  │ │ │                      │ │ │  │
│  │ │ └──────────────────────┘ │ │  │
│  │ └──────────────────────────┘ │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

**Analogy**: Think of middleware like airport security checkpoints. Every passenger (request) goes through the same checkpoints (middleware), regardless of their destination (endpoint). Security check, passport control, baggage scan—these happen for everyone, not implemented separately at each gate.

### Why GenAI APIs Need Middleware

GenAI applications have specific cross-cutting concerns that middleware handles perfectly:

| Concern | Why It Matters for GenAI | Middleware Solution |
|---------|--------------------------|---------------------|
| **CORS** | React/Next.js chat UI needs to call your API from a different origin | CORSMiddleware |
| **Request IDs** | LLM calls fail—need to trace which request hit which provider | Request ID Middleware |
| **Timing** | LLM calls take 1-30s—need to know which endpoints are slow | Timing Middleware |
| **Logging** | Production debugging requires structured request/response logs | Logging Middleware |
| **Security** | Every request needs security headers regardless of endpoint | Security Middleware |

### Real-World GenAI Scenario

```
Your GenAI app in production:

  React Chat UI ──────── POST /chat ──────→ Your FastAPI API
  (localhost:3000)                           (localhost:8000)
       ↑                                          │
       │                               ┌──────────▼──────────┐
       │                               │ CORS Middleware      │
       │                               │ "Is this origin      │
       │                               │  allowed?"           │
       │                               └──────────┬──────────┘
       │                               ┌──────────▼──────────┐
       │                               │ Request ID MW       │
       │                               │ "ID: abc-123"       │
       │                               └──────────┬──────────┘
       │                               ┌──────────▼──────────┐
       │                               │ Timing MW           │
       │                               │ "Start timer"       │
       │                               └──────────┬──────────┘
       │                               ┌──────────▼──────────┐
       │                               │ /chat endpoint      │
       │                               │ → Call Claude API    │
       │                               │ → 3.2s response     │
       │                               └──────────┬──────────┘
       │                               ┌──────────▼──────────┐
       │                               │ Timing MW           │
       │                               │ "3.2s elapsed"      │
       │                               │ "X-Process-Time:    │
       │                               │  3.2s"              │
       │                               └──────────┬──────────┘
       │                               ┌──────────▼──────────┐
       │                               │ Request ID MW       │
       │                               │ "X-Request-ID:      │
       │                               │  abc-123"           │
       │                               └──────────┬──────────┘
       │                               ┌──────────▼──────────┐
       │                               │ CORS MW             │
       │                               │ "Access-Control-    │
       │                               │  Allow-Origin:      │
       │                               │  localhost:3000"    │
       │                               └──────────┬──────────┘
       │                                          │
       └──────── Response + Headers ◄─────────────┘
```

Without middleware, you'd implement all of this in every single endpoint. With middleware, you write it once.

---

## Request/Response Lifecycle in FastAPI

### The Complete Journey of a Request

Understanding the full lifecycle helps you know **where** to put your logic and **when** it executes.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FULL REQUEST LIFECYCLE                            │
│                                                                     │
│  Client (browser/curl)                                              │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────────┐                                            │
│  │  ASGI Server         │  (Uvicorn receives raw HTTP)              │
│  │  (Uvicorn)           │                                           │
│  └──────────┬──────────┘                                            │
│             │                                                       │
│             ▼                                                       │
│  ┌─────────────────────┐                                            │
│  │  Middleware Stack    │  (Process request top-down)                │
│  │  ┌─────────────┐    │                                            │
│  │  │ Middleware 1 │────│──→ Can modify request, add state          │
│  │  └──────┬──────┘    │                                            │
│  │  ┌──────▼──────┐    │                                            │
│  │  │ Middleware 2 │────│──→ Can short-circuit (return early)       │
│  │  └──────┬──────┘    │                                            │
│  │  ┌──────▼──────┐    │                                            │
│  │  │ Middleware N │────│──→ Last middleware before routing          │
│  │  └──────┬──────┘    │                                            │
│  └─────────┼───────────┘                                            │
│            │                                                        │
│            ▼                                                        │
│  ┌─────────────────────┐                                            │
│  │  Router              │  (Match URL to endpoint function)         │
│  └──────────┬──────────┘                                            │
│             │                                                       │
│             ▼                                                       │
│  ┌─────────────────────┐                                            │
│  │  Exception Handlers  │  (Catch exceptions from endpoint/deps)    │
│  └──────────┬──────────┘                                            │
│             │                                                       │
│             ▼                                                       │
│  ┌─────────────────────┐                                            │
│  │  Dependencies        │  (Resolve Depends() parameters)           │
│  │  (from Topic 2)      │  auth, validation, DB sessions            │
│  └──────────┬──────────┘                                            │
│             │                                                       │
│             ▼                                                       │
│  ┌─────────────────────┐                                            │
│  │  Endpoint Function   │  (Your business logic executes)           │
│  └──────────┬──────────┘                                            │
│             │                                                       │
│             ▼                                                       │
│  ┌─────────────────────┐                                            │
│  │  Response Model      │  (Pydantic validates response)            │
│  └──────────┬──────────┘                                            │
│             │                                                       │
│             ▼                                                       │
│  ┌─────────────────────┐                                            │
│  │  Middleware Stack    │  (Process response bottom-up)              │
│  │  ┌─────────────┐    │                                            │
│  │  │ Middleware N │────│──→ Can modify response headers            │
│  │  └──────┬──────┘    │                                            │
│  │  ┌──────▼──────┐    │                                            │
│  │  │ Middleware 2 │────│──→ Can add timing headers                 │
│  │  └──────┬──────┘    │                                            │
│  │  ┌──────▼──────┐    │                                            │
│  │  │ Middleware 1 │────│──→ Can log the final response             │
│  │  └──────┬──────┘    │                                            │
│  └─────────┼───────────┘                                            │
│            │                                                        │
│            ▼                                                        │
│  Client receives response                                           │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Concepts in the Lifecycle

#### 1. `request.state` — Passing Data Through the Lifecycle

`request.state` is a container that lets middleware share data with endpoints and other middleware:

```python
# Middleware sets data
request.state.request_id = "abc-123"
request.state.start_time = time.time()

# Endpoint reads data
@app.get("/chat")
async def chat(request: Request):
    print(request.state.request_id)  # "abc-123"
```

**What's happening:**
- `request.state` is an empty object that you can attach any attributes to
- Middleware sets attributes early in the lifecycle
- Endpoints and exception handlers read those attributes later
- This is how you pass data across layers without global variables

**Connection to Topic 3**: Remember how exception handlers used `request.state.request_id`? Middleware is where that value gets set.

#### 2. Exception Handling in the Lifecycle

```
Exception raised in endpoint
         │
         ▼
┌────────────────────────┐
│ Exception handlers     │  ← From Topic 3 (@app.exception_handler)
│ catch the exception    │
│ and return JSONResponse│
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│ Middleware still runs   │  ← Response passes through middleware
│ on the response!       │     (timing, logging still work)
└────────────────────────┘
```

**Important**: Even when an exception is raised, middleware **still processes the response**. This means:
- Timing middleware still captures the duration (even for error responses)
- Logging middleware still logs the error response
- Request ID middleware still adds the ID to the error response

This is a key difference from dependencies—if a dependency raises an exception, other dependencies may not run. But middleware always runs on the response.

#### 3. Short-Circuiting

Middleware can return a response **without calling the endpoint**:

```python
class MaintenanceModeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if MAINTENANCE_MODE and request.url.path != "/health":
            # Short-circuit: never reaches the endpoint
            return JSONResponse(
                status_code=503,
                content={"error": "Service under maintenance"}
            )

        # Normal flow: call the endpoint
        response = await call_next(request)
        return response
```

**GenAI use case**: During a provider outage, you might short-circuit all `/chat` requests with a maintenance message instead of attempting (and failing) LLM calls.

---

## CORS Middleware: Enabling Web Frontends

### What Is CORS?

**CORS** (Cross-Origin Resource Sharing) is a browser security feature that blocks web pages from making requests to a different domain/port than the one they were loaded from.

```
The Problem:

  Your React Chat UI         Your FastAPI API
  http://localhost:3000      http://localhost:8000
         │                            │
         │  POST /chat ──────────────→│
         │                            │
         │  ← BLOCKED BY BROWSER! ────│
         │                            │
  Browser says: "You're on port 3000, │
  but trying to reach port 8000.      │
  That's a different origin.          │
  I won't allow this unless the       │
  server says it's OK."               │
```

**Why browsers do this**: Without CORS, any website you visit could make requests to your bank's API using your cookies. CORS prevents this by requiring the server to explicitly allow cross-origin requests.

### The Preflight Request Flow

For non-simple requests (POST with JSON, custom headers), browsers send a "preflight" OPTIONS request first:

```
Browser                          Your FastAPI API
   │                                    │
   │  1. OPTIONS /chat                  │
   │  Origin: http://localhost:3000     │
   │  Access-Control-Request-Method:    │
   │    POST                            │
   │  Access-Control-Request-Headers:   │
   │    Content-Type, Authorization     │
   │──────────────────────────────────→ │
   │                                    │
   │  2. Response (no body)             │
   │  Access-Control-Allow-Origin:      │
   │    http://localhost:3000           │
   │  Access-Control-Allow-Methods:     │
   │    GET, POST, DELETE               │
   │  Access-Control-Allow-Headers:     │
   │    Content-Type, Authorization     │
   │ ←────────────────────────────────  │
   │                                    │
   │  3. Browser checks: "OK, the      │
   │     server allows this origin."    │
   │                                    │
   │  4. POST /chat                     │
   │  Content-Type: application/json    │
   │  Authorization: Bearer <token>     │
   │  {"messages": [...]}               │
   │──────────────────────────────────→ │
   │                                    │
   │  5. Response (with CORS headers)   │
   │  Access-Control-Allow-Origin:      │
   │    http://localhost:3000           │
   │  {"response": "Hello!"}           │
   │ ←────────────────────────────────  │
```

### Configuring CORS in FastAPI

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ CORRECT: Specific origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",        # React dev server
        "http://localhost:5173",        # Vite dev server
        "https://your-app.vercel.app",  # Production frontend
    ],
    allow_credentials=True,     # Allow cookies/auth headers
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-API-Key",
        "X-Request-ID",
    ],
)
```

**What each parameter does:**

| Parameter | Purpose | Common Values |
|-----------|---------|---------------|
| `allow_origins` | Which domains can call your API | Specific URLs (never `["*"]` with credentials) |
| `allow_credentials` | Allow cookies/Authorization header | `True` for authenticated apps |
| `allow_methods` | Which HTTP methods are allowed | `["GET", "POST", "PUT", "DELETE", "OPTIONS"]` |
| `allow_headers` | Which request headers are allowed | `["Content-Type", "Authorization", "X-API-Key"]` |

### ✅ Correct vs ❌ Incorrect CORS Configuration

```python
# ❌ WRONG: Wildcard origin with credentials
# Browsers will reject this combination!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Wildcard
    allow_credentials=True,        # + Credentials
    # Browser error: "Cannot use wildcard origin with credentials"
)

# ❌ WRONG: Missing Authorization header
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Content-Type"],  # Missing Authorization!
    # Result: Bearer token requests will be blocked
)

# ✅ CORRECT: Specific origins, all needed headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

# ✅ CORRECT: Development (permissive) vs Production (strict)
import os

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "development":
    origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
    ]
else:
    origins = [
        "https://your-app.vercel.app",
        "https://your-custom-domain.com",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)
```

### Common CORS Debugging

When CORS isn't working, you'll see browser console errors like:

```
Access to fetch at 'http://localhost:8000/chat' from origin
'http://localhost:3000' has been blocked by CORS policy:
No 'Access-Control-Allow-Origin' header is present on the
requested resource.
```

**Debugging checklist:**
1. Is `CORSMiddleware` added to your app? (most common issue)
2. Is your frontend's exact origin in `allow_origins`? (including port!)
3. Are the headers your frontend sends in `allow_headers`?
4. If using auth, is `allow_credentials=True`?
5. Is the origin `http://` vs `https://` correct?

```python
# Common mistake: port mismatch
# Frontend runs on http://localhost:3000
# But you have:
allow_origins=["http://localhost:8000"]  # ❌ This is YOUR API's port!
allow_origins=["http://localhost:3000"]  # ✅ This is your FRONTEND's port!
```

### GenAI-Specific CORS Considerations

For GenAI apps, you often need additional headers:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type",       # JSON body (chat messages)
        "Authorization",      # JWT bearer token
        "X-API-Key",          # API key authentication
        "X-Request-ID",       # Client-provided request ID
    ],
    expose_headers=[
        "X-Request-ID",       # Let frontend read request ID
        "X-Process-Time",     # Let frontend show response time
        "Retry-After",        # Let frontend handle rate limits
    ],
)
```

**`expose_headers`** is often overlooked—by default, browsers only expose a few standard response headers to JavaScript. If your frontend needs to read custom headers (like `X-Request-ID` for error reporting), you must list them in `expose_headers`.

---

## Custom Middleware: The BaseHTTPMiddleware Pattern

### The Standard Pattern

FastAPI (via Starlette) provides `BaseHTTPMiddleware` as the standard way to create custom middleware:

```python
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

class MyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # ──── BEFORE the endpoint runs ────
        # Modify request, add state, check conditions

        # Call the next middleware or endpoint
        response = await call_next(request)

        # ──── AFTER the endpoint runs ────
        # Modify response, add headers, log results

        return response
```

**What's happening:**
- `dispatch()` is called for every request
- `request` is the incoming HTTP request
- `call_next(request)` passes the request to the next middleware (or to the endpoint if this is the last middleware)
- Everything before `call_next()` runs **before** your endpoint
- Everything after `call_next()` runs **after** your endpoint
- You must return the `response`

### The Anatomy of Middleware

```python
class ExampleMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, some_config: str = "default"):
        super().__init__(app)
        self.some_config = some_config  # Configuration stored at startup

    async def dispatch(self, request: Request, call_next):
        # ┌─── PRE-PROCESSING (runs before endpoint) ───────────────┐
        # │ - Read request headers                                   │
        # │ - Add data to request.state                              │
        # │ - Validate conditions                                    │
        # │ - Short-circuit if needed (return Response without       │
        # │   calling call_next)                                     │
        # └──────────────────────────────────────────────────────────┘

        request.state.custom_data = "set by middleware"

        # ┌─── CALL NEXT (invokes endpoint or next middleware) ──────┐
        response = await call_next(request)
        # └──────────────────────────────────────────────────────────┘

        # ┌─── POST-PROCESSING (runs after endpoint) ───────────────┐
        # │ - Read response status code                              │
        # │ - Add response headers                                   │
        # │ - Log results                                            │
        # │ - Cannot modify response body (it's already streamed)    │
        # └──────────────────────────────────────────────────────────┘

        response.headers["X-Custom-Header"] = "value"

        return response
```

### Registering Middleware

```python
from fastapi import FastAPI

app = FastAPI()

# Method 1: app.add_middleware() — most common
app.add_middleware(MyMiddleware, some_config="custom_value")

# Method 2: @app.middleware("http") decorator — for simple cases
@app.middleware("http")
async def simple_middleware(request: Request, call_next):
    # Same pattern: before → call_next → after
    response = await call_next(request)
    response.headers["X-Simple"] = "true"
    return response
```

**When to use which:**
- `BaseHTTPMiddleware` class: When you need configuration, state, or complex logic
- `@app.middleware("http")` decorator: For simple, one-off middleware (a few lines)

### Error Handling in Middleware

```python
class SafeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            # If call_next raises an unhandled exception,
            # catch it here and return a safe response
            return JSONResponse(
                status_code=500,
                content={"error": "internal_server_error"}
            )
```

**Important**: In practice, FastAPI's exception handlers (from Topic 3) catch most exceptions before they reach middleware. But it's good practice to have a safety net in critical middleware.

---

## Request ID Middleware

### Why Request IDs Are Critical

In production, when a user reports "my chat request failed," you need to find that **specific request** in your logs across potentially millions of log entries.

```
Without Request IDs:
  User: "My request failed!"
  You: *searches logs for errors*
  You: *finds 500 errors from 50 different users*
  You: "Which one is yours? 🤷"

With Request IDs:
  User: "My request failed! ID: abc-123-def"
  You: grep "abc-123-def" logs.txt
  You: *immediately finds the exact request, sees Claude API returned 429*
  You: "Claude was rate-limited. Try again in 60 seconds."
```

### Implementation

You saw a basic version of Request ID middleware in Topic 3's error handling section. Here's the full, production-ready version:

```python
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Add a unique request ID to every request.

    - Generates a UUID if client doesn't provide one
    - Accepts client-provided IDs via X-Request-ID header
    - Stores ID in request.state for use in endpoints/handlers
    - Adds ID to response headers for client reference
    """

    async def dispatch(self, request: Request, call_next):
        # Accept client-provided ID, or generate a new one
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Store in request.state (accessible in endpoints, handlers, other middleware)
        request.state.request_id = request_id

        # Call the endpoint
        response = await call_next(request)

        # Add ID to response headers (client can log this for support)
        response.headers["X-Request-ID"] = request_id

        return response


# Register
app.add_middleware(RequestIDMiddleware)
```

**What's happening:**
1. Check if the client sent an `X-Request-ID` header (some clients generate their own for end-to-end tracing)
2. If not, generate a UUID v4 (random, unique)
3. Store it in `request.state.request_id` so endpoints, dependencies, and exception handlers can access it
4. After the response is generated, add the ID to the response headers
5. Client receives the ID in the response and can reference it for debugging

### Using Request ID Everywhere

Once the middleware sets the request ID, it's available throughout the lifecycle:

```python
# In endpoints
@app.get("/chat")
async def chat(request: Request):
    print(f"Processing request {request.state.request_id}")
    # ...

# In exception handlers (from Topic 3)
@app.exception_handler(ProviderError)
async def handle_provider_error(request: Request, exc: ProviderError):
    logger.error("provider_error", extra={
        "request_id": request.state.request_id,
        "provider": exc.provider
    })
    # ...

# In dependencies
async def get_current_user(request: Request):
    logger.info("auth_check", extra={
        "request_id": request.state.request_id
    })
    # ...
```

### GenAI Use Case: Tracing LLM Calls

```python
@app.post("/chat")
async def chat(request: Request, body: ChatRequest):
    request_id = request.state.request_id

    logger.info("llm_call_start", extra={
        "request_id": request_id,
        "provider": "anthropic",
        "model": body.model
    })

    try:
        response = await anthropic_client.messages.create(
            model=body.model,
            messages=body.messages,
            # Pass request ID to provider for their logging
            metadata={"request_id": request_id}
        )

        logger.info("llm_call_success", extra={
            "request_id": request_id,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
            "latency_ms": response.latency_ms
        })

    except Exception as e:
        logger.error("llm_call_failed", extra={
            "request_id": request_id,
            "error": str(e)
        })
        raise

    return response
```

Now in your logs, you can trace a single user's chat request through your entire system:

```
INFO  request_id=abc-123 "incoming_request" method=POST path=/chat
INFO  request_id=abc-123 "auth_check" user=user_456
INFO  request_id=abc-123 "llm_call_start" provider=anthropic model=claude-3
INFO  request_id=abc-123 "llm_call_success" tokens=150 latency=2300ms
INFO  request_id=abc-123 "response_sent" status=200 duration=2.4s
```

---

## Timing Middleware: Finding Slow LLM Calls

### Why Timing Matters for GenAI

LLM API calls are **orders of magnitude slower** than typical backend operations:

```
Operation                    Typical Latency
─────────────────────────────────────────────
Database query               1-50ms
Redis cache lookup           <1ms
Regular API response         10-100ms
LLM API call (short)         500ms-2s
LLM API call (medium)        2-10s
LLM API call (long/complex)  10-30s
LLM API call (streaming)     First token: 500ms, Full: 5-30s
```

Without timing middleware, you have no visibility into which requests are slow. With it, you can:
- Identify endpoints that consistently exceed thresholds
- Monitor LLM provider latency trends
- Alert on degraded performance
- Show response time to users (so they know it's working, not hung)

### Implementation

```python
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

logger = logging.getLogger(__name__)

class TimingMiddleware(BaseHTTPMiddleware):
    """
    Measure and log request processing time.

    - Adds X-Process-Time header to every response
    - Logs warnings for slow requests (above threshold)
    - Helps identify slow LLM calls in production
    """

    def __init__(self, app, slow_threshold: float = 5.0):
        super().__init__(app)
        self.slow_threshold = slow_threshold  # seconds

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Call the endpoint
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Add timing header to response
        response.headers["X-Process-Time"] = f"{duration:.3f}s"

        # Store in request.state for other middleware to use
        request.state.process_time = duration

        # Log slow requests
        if duration > self.slow_threshold:
            logger.warning(
                "slow_request",
                extra={
                    "request_id": getattr(request.state, "request_id", "unknown"),
                    "method": request.method,
                    "path": request.url.path,
                    "duration_seconds": round(duration, 3),
                    "threshold": self.slow_threshold,
                    "status_code": response.status_code
                }
            )
        else:
            logger.info(
                "request_completed",
                extra={
                    "request_id": getattr(request.state, "request_id", "unknown"),
                    "method": request.method,
                    "path": request.url.path,
                    "duration_seconds": round(duration, 3),
                    "status_code": response.status_code
                }
            )

        return response


# Register with custom threshold
app.add_middleware(TimingMiddleware, slow_threshold=5.0)
```

**What's happening:**
1. Record `start_time` before the endpoint runs
2. Call `call_next(request)` — this includes all endpoint execution, database queries, LLM calls
3. Calculate the duration after the endpoint returns
4. Add `X-Process-Time` header so the client (or browser dev tools) can see the timing
5. If the request took longer than the threshold, log a WARNING for easy monitoring
6. All requests get their timing logged at INFO level

### Reading Timing Headers

```bash
# curl -v shows response headers
curl -v http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3", "messages": [{"role": "user", "content": "Hello"}]}'

# Response headers include:
# < X-Process-Time: 2.347s
# < X-Request-ID: abc-123-def
```

### GenAI-Specific Timing Patterns

```python
# Different thresholds for different endpoint types
ENDPOINT_THRESHOLDS = {
    "/health": 0.1,        # Health checks should be instant
    "/models": 0.5,        # Model listing should be fast
    "/chat": 10.0,         # Chat can take up to 10s (LLM call)
    "/chat/stream": 30.0,  # Streaming can take longer
}

class SmartTimingMiddleware(BaseHTTPMiddleware):
    """Timing middleware with per-endpoint thresholds"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        response.headers["X-Process-Time"] = f"{duration:.3f}s"

        # Get threshold for this specific endpoint
        threshold = ENDPOINT_THRESHOLDS.get(
            request.url.path,
            5.0  # Default threshold
        )

        if duration > threshold:
            logger.warning(
                "slow_request",
                extra={
                    "request_id": getattr(request.state, "request_id", "unknown"),
                    "path": request.url.path,
                    "duration": round(duration, 3),
                    "threshold": threshold,
                    "exceeded_by": round(duration - threshold, 3)
                }
            )

        return response
```

---

## Logging Middleware: Debugging in Production

### Why Logging Middleware?

In development, you can see print statements and debugger output. In production, **logs are your only window** into what's happening. Logging middleware ensures every request and response is recorded.

### Implementation

```python
import logging
import time
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Log every request and response with structured data.

    Logs:
    - Incoming request: method, path, client IP
    - Outgoing response: status code, duration
    - Does NOT log: request bodies (may contain sensitive prompts)
    """

    async def dispatch(self, request: Request, call_next):
        # Get request ID (set by RequestIDMiddleware earlier in the chain)
        request_id = getattr(request.state, "request_id", "unknown")

        # Log incoming request
        logger.info(
            "incoming_request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query": str(request.query_params) if request.query_params else None,
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
            }
        )

        # Track timing
        start_time = time.time()

        # Call the endpoint
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Determine log level based on status code
        if response.status_code >= 500:
            log_func = logger.error
        elif response.status_code >= 400:
            log_func = logger.warning
        else:
            log_func = logger.info

        # Log outgoing response
        log_func(
            "response_sent",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_seconds": round(duration, 3),
            }
        )

        return response


# Register
app.add_middleware(LoggingMiddleware)
```

### What to Log vs What NOT to Log

```python
# ✅ LOG THESE (infrastructure data):
- Request method and path       # "POST /chat"
- Status code                   # 200, 404, 500
- Duration                      # 2.3 seconds
- Client IP                     # For rate limiting / abuse detection
- Request ID                    # For correlation
- User agent                    # Browser/SDK identification

# ❌ DO NOT LOG THESE (sensitive data):
- Request body                  # Contains user prompts (PII!)
- Authorization header          # Contains JWT tokens / API keys
- Full LLM responses            # May contain user-specific data
- Passwords                     # Obviously never log these
- Database connection strings   # Security risk
```

**Why not log request bodies?** In GenAI apps, request bodies contain user prompts. These might include personal information, business secrets, or other sensitive data. Logging them creates privacy and compliance risks.

```python
# ❌ WRONG: Logging full request body
logger.info(f"Request body: {await request.body()}")
# Logs: "Request body: {"messages": [{"role": "user", "content": "My SSN is 123-45-6789, please..."}]}"

# ✅ CORRECT: Log metadata only
logger.info("incoming_request", extra={
    "path": "/chat",
    "content_length": request.headers.get("content-length"),
    "content_type": request.headers.get("content-type"),
})
```

### Structured Logging Format

For production GenAI apps, use structured logging (JSON) instead of plain text:

```python
import json
import logging

class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured log analysis"""

    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        # Add extra fields (request_id, path, etc.)
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "path"):
            log_data["path"] = record.path
        if hasattr(record, "status_code"):
            log_data["status_code"] = record.status_code
        if hasattr(record, "duration_seconds"):
            log_data["duration_seconds"] = record.duration_seconds

        return json.dumps(log_data)


# Setup
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.getLogger().addHandler(handler)
```

**Why structured logging?** In production, you'll use log aggregation tools (e.g., CloudWatch, Datadog, ELK). JSON logs are easily searchable:

```json
{"timestamp": "2024-01-15T10:30:45", "level": "WARNING", "message": "slow_request", "request_id": "abc-123", "path": "/chat", "duration_seconds": 8.5}
```

You can search: `request_id=abc-123` or `duration_seconds > 5` or `level=ERROR AND path=/chat`.

---

## Middleware Execution Order

### The Critical Concept: Stack Order

Middleware executes in a **stack** pattern—like Russian nesting dolls:

```
Registration order:                    Execution order:
app.add_middleware(C)                  Request:  A → B → C → Endpoint
app.add_middleware(B)                  Response: Endpoint → C → B → A
app.add_middleware(A)

IMPORTANT: In FastAPI (Starlette), the LAST middleware added
is the FIRST to process the request (outermost layer).
```

**Wait, that's confusing!** Here's why:

```python
# You register middleware like this:
app.add_middleware(LoggingMiddleware)       # Added first
app.add_middleware(TimingMiddleware)         # Added second
app.add_middleware(RequestIDMiddleware)      # Added third (LAST)
app.add_middleware(CORSMiddleware, ...)      # Added fourth (LAST LAST)

# But they execute like this:
#
# REQUEST flows inward:
#   CORSMiddleware (outermost - added last)
#     → RequestIDMiddleware
#       → TimingMiddleware
#         → LoggingMiddleware (innermost - added first)
#           → Endpoint
#
# RESPONSE flows outward:
#           Endpoint →
#         LoggingMiddleware →
#       TimingMiddleware →
#     RequestIDMiddleware →
#   CORSMiddleware →
# Client
```

**Think of it like wrapping a gift**: The last wrapper you put on is the first one someone unwraps.

### Visual: Middleware Stack Execution

```
┌────────────────────────────────────────────────────┐
│ CORS Middleware (added last = outermost)            │
│  ┌──── Request enters ────────────────────────┐    │
│  │ Check origin, add CORS headers             │    │
│  │                                            │    │
│  │  ┌─────────────────────────────────────┐   │    │
│  │  │ Request ID Middleware               │   │    │
│  │  │  Generate/accept X-Request-ID       │   │    │
│  │  │  Store in request.state             │   │    │
│  │  │                                     │   │    │
│  │  │  ┌──────────────────────────────┐   │   │    │
│  │  │  │ Timing Middleware            │   │   │    │
│  │  │  │  Record start_time           │   │   │    │
│  │  │  │                              │   │   │    │
│  │  │  │  ┌───────────────────────┐   │   │   │    │
│  │  │  │  │ Logging Middleware    │   │   │   │    │
│  │  │  │  │  Log: incoming_request│   │   │   │    │
│  │  │  │  │                       │   │   │   │    │
│  │  │  │  │  ┌────────────────┐   │   │   │   │    │
│  │  │  │  │  │   ENDPOINT     │   │   │   │   │    │
│  │  │  │  │  │   /chat        │   │   │   │   │    │
│  │  │  │  │  │   (LLM call)   │   │   │   │   │    │
│  │  │  │  │  └────────────────┘   │   │   │   │    │
│  │  │  │  │                       │   │   │   │    │
│  │  │  │  │  Log: response_sent   │   │   │   │    │
│  │  │  │  └───────────────────────┘   │   │   │    │
│  │  │  │                              │   │   │    │
│  │  │  │  Calculate duration          │   │   │    │
│  │  │  │  Add X-Process-Time header   │   │   │    │
│  │  │  └──────────────────────────────┘   │   │    │
│  │  │                                     │   │    │
│  │  │  Add X-Request-ID header            │   │    │
│  │  └─────────────────────────────────────┘   │    │
│  │                                            │    │
│  │ Add Access-Control-Allow-Origin header     │    │
│  └──── Response exits ────────────────────────┘    │
│                                                    │
└────────────────────────────────────────────────────┘
```

### Why Order Matters for GenAI APIs

The correct order for a GenAI API middleware stack:

```python
# ⚠️ ORDER MATTERS — add in this sequence:
# (Remember: LAST added = FIRST to execute on request)

# 4. Logging — innermost, has access to all state set by outer middleware
app.add_middleware(LoggingMiddleware)

# 3. Timing — wraps logging + endpoint to measure total time
app.add_middleware(TimingMiddleware, slow_threshold=5.0)

# 2. Request ID — generates ID early so all middleware can use it
app.add_middleware(RequestIDMiddleware)

# 1. CORS — outermost, handles preflight before anything else
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Why this order?**

| Position | Middleware | Reason |
|----------|-----------|--------|
| 1st (outermost) | CORS | Must handle preflight OPTIONS requests immediately |
| 2nd | Request ID | Generates ID that all subsequent middleware and the endpoint need |
| 3rd | Timing | Wraps everything inside to get accurate total duration |
| 4th (innermost) | Logging | Runs closest to endpoint, has access to request ID and timing |

**If you get the order wrong:**
```python
# ❌ WRONG ORDER: Logging added after Request ID
app.add_middleware(RequestIDMiddleware)   # Added last = outermost
app.add_middleware(LoggingMiddleware)      # Added first = innermost

# Result: LoggingMiddleware runs BEFORE RequestIDMiddleware
# request.state.request_id doesn't exist yet when logging tries to read it!
# Logs: request_id=unknown (BAD)
```

---

## Middleware vs Dependencies: When to Use Which

### The Decision Tree

```
Need cross-cutting logic?
│
├─ Does it need to run on EVERY request (including errors)?
│  ├─ Yes → Use Middleware
│  │   Examples: CORS, Request ID, Timing, Logging
│  │
│  └─ No, only specific endpoints → Use Dependency
│      Examples: Auth, Rate Limiting, Input Validation
│
├─ Does it need access to path/query parameters?
│  ├─ Yes → Use Dependency (middleware can't access typed params)
│  │   Examples: validate_model(model_id: str)
│  │
│  └─ No → Either works, but middleware is simpler for global needs
│
├─ Does it need to modify the response headers?
│  ├─ Yes → Use Middleware (dependencies can't modify response)
│  │   Examples: X-Request-ID, X-Process-Time, CORS headers
│  │
│  └─ No → Dependency is fine
│
└─ Does it produce a value used by the endpoint?
   ├─ Yes → Use Dependency (returns value via Depends())
   │   Examples: get_current_user(), get_db_session()
   │
   └─ No, it's side effects only → Use Middleware
       Examples: logging, timing, header injection
```

### Comparison Table

| Feature | Middleware | Dependency |
|---------|-----------|------------|
| **Runs on** | Every request | Only endpoints that declare it |
| **Access to path params** | No (only raw request) | Yes (`model_id: str`) |
| **Can modify response** | Yes (headers) | No |
| **Can return values** | No (side effects only) | Yes (`Depends()` returns) |
| **Runs on errors** | Yes (response still passes through) | May not (if exception before dependency resolves) |
| **Registration** | `app.add_middleware()` | `Depends()` in endpoint params |
| **Typical use** | Infrastructure | Business logic |

### GenAI Application: What Goes Where

```python
# ── MIDDLEWARE (infrastructure, runs on everything) ──
app.add_middleware(CORSMiddleware, ...)         # Browser access
app.add_middleware(RequestIDMiddleware)          # Debug tracing
app.add_middleware(TimingMiddleware)             # Performance monitoring
app.add_middleware(LoggingMiddleware)            # Request/response audit

# ── DEPENDENCIES (business logic, per-endpoint) ──
async def verify_api_key(x_api_key: str = Header()):
    """Auth — returns user info"""
    ...

async def check_rate_limit(user: User = Depends(verify_api_key)):
    """Rate limiting — depends on knowing the user"""
    ...

async def validate_model(model_id: str, user: User = Depends(verify_api_key)):
    """Model validation — needs path parameter + auth"""
    ...

# ── ENDPOINT uses dependencies, middleware runs automatically ──
@app.post("/chat")
async def chat(
    request: ChatRequest,
    user: User = Depends(verify_api_key),        # Auth
    _rate: None = Depends(check_rate_limit),      # Rate limit
):
    # Middleware already handled: CORS, Request ID, Timing, Logging
    # Dependencies handled: Auth, Rate Limiting
    # Endpoint focuses on: Business logic only
    ...
```

**Connection to Topic 2 (Dependency Injection):** Dependencies were covered in depth in Topic 2. Middleware complements them—dependencies handle per-endpoint business logic, middleware handles global infrastructure.

---

## Complete Working Example

### Full GenAI API with Middleware Stack

This example brings together all middleware types into a production-ready configuration:

```python
"""
Complete Middleware Stack for a GenAI API

Demonstrates:
1. CORS — allows React frontend to call the API
2. Request ID — unique tracking per request
3. Timing — measures endpoint latency
4. Logging — structured request/response logs

Run: uvicorn app:app --reload --port 8000
Test: See curl commands at the bottom
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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("genai_api")


# ===== CUSTOM MIDDLEWARE =====

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to every request/response"""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Measure request processing time"""

    def __init__(self, app, slow_threshold: float = 5.0):
        super().__init__(app)
        self.slow_threshold = slow_threshold

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        response.headers["X-Process-Time"] = f"{duration:.3f}s"
        request.state.process_time = duration

        request_id = getattr(request.state, "request_id", "unknown")

        if duration > self.slow_threshold:
            logger.warning(
                "slow_request | request_id=%s path=%s duration=%.3fs threshold=%.1fs",
                request_id, request.url.path, duration, self.slow_threshold
            )

        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log every request and response"""

    async def dispatch(self, request: Request, call_next):
        request_id = getattr(request.state, "request_id", "unknown")

        logger.info(
            "incoming_request | request_id=%s method=%s path=%s client=%s",
            request_id,
            request.method,
            request.url.path,
            request.client.host if request.client else "unknown"
        )

        response = await call_next(request)

        duration = getattr(request.state, "process_time", 0)

        log_func = logger.error if response.status_code >= 500 else (
            logger.warning if response.status_code >= 400 else logger.info
        )

        log_func(
            "response_sent | request_id=%s method=%s path=%s status=%d duration=%.3fs",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration
        )

        return response


# ===== APPLICATION SETUP =====

app = FastAPI(title="GenAI API with Middleware Stack")


# ===== REGISTER MIDDLEWARE (order matters!) =====
# Last added = first to execute on request

# 4. Logging (innermost — runs closest to endpoint)
app.add_middleware(LoggingMiddleware)

# 3. Timing (wraps logging + endpoint)
app.add_middleware(TimingMiddleware, slow_threshold=5.0)

# 2. Request ID (generates ID for all other middleware)
app.add_middleware(RequestIDMiddleware)

# 1. CORS (outermost — handles preflight first)
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


class ChatResponse(BaseModel):
    response: str
    model: str
    tokens_used: int


# ===== MOCK DATA =====

SUPPORTED_MODELS = {
    "claude-3": {"name": "Claude 3", "provider": "Anthropic", "max_tokens": 4096},
    "gpt-4": {"name": "GPT-4", "provider": "OpenAI", "max_tokens": 8192},
}


# ===== ENDPOINTS =====

@app.get("/health")
async def health():
    """Fast health check — middleware still processes this"""
    return {"status": "healthy"}


@app.get("/models")
async def list_models():
    """List available models — fast endpoint"""
    return {"models": list(SUPPORTED_MODELS.values())}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    """
    Chat endpoint — simulates LLM call with delay.

    This endpoint takes ~2 seconds to respond (simulating LLM latency).
    The timing middleware will capture this duration.
    """
    # Validate model
    if body.model not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{body.model}' not found. Available: {list(SUPPORTED_MODELS.keys())}"
        )

    # Simulate LLM API call delay (2 seconds)
    await asyncio.sleep(2)

    request_id = request.state.request_id
    logger.info(
        "llm_call_completed | request_id=%s model=%s",
        request_id, body.model
    )

    return ChatResponse(
        response=f"Simulated response from {body.model}",
        model=body.model,
        tokens_used=42
    )


@app.post("/chat/slow")
async def chat_slow(request: Request):
    """
    Deliberately slow endpoint — simulates a long LLM completion.
    Will trigger the timing middleware's slow request warning.
    """
    # Simulate very slow LLM call (7 seconds)
    await asyncio.sleep(7)

    return {"response": "This took a while!", "duration": "~7s"}
```

### Running and Testing

```bash
# Start the server
uvicorn app:app --reload --port 8000

# ── Test 1: Health check (fast) ──
curl -v http://localhost:8000/health
# Response headers include:
#   X-Request-ID: <uuid>
#   X-Process-Time: 0.001s
#   access-control-allow-origin (if Origin header sent)

# ── Test 2: Chat endpoint (simulated LLM delay ~2s) ──
curl -v -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3", "messages": [{"role": "user", "content": "Hello"}]}'
# Response headers include:
#   X-Request-ID: <uuid>
#   X-Process-Time: 2.005s

# ── Test 3: Slow endpoint (triggers warning) ──
curl -v -X POST http://localhost:8000/chat/slow
# Response headers include:
#   X-Process-Time: 7.002s
# Server logs: WARNING slow_request | duration=7.002s threshold=5.0s

# ── Test 4: CORS preflight ──
curl -v -X OPTIONS http://localhost:8000/chat \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type, Authorization"
# Response headers include:
#   access-control-allow-origin: http://localhost:3000
#   access-control-allow-methods: GET, POST, PUT, DELETE, OPTIONS
#   access-control-allow-headers: Content-Type, Authorization, X-API-Key, X-Request-ID

# ── Test 5: Custom Request ID ──
curl -v http://localhost:8000/health \
  -H "X-Request-ID: my-custom-id-123"
# Response header: X-Request-ID: my-custom-id-123
# (Uses your ID instead of generating one)

# ── Test 6: 404 error (middleware still runs) ──
curl -v -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "nonexistent", "messages": []}'
# Status: 404
# Headers still include X-Request-ID and X-Process-Time
# Server logs: WARNING response_sent | status=404
```

### What You'll See in Server Logs

```
# Health check (fast)
INFO  incoming_request | request_id=abc-123 method=GET path=/health client=127.0.0.1
INFO  response_sent | request_id=abc-123 method=GET path=/health status=200 duration=0.001s

# Chat endpoint (normal speed)
INFO  incoming_request | request_id=def-456 method=POST path=/chat client=127.0.0.1
INFO  llm_call_completed | request_id=def-456 model=claude-3
INFO  response_sent | request_id=def-456 method=POST path=/chat status=200 duration=2.005s

# Slow endpoint (triggers warning)
INFO  incoming_request | request_id=ghi-789 method=POST path=/chat/slow client=127.0.0.1
WARNING slow_request | request_id=ghi-789 path=/chat/slow duration=7.002s threshold=5.0s
INFO  response_sent | request_id=ghi-789 method=POST path=/chat/slow status=200 duration=7.002s

# Error request (404 — middleware still runs)
INFO  incoming_request | request_id=jkl-012 method=POST path=/chat client=127.0.0.1
WARNING response_sent | request_id=jkl-012 method=POST path=/chat status=404 duration=0.002s
```

---

## Key Insights for Your Learning

```
★ Insight ─────────────────────────────────────

1. Middleware execution order is a stack (FILO): The LAST middleware
   you add with app.add_middleware() is the FIRST to process the
   request (outermost layer). This catches many developers off guard.
   Think: wrapping a gift — last wrapper on = first unwrapped.

2. CORS is non-negotiable for browser-based GenAI apps: If you're
   building a React/Next.js chat UI that calls your FastAPI backend,
   you MUST have CORS middleware. Without it, the browser blocks every
   request. No amount of backend code fixes this — it's a browser
   security feature.

3. Request IDs are your production lifeline: When you're debugging a
   failed LLM call at 2 AM, request IDs let you grep through millions
   of log lines to find the exact request. Without them, you're
   searching blind. Always add them.

4. Middleware vs Dependencies is about scope: Middleware = every request
   (infrastructure). Dependencies = specific endpoints (business logic).
   CORS, timing, logging → middleware. Auth, rate limiting, validation
   → dependencies. Mixing them up works but makes your code harder
   to reason about.

5. Timing reveals hidden performance issues: LLM calls take 1-30
   seconds, but you won't notice until users complain. Timing middleware
   automatically flags slow requests. Set thresholds per endpoint:
   health < 100ms, models < 500ms, chat < 10s.

─────────────────────────────────────────────────
```

---

## Quick Reference

### Middleware Types for GenAI APIs

| Middleware | Purpose | When Required | Key Configuration |
|-----------|---------|---------------|-------------------|
| **CORS** | Enable browser frontend access | Any web app with separate frontend | `allow_origins`, `allow_headers`, `expose_headers` |
| **Request ID** | Unique identifier per request for debugging | Always (production) | UUID generation, `X-Request-ID` header |
| **Timing** | Measure endpoint response time | Always | Threshold (e.g., 5s), `X-Process-Time` header |
| **Logging** | Structured request/response audit trail | Always | Log levels, structured format, sensitive data filtering |

### Registration Order Template

```python
# Add in this order (last added = first to execute):
app.add_middleware(LoggingMiddleware)                    # 4. Innermost
app.add_middleware(TimingMiddleware, slow_threshold=5.0) # 3.
app.add_middleware(RequestIDMiddleware)                  # 2.
app.add_middleware(CORSMiddleware, ...)                  # 1. Outermost
```

### Common Response Headers Added by Middleware

| Header | Source | Purpose |
|--------|--------|---------|
| `X-Request-ID` | Request ID MW | Debug tracing |
| `X-Process-Time` | Timing MW | Show response latency |
| `Access-Control-Allow-Origin` | CORS MW | Browser cross-origin access |
| `Access-Control-Allow-Methods` | CORS MW | Allowed HTTP methods |
| `Access-Control-Allow-Headers` | CORS MW | Allowed request headers |
| `Retry-After` | Error handlers (Topic 3) | Rate limit recovery time |

### Middleware vs Dependencies Decision

| Question | Middleware | Dependency |
|----------|-----------|------------|
| Runs on every request? | ✅ | ❌ (only declared endpoints) |
| Access to typed params? | ❌ | ✅ (`model_id: str`) |
| Can modify response headers? | ✅ | ❌ |
| Can return values to endpoint? | ❌ | ✅ (`Depends()`) |
| Runs even on error responses? | ✅ | ❌ (may not) |
| Best for | Infrastructure | Business logic |

### CORS Debugging Checklist

```
CORS not working? Check these:
1. ☐ CORSMiddleware is added to app
2. ☐ Frontend's EXACT origin is in allow_origins (including port!)
3. ☐ Request headers (Authorization, X-API-Key) are in allow_headers
4. ☐ allow_credentials=True if using auth
5. ☐ Response headers (X-Request-ID) are in expose_headers
6. ☐ Not using allow_origins=["*"] with allow_credentials=True
```

---

## Next Steps

**After completing this topic:**

1. **Practice the hands-on exercises** in `04_Middleware_Practice.md`
   - Exercise 1: Configure CORS for a React frontend
   - Exercise 2: Create Request ID middleware
   - Exercise 3: Add timing middleware with slow request warnings
   - Exercise 4: Build a complete middleware stack with proper ordering

2. **Integrate with your existing code**
   - Add the middleware stack from this topic to your mini_project code
   - Combine with the error handling from Topic 3 (request IDs in exception handlers)
   - Test CORS with a simple frontend or browser fetch

3. **Move to Topic 5: Async FastAPI**
   - Learn why async is **critical** for LLM-powered applications
   - Understand concurrent request handling
   - Build async patterns for parallel LLM calls

**Key Takeaway**: Middleware is the invisible infrastructure that makes your API production-ready. Users won't see it, but they'll feel the difference—fast CORS responses, traceable request IDs in error messages, and performance monitoring that catches slow LLM calls before users complain.

---

*Every production GenAI API needs these four middleware. Build them once, benefit on every request.*
