# Async FastAPI: Concurrency for GenAI Backends

> **Context**: LLM API calls take 1-30 seconds — orders of magnitude slower than typical backend operations. Without async, a single slow Claude or GPT-4 call blocks your entire server from handling other users. Async programming lets your FastAPI server handle hundreds of concurrent LLM requests without wasting CPU time waiting for responses. As a Senior ML Engineer familiar with Spark's parallelism, think of async as concurrency for I/O-bound work — like how Spark parallelizes data tasks across executors, async parallelizes waiting across coroutines on a single thread.

---

## Table of Contents

1. [Why Async Matters for GenAI](#why-async-matters-for-genai)
2. [The Event Loop: How Async Works](#the-event-loop-how-async-works)
3. [async/await Fundamentals in FastAPI](#asyncawait-fundamentals-in-fastapi)
4. [When to Use async vs sync in FastAPI](#when-to-use-async-vs-sync-in-fastapi)
5. [Concurrent Operations with asyncio.gather()](#concurrent-operations-with-asynciogather)
6. [Racing Providers with asyncio.wait()](#racing-providers-with-asynciowait)
7. [Timeout Handling with asyncio.wait_for()](#timeout-handling-with-asynciowait_for)
8. [Background Tasks for Non-Blocking Work](#background-tasks-for-non-blocking-work)
9. [Async HTTP Clients (httpx)](#async-http-clients-httpx)
10. [Error Handling in Concurrent Operations](#error-handling-in-concurrent-operations)
11. [Key Insights](#key-insights)
12. [Quick Reference](#quick-reference)
13. [Next Steps](#next-steps)

---

## Why Async Matters for GenAI

### The Problem: Sync Blocks Everything

With synchronous code, your server can only do **one thing at a time** per worker. When one request waits for an LLM response, everyone else waits too.

```
SYNCHRONOUS SERVER (1 worker):

Time →  0s        2s        4s        6s        8s       10s
        │         │         │         │         │         │
User A: ████████████████████│         │         │         │
        │ Waiting for Claude (3s)     │         │         │
        │         │         │         │         │         │
User B: │ BLOCKED ──────────████████████████████│         │
        │         │         │ Waiting for GPT-4 (3s)      │
        │         │         │         │         │         │
User C: │ BLOCKED ──────────────────────────────██████████│
        │         │         │         │         │ Waiting (3s)
        │         │         │         │         │         │
Total: 9 seconds for 3 requests (sequential)
```

### The Solution: Async Handles Concurrent I/O

With async, while one request waits for an LLM response, the server handles other requests. The CPU isn't doing work during the wait — it's just waiting for network I/O.

```
ASYNC SERVER (1 worker):

Time →  0s        1s        2s        3s
        │         │         │         │
User A: ██████████████████████████████│
        │ Waiting for Claude (3s)     │
User B: ██████████████████████████████│
        │ Waiting for GPT-4 (3s)      │
User C: ██████████████████████████████│
        │ Waiting for Claude (3s)     │
        │         │         │         │
Total: 3 seconds for 3 requests (concurrent!)
```

**The key insight**: LLM calls are **I/O-bound** (waiting for network), not **CPU-bound** (doing computation). During that wait time, the CPU can do other things — like start processing the next request.

### Real-World Impact

| Metric | Sync Server | Async Server |
|--------|-------------|--------------|
| Concurrent users | ~4 (with 4 workers) | ~100+ (single worker) |
| Memory per connection | ~8MB (thread/process) | ~KB (coroutine) |
| LLM call wait time | Blocks worker | Yields to event loop |
| Requests/sec at 3s latency | ~1.3/worker | ~33+/worker |

```
★ Key Insight ─────────────────────────────────────────
LLM APIs are the perfect use case for async. Your server
spends 95%+ of its time waiting for network responses,
not computing. Async lets you use that wait time to serve
other users instead of sitting idle.
─────────────────────────────────────────────────────────
```

---

## The Event Loop: How Async Works

### The Core Concept

Python's `asyncio` event loop is a single-threaded scheduler that manages many concurrent tasks. When one task says "I'm waiting for I/O," the event loop switches to another task.

```
┌──────────────────────────────────────────────────────────┐
│                    EVENT LOOP                              │
│                                                           │
│  Task Queue: [handle_user_A, handle_user_B, handle_user_C]│
│                                                           │
│  Step 1: Run handle_user_A                                │
│          → calls LLM API                                  │
│          → hits "await" → YIELDS control                  │
│          → goes to waiting list                           │
│                                                           │
│  Step 2: Run handle_user_B                                │
│          → calls LLM API                                  │
│          → hits "await" → YIELDS control                  │
│          → goes to waiting list                           │
│                                                           │
│  Step 3: Run handle_user_C                                │
│          → calls LLM API                                  │
│          → hits "await" → YIELDS control                  │
│          → goes to waiting list                           │
│                                                           │
│  Step 4: User A's LLM response arrives!                   │
│          → resume handle_user_A                           │
│          → return response to User A                      │
│                                                           │
│  Step 5: User B's LLM response arrives!                   │
│          → resume handle_user_B                           │
│          → return response to User B                      │
│                                                           │
│  ...and so on                                             │
└──────────────────────────────────────────────────────────┘
```

### Analogy: Restaurant Waiter

Think of the event loop as a single waiter in a restaurant:

- **Sync waiter**: Takes User A's order, goes to kitchen, stands there until food is ready, brings food to User A, THEN takes User B's order. Everyone waits.
- **Async waiter**: Takes User A's order, sends to kitchen, immediately takes User B's order, sends to kitchen, takes User C's order... When food comes up for any table, delivers it.

The waiter (CPU) isn't cooking (that's the LLM provider's job). They're just placing orders and delivering food. Waiting at the kitchen window is wasted time.

```
★ Key Insight ─────────────────────────────────────────
The event loop is NOT parallel execution. It's concurrent
execution on a single thread. Tasks take turns running,
yielding control at "await" points. This works because
I/O (network calls) doesn't need the CPU — it needs the
network card. The CPU is free to do other work while waiting.
─────────────────────────────────────────────────────────
```

---

## async/await Fundamentals in FastAPI

### Defining Async Endpoints

```python
# ✅ ASYNC endpoint — uses the event loop directly
@app.post("/chat")
async def chat(body: ChatRequest):
    # "await" yields control while waiting for the LLM response
    response = await call_llm_provider(body.messages)
    return {"response": response}

# ✅ SYNC endpoint — FastAPI runs this in a thread pool
@app.get("/health")
def health():
    # No async operations needed here
    return {"status": "healthy"}
```

**What's happening:**
- `async def` declares a coroutine — a function that can yield control at `await` points
- `await` pauses the coroutine and returns control to the event loop until the awaited operation completes
- FastAPI handles `async def` endpoints on the event loop and `def` endpoints in a separate thread pool

### The await Keyword

`await` can only be used inside `async def` functions. It means "start this operation and give up control until it finishes."

```python
# ✅ CORRECT: await an async operation
async def chat(body: ChatRequest):
    # This yields control — other requests can be processed while waiting
    result = await httpx_client.post("https://api.anthropic.com/v1/messages", ...)
    return result

# ❌ WRONG: calling sync blocking code in an async function
async def chat(body: ChatRequest):
    # This BLOCKS the event loop! No other requests can be processed!
    result = requests.post("https://api.anthropic.com/v1/messages", ...)
    return result
```

### The Critical Rule: Never Block the Event Loop

```python
# ❌ BAD: Blocking call inside async function
async def process_chat():
    # requests.post is synchronous — blocks the entire event loop
    # ALL other users are frozen while this runs
    response = requests.post("https://api.anthropic.com/...", json=data)
    return response.json()

# ❌ BAD: time.sleep blocks the event loop
async def simulate_delay():
    time.sleep(5)  # Freezes ALL requests for 5 seconds!
    return {"done": True}

# ✅ GOOD: Use async libraries
async def process_chat():
    async with httpx.AsyncClient() as client:
        response = await client.post("https://api.anthropic.com/...", json=data)
    return response.json()

# ✅ GOOD: asyncio.sleep yields control
async def simulate_delay():
    await asyncio.sleep(5)  # Other requests proceed during the wait
    return {"done": True}
```

```
Blocking vs Non-Blocking:

time.sleep(5)           → Event loop FROZEN for 5s. No one served.
await asyncio.sleep(5)  → Event loop FREE for 5s. Others served.

requests.post(url)      → Event loop FROZEN during HTTP call.
await httpx.post(url)   → Event loop FREE during HTTP call.
```

### Async Context Managers

Many async resources need setup and cleanup — connections, sessions, clients. Use `async with`:

```python
# ✅ Async context manager for HTTP client
async def call_provider():
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data)
        return response.json()
    # Client is automatically closed when exiting the block

# ✅ Async context manager for database session
async def get_user(db: AsyncSession):
    async with db.begin():
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
```

---

## When to Use async vs sync in FastAPI

FastAPI supports both `async def` and regular `def` endpoints. The choice matters.

### Decision Guide

```
Does your endpoint do I/O (network calls, database, file reads)?
│
├─ YES: Is the I/O library async-compatible?
│  │
│  ├─ YES (httpx, asyncpg, aiofiles) → Use async def + await
│  │   Example: calling LLM APIs with httpx
│  │
│  └─ NO (requests, psycopg2, open()) → Use regular def
│      FastAPI runs it in a thread pool automatically
│      Example: legacy library without async support
│
└─ NO: Is it pure computation (no I/O)?
   │
   ├─ Fast computation → Either works (def is simpler)
   │   Example: health check, returning static data
   │
   └─ Heavy computation (CPU-bound) → Use def
       FastAPI runs it in a thread pool so it doesn't block
       Example: tokenization, embedding preprocessing
```

### ✅ Correct Choices

```python
# ✅ Async endpoint with async I/O (httpx)
@app.post("/chat")
async def chat(body: ChatRequest):
    async with httpx.AsyncClient() as client:
        resp = await client.post(LLM_URL, json=body.dict())
    return resp.json()

# ✅ Sync endpoint with sync library (requests)
# FastAPI automatically runs this in a thread pool
@app.post("/chat-legacy")
def chat_legacy(body: ChatRequest):
    resp = requests.post(LLM_URL, json=body.dict())
    return resp.json()

# ✅ Simple endpoint — no I/O, either works
@app.get("/health")
def health():
    return {"status": "healthy"}
```

### ❌ Incorrect Choices

```python
# ❌ WRONG: Sync blocking call inside async function
# Blocks the event loop — defeats the entire purpose of async
@app.post("/chat")
async def chat(body: ChatRequest):
    resp = requests.post(LLM_URL, json=body.dict())  # BLOCKS!
    return resp.json()

# ❌ WRONG: Using async def but never awaiting anything
# Wastes the overhead of a coroutine for no benefit
@app.get("/health")
async def health():
    return {"status": "healthy"}  # No await needed, use def instead
```

```
★ Key Insight ─────────────────────────────────────────
The most dangerous pattern in async Python is calling a
synchronous blocking function inside an async endpoint.
It looks correct but freezes the entire event loop. If
you use `async def`, every I/O operation inside MUST use
an async library (httpx, not requests; asyncio.sleep, not
time.sleep; aiosqlite, not sqlite3).
─────────────────────────────────────────────────────────
```

---

## Concurrent Operations with asyncio.gather()

### The Problem: Sequential LLM Calls Are Slow

```python
# ❌ SLOW: Sequential calls — total time = sum of all calls
async def get_analysis(text: str):
    sentiment = await call_llm("Analyze sentiment of: " + text)     # 2s
    summary = await call_llm("Summarize: " + text)                  # 3s
    keywords = await call_llm("Extract keywords from: " + text)     # 1s
    return {"sentiment": sentiment, "summary": summary, "keywords": keywords}
    # Total: 6 seconds (2 + 3 + 1)
```

### The Solution: asyncio.gather() Runs Tasks in Parallel

```python
# ✅ FAST: Parallel calls — total time = max of all calls
import asyncio

async def get_analysis(text: str):
    sentiment, summary, keywords = await asyncio.gather(
        call_llm("Analyze sentiment of: " + text),     # 2s ─┐
        call_llm("Summarize: " + text),                 # 3s ─┤ Run together
        call_llm("Extract keywords from: " + text),     # 1s ─┘
    )
    return {"sentiment": sentiment, "summary": summary, "keywords": keywords}
    # Total: 3 seconds (max of 2, 3, 1)
```

```
Sequential:
  Task A: ████████ (2s)
  Task B:          ██████████████ (3s)
  Task C:                        ████ (1s)
  Total:  ──────────────────────────── 6s

Parallel (asyncio.gather):
  Task A: ████████ (2s)
  Task B: ██████████████ (3s)
  Task C: ████ (1s)
  Total:  ────────────── 3s (= slowest task)
```

### How asyncio.gather() Works

```python
import asyncio

async def call_provider(provider: str, prompt: str) -> dict:
    """Simulate calling an LLM provider"""
    delays = {"anthropic": 2.0, "openai": 1.5, "cohere": 3.0}
    await asyncio.sleep(delays.get(provider, 2.0))
    return {"provider": provider, "response": f"Response from {provider}"}

# Gather runs all coroutines concurrently and returns results in order
async def multi_provider_analysis(prompt: str):
    results = await asyncio.gather(
        call_provider("anthropic", prompt),
        call_provider("openai", prompt),
        call_provider("cohere", prompt),
    )
    # results is a list: [anthropic_result, openai_result, cohere_result]
    # Order matches the order you passed them in
    return results
```

**Key properties of `asyncio.gather()`:**
- Starts all coroutines concurrently
- Returns results in the **same order** as the arguments (not completion order)
- Waits for **all** coroutines to complete before returning
- If one raises an exception, all results are lost by default

### GenAI Use Case: Parallel Embeddings

```python
async def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple text chunks in parallel"""
    async with httpx.AsyncClient() as client:
        tasks = [
            client.post(
                "https://api.openai.com/v1/embeddings",
                json={"input": chunk, "model": "text-embedding-3-small"},
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            for chunk in chunks
        ]
        responses = await asyncio.gather(*tasks)
        return [r.json()["data"][0]["embedding"] for r in responses]

# 10 chunks at 200ms each:
# Sequential: 2000ms
# Parallel:    200ms
```

---

## Racing Providers with asyncio.wait()

### The Problem: You Want the Fastest Response

Sometimes you don't need all results — you want whichever LLM provider responds first.

### asyncio.wait() with FIRST_COMPLETED

```python
import asyncio

async def call_anthropic(prompt: str) -> dict:
    await asyncio.sleep(3.0)  # Simulate Anthropic latency
    return {"provider": "anthropic", "response": "Claude's response"}

async def call_openai(prompt: str) -> dict:
    await asyncio.sleep(1.5)  # Simulate OpenAI latency
    return {"provider": "openai", "response": "GPT's response"}

async def fastest_response(prompt: str) -> dict:
    # Create tasks (not coroutines — tasks start immediately)
    tasks = {
        asyncio.create_task(call_anthropic(prompt), name="anthropic"),
        asyncio.create_task(call_openai(prompt), name="openai"),
    }

    # Wait for the FIRST one to complete
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    # Cancel the slower providers (don't waste resources)
    for task in pending:
        task.cancel()

    # Get the winner's result
    winner = done.pop()
    return winner.result()
```

```
Racing Providers:

  Anthropic: ██████████████████████████████ (3.0s)
  OpenAI:    ██████████████ (1.5s) ← WINNER
                            │
                            └─ Return immediately, cancel Anthropic

  User gets response in 1.5s instead of waiting for both (3.0s)
```

### gather() vs wait() Comparison

| Feature | `asyncio.gather()` | `asyncio.wait()` |
|---------|-------------------|------------------|
| **Returns** | List of results (ordered) | Sets of done/pending tasks |
| **Waits for** | All tasks to complete | Configurable (FIRST_COMPLETED, ALL_COMPLETED) |
| **Use case** | Need ALL results (parallel analysis) | Need FASTEST result (provider racing) |
| **Cancellation** | No built-in cancellation | You handle cancellation of pending tasks |
| **Error handling** | One error can lose all results | Each task's error is independent |

### Cancelling Pending Tasks

Always cancel tasks you no longer need — they consume resources otherwise:

```python
# ✅ CORRECT: Cancel pending tasks after getting the winner
done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
for task in pending:
    task.cancel()
    try:
        await task  # Wait for cancellation to propagate
    except asyncio.CancelledError:
        pass  # Expected

# ❌ WRONG: Leaving pending tasks running
done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
# pending tasks continue running in the background, wasting resources!
```

---

## Timeout Handling with asyncio.wait_for()

### The Problem: LLM Calls Can Hang

LLM providers occasionally take much longer than expected or hang indefinitely. Without timeouts, your users wait forever.

```
Without timeouts:
  User sends chat request
  Server calls Claude API
  Claude API is experiencing issues...
  ...
  ... (30 seconds pass)
  ...
  ... (60 seconds pass)
  ...
  User gives up and closes browser
  Server is STILL waiting for Claude
```

### asyncio.wait_for() Adds Timeouts

```python
import asyncio

async def call_llm(prompt: str) -> str:
    """Simulate an LLM call that might be slow"""
    await asyncio.sleep(15)  # Simulate a very slow response
    return "LLM response"

async def chat_with_timeout(prompt: str) -> dict:
    try:
        # Wait at most 10 seconds for the LLM response
        result = await asyncio.wait_for(
            call_llm(prompt),
            timeout=10.0
        )
        return {"response": result}

    except asyncio.TimeoutError:
        # LLM took too long — return a user-friendly error
        return {"error": "LLM response timed out. Please try again."}
```

```
With timeout (10s):

  LLM Call: ██████████████████████████████ (would take 15s)
                       │
            Timeout at 10s ← asyncio.TimeoutError raised
                       │
            Return error immediately, user isn't stuck
```

### Combining Timeouts with Provider Failover

```python
async def chat_with_failover(prompt: str) -> dict:
    """Try primary provider with timeout, fall back to secondary"""

    # Try Anthropic first (10s timeout)
    try:
        result = await asyncio.wait_for(
            call_anthropic(prompt),
            timeout=10.0
        )
        return {"provider": "anthropic", "response": result}
    except asyncio.TimeoutError:
        logger.warning("Anthropic timed out, falling back to OpenAI")
    except Exception as e:
        logger.warning(f"Anthropic error: {e}, falling back to OpenAI")

    # Fall back to OpenAI (10s timeout)
    try:
        result = await asyncio.wait_for(
            call_openai(prompt),
            timeout=10.0
        )
        return {"provider": "openai", "response": result}
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="All LLM providers timed out. Please try again later."
        )
```

### Timeout Best Practices for GenAI

```python
# Recommended timeouts by operation type
TIMEOUTS = {
    "chat_completion": 30.0,       # Long-form generation can take a while
    "short_completion": 10.0,      # Quick Q&A should be fast
    "embedding": 5.0,              # Embeddings are fast
    "health_check": 2.0,           # If health check is slow, something's wrong
    "provider_failover_total": 45.0,  # Total time for all provider attempts
}
```

```
★ Key Insight ─────────────────────────────────────────
Always add timeouts to LLM calls. A missing timeout is a
production incident waiting to happen. When a provider
has an outage, requests without timeouts pile up until
your server runs out of connections and crashes. 10-30
seconds is reasonable for chat completions; 5 seconds
for embeddings.
─────────────────────────────────────────────────────────
```

---

## Background Tasks for Non-Blocking Work

### The Problem: Post-Response Work Slows Down Users

After responding to a chat request, you might need to:
- Log token usage to the database
- Update analytics counters
- Send webhook notifications
- Cache the response

None of this should delay the user's response.

### FastAPI's BackgroundTasks

```python
from fastapi import BackgroundTasks

async def log_usage(request_id: str, model: str, tokens: int):
    """Log token usage — runs AFTER the response is sent"""
    # Simulate database write
    await asyncio.sleep(0.5)
    logger.info(f"Logged usage: request={request_id} model={model} tokens={tokens}")

async def update_analytics(user_id: str, model: str):
    """Update analytics counters — runs AFTER the response is sent"""
    await asyncio.sleep(0.3)
    logger.info(f"Analytics updated: user={user_id} model={model}")

@app.post("/chat")
async def chat(body: ChatRequest, background_tasks: BackgroundTasks):
    # Process the chat request
    response = await call_llm(body.messages)

    # Schedule work to run AFTER the response is sent
    background_tasks.add_task(log_usage, "req-123", body.model, 150)
    background_tasks.add_task(update_analytics, "user-456", body.model)

    # Response is sent immediately — background tasks run afterwards
    return {"response": response}
```

```
Request Timeline:

  Client Request ──→ Process Chat ──→ Response Sent ──→ Client receives response
                                           │
                                           └──→ Background: log_usage()
                                           └──→ Background: update_analytics()
                                                 (user doesn't wait for these)
```

### How BackgroundTasks Work

```
┌─────────────────────────────────────────────────────────┐
│                    Request Flow                          │
│                                                         │
│  1. Request arrives                                      │
│  2. Endpoint runs, calls LLM, gets response             │
│  3. Endpoint adds background tasks                       │
│  4. Response is sent to client ← User gets response HERE │
│  5. Background tasks execute (log, analytics, cache)     │
│  6. Background tasks complete (user doesn't know/care)   │
│                                                         │
│  User waits for: steps 1-4 only                         │
│  Server does:    steps 5-6 after user has their response │
└─────────────────────────────────────────────────────────┘
```

### When to Use Background Tasks vs Await

| Scenario | Use `await` | Use `BackgroundTasks` |
|----------|-------------|----------------------|
| LLM call (user needs result) | ✅ | |
| Log token usage | | ✅ |
| Update analytics | | ✅ |
| Save to database (user needs confirmation) | ✅ | |
| Save to database (audit log) | | ✅ |
| Send webhook notification | | ✅ |
| Cache the response | | ✅ |

**Rule of thumb**: If the user needs the result, `await` it. If it's bookkeeping, background it.

---

## Async HTTP Clients (httpx)

### Why httpx Instead of requests

The `requests` library is synchronous — it blocks the event loop. `httpx` is the async-compatible alternative.

```python
# ❌ WRONG: requests blocks the event loop
import requests

async def call_llm():
    response = requests.post(url, json=data)  # BLOCKS!
    return response.json()

# ✅ CORRECT: httpx is async-native
import httpx

async def call_llm():
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data)  # Non-blocking
        return response.json()
```

### Reusable Async Client Pattern

Creating a new client per request is wasteful. Use a shared client with connection pooling:

```python
from contextlib import asynccontextmanager
import httpx

# Application-level HTTP client (shared across all requests)
http_client: httpx.AsyncClient | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create shared HTTP client on startup, close on shutdown"""
    global http_client
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=5.0),  # 30s read, 5s connect
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    yield
    await http_client.aclose()

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def chat(body: ChatRequest):
    response = await http_client.post(
        "https://api.anthropic.com/v1/messages",
        json={"model": body.model, "messages": body.messages},
        headers={"X-API-Key": API_KEY, "anthropic-version": "2023-06-01"},
    )
    return response.json()
```

### httpx Timeout Configuration

```python
# Granular timeout control
client = httpx.AsyncClient(
    timeout=httpx.Timeout(
        connect=5.0,   # Time to establish connection
        read=30.0,     # Time to read response (LLM generation)
        write=10.0,    # Time to send request body
        pool=5.0,      # Time to acquire connection from pool
    )
)
```

---

## Error Handling in Concurrent Operations

### asyncio.gather() with return_exceptions

By default, `gather()` raises the first exception immediately, losing other results. Use `return_exceptions=True` to get all results (including exceptions):

```python
# ❌ DEFAULT: One failure kills everything
async def risky():
    try:
        results = await asyncio.gather(
            call_anthropic(prompt),     # Succeeds
            call_openai(prompt),        # Raises an error
            call_cohere(prompt),        # Never gets its result
        )
    except Exception:
        # We lost the successful Anthropic result!
        pass

# ✅ BETTER: return_exceptions=True
async def resilient():
    results = await asyncio.gather(
        call_anthropic(prompt),
        call_openai(prompt),
        call_cohere(prompt),
        return_exceptions=True,  # Exceptions become return values
    )

    successful = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"Provider {i} failed: {result}")
        else:
            successful.append(result)

    if not successful:
        raise HTTPException(500, "All providers failed")

    return successful[0]  # Return first successful result
```

### Task-Level Error Handling with asyncio.wait()

```python
async def resilient_race(prompt: str) -> dict:
    """Race providers, handle individual failures"""
    tasks = {
        asyncio.create_task(call_anthropic(prompt), name="anthropic"),
        asyncio.create_task(call_openai(prompt), name="openai"),
    }

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    # Cancel pending tasks
    for task in pending:
        task.cancel()

    # Check if the completed task succeeded
    winner = done.pop()
    if winner.exception():
        # First finisher failed — wait for the other
        if pending:
            remaining = pending.pop()
            try:
                result = await asyncio.wait_for(remaining, timeout=10.0)
                return result
            except (asyncio.TimeoutError, Exception):
                raise HTTPException(500, "All providers failed")
        raise HTTPException(500, f"Provider failed: {winner.exception()}")

    return winner.result()
```

```
★ Key Insight ─────────────────────────────────────────
Always use return_exceptions=True with asyncio.gather()
in production. Without it, one flaky provider can crash
your entire multi-provider call. With it, you gracefully
handle partial failures and return whatever succeeded.
─────────────────────────────────────────────────────────
```

---

## Key Insights

```
★ Insight ─────────────────────────────────────────────

1. Async is not parallel — it's concurrent. A single thread
   switches between tasks at await points. This is perfect for
   I/O-bound work (LLM calls) but doesn't help with CPU-bound
   work (computation). For CPU-bound tasks, use threads or
   processes via run_in_executor().

2. Never block the event loop. The #1 async mistake is calling
   synchronous libraries (requests, time.sleep, sqlite3) inside
   async functions. Every I/O call must use an async-compatible
   library. If you must use a sync library, use a regular def
   endpoint (FastAPI runs it in a thread pool).

3. asyncio.gather() is your daily driver. Most GenAI patterns
   involve "do multiple things at once" — embeddings for
   multiple chunks, sentiment + summary + keywords, or checking
   multiple caches. gather() handles all of these.

4. Always add timeouts to LLM calls. A provider outage without
   timeouts means requests queue up until your server crashes.
   asyncio.wait_for() with 10-30s timeout is essential for
   production stability.

5. Background tasks separate user-facing work from bookkeeping.
   Users get fast responses while logging, analytics, and
   caching happen afterwards. This is a simple but high-impact
   pattern for GenAI APIs.

───────────────────────────────────────────────────────────
```

---

## Quick Reference

### Async Patterns for GenAI

| Pattern | Tool | Use Case |
|---------|------|----------|
| Run tasks in parallel, get all results | `asyncio.gather()` | Parallel embeddings, multi-analysis |
| Run tasks in parallel, get fastest | `asyncio.wait(FIRST_COMPLETED)` | Provider racing/failover |
| Add timeout to any async call | `asyncio.wait_for(coro, timeout=N)` | Prevent hanging LLM calls |
| Post-response work | `BackgroundTasks` | Logging, analytics, caching |
| Async HTTP calls | `httpx.AsyncClient` | Calling LLM provider APIs |
| Sleep without blocking | `await asyncio.sleep(N)` | Simulating delays, backoff |

### Common Mistakes

| Mistake | Fix |
|---------|-----|
| `requests.post()` in `async def` | Use `httpx.AsyncClient` |
| `time.sleep()` in `async def` | Use `await asyncio.sleep()` |
| No timeout on LLM calls | Wrap with `asyncio.wait_for()` |
| `asyncio.gather()` without `return_exceptions` | Add `return_exceptions=True` |
| Not cancelling pending tasks after `wait()` | Always cancel and await pending tasks |
| Creating new `httpx.AsyncClient` per request | Share client via app lifespan |

### Recommended Timeouts

| Operation | Timeout |
|-----------|---------|
| Chat completion | 30s |
| Short completion | 10s |
| Embedding generation | 5s |
| Health check | 2s |
| Connection establishment | 5s |

### async vs sync Decision

| Scenario | Use |
|----------|-----|
| Async I/O library available (httpx, asyncpg) | `async def` + `await` |
| Only sync library available (requests, sqlite3) | Regular `def` |
| No I/O, simple computation | Regular `def` |
| CPU-heavy work | Regular `def` (thread pool) |

---

## Next Steps

**After completing this topic:**

1. **Practice the hands-on exercises** in `05_Async_FastAPI_Practice.md`
   - Exercise 1: Convert sync to async, observe concurrency
   - Exercise 2: Parallel LLM calls with `asyncio.gather()`
   - Exercise 3: Timeout handling with `asyncio.wait_for()`
   - Exercise 4: Background tasks for non-blocking logging
   - Exercise 5: Multi-provider failover with `asyncio.wait()`

2. **Integrate with your middleware stack**
   - Your middleware from Topic 4 already uses `async def` — now you understand why
   - Timing middleware becomes more important when LLM calls take seconds

3. **Move to Topic 6: Database Integration**
   - Async SQLAlchemy for non-blocking database queries
   - Connection pooling with async
   - Combining async database and LLM operations

**Key Takeaway**: Async is the foundation of scalable GenAI backends. Without it, one slow LLM call blocks all other users. With it, your server efficiently handles hundreds of concurrent LLM requests on a single worker. Every production GenAI API should use async endpoints with async HTTP clients.

---

*Async isn't optional for GenAI — it's essential. Your users shouldn't wait in line for the event loop.*
