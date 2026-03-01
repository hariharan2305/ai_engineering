# Async FastAPI Hands-On Practice: Build & Test

This guide takes you through **practical implementation** of async patterns critical for GenAI backends. Each exercise builds on realistic LLM-call scenarios using simulated latency (`asyncio.sleep`) so you can see concurrency in action without needing API keys.

Each exercise builds in complexity. Follow them in order.

---

## Exercise 1: Convert Sync to Async — Observe Concurrency (25 minutes)

### Goal

Build two versions of the same endpoint — sync and async — and observe how the async version handles concurrent requests while the sync version blocks.

### Steps

**1. Create `projects/fastapi_concepts_hands_on/09_1_basic_async_endpoint.py`:**

```python
"""
Exercise 1: Sync vs Async Endpoints

This script demonstrates the fundamental difference between sync and async:
- A sync endpoint that blocks during simulated LLM calls
- An async endpoint that yields control during waits
- How to observe concurrency by sending multiple requests

GenAI Context:
When your chat endpoint calls an LLM provider, the call takes 1-30 seconds.
A sync endpoint blocks the entire worker during that wait. An async endpoint
yields control so other requests can be processed simultaneously.

Concepts covered:
- async def vs def endpoint declarations
- await asyncio.sleep() vs time.sleep()
- Observing concurrent request handling
- Why async matters when LLM calls take seconds

Run: uv run uvicorn 09_1_basic_async_endpoint:app --reload --port 8000
"""

import time
import asyncio
import logging

from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Sync vs Async Comparison")


# ===== SIMULATED LLM CALL (ASYNC) =====

async def simulate_llm_call(model: str, delay: float = 2.0) -> dict:
    """
    Simulates an async LLM API call.
    Uses asyncio.sleep — yields control to the event loop during the wait.
    """
    logger.info(f"LLM call START | model={model} (will take {delay}s)")
    await asyncio.sleep(delay)  # Non-blocking: event loop is free
    logger.info(f"LLM call END   | model={model}")
    return {
        "model": model,
        "response": f"Simulated response from {model}",
        "latency": f"{delay}s",
    }


# ===== SIMULATED LLM CALL (SYNC) =====

def simulate_llm_call_sync(model: str, delay: float = 2.0) -> dict:
    """
    Simulates a SYNC LLM API call.
    Uses time.sleep — BLOCKS the thread during the wait.
    """
    logger.info(f"LLM call START (sync) | model={model} (will take {delay}s)")
    time.sleep(delay)  # Blocking: nothing else runs
    logger.info(f"LLM call END (sync)   | model={model}")
    return {
        "model": model,
        "response": f"Simulated sync response from {model}",
        "latency": f"{delay}s",
    }


# ===== ENDPOINTS =====

@app.get("/health")
def health():
    """Quick health check"""
    return {"status": "healthy"}


@app.post("/chat/async")
async def chat_async(model: str = "claude-3", delay: float = 2.0):
    """
    ASYNC chat endpoint.

    Multiple concurrent requests to this endpoint will run in parallel.
    Total time for N requests ≈ max(individual delays), not sum.
    """
    start = time.time()
    result = await simulate_llm_call(model, delay)
    elapsed = time.time() - start

    return {
        **result,
        "endpoint": "async",
        "elapsed": f"{elapsed:.3f}s",
    }


@app.post("/chat/sync")
def chat_sync(model: str = "gpt-4", delay: float = 2.0):
    """
    SYNC chat endpoint.

    FastAPI runs this in a thread pool. With default settings,
    concurrent requests are limited by the thread pool size.
    Requests queue up when all threads are busy.
    """
    start = time.time()
    result = simulate_llm_call_sync(model, delay)
    elapsed = time.time() - start

    return {
        **result,
        "endpoint": "sync",
        "elapsed": f"{elapsed:.3f}s",
    }
```

**2. Run the server:**

```bash
cd FastAPI/projects/fastapi_concepts_hands_on
uv run uvicorn 09_1_basic_async_endpoint:app --reload --port 8000
```

### Test It

```bash
# ── Test 1: Single async request ──
curl -X POST "http://localhost:8000/chat/async?model=claude-3&delay=2"
# Should take ~2 seconds

# ── Test 2: Single sync request ──
curl -X POST "http://localhost:8000/chat/sync?model=gpt-4&delay=2"
# Should take ~2 seconds

# ── Test 3: Concurrent async requests (THE KEY TEST) ──
# Open 3 terminals and run these simultaneously:
# Terminal 1:
curl -w "\nTotal: %{time_total}s\n" -X POST "http://localhost:8000/chat/async?model=claude-3&delay=3"
# Terminal 2:
curl -w "\nTotal: %{time_total}s\n" -X POST "http://localhost:8000/chat/async?model=gpt-4&delay=3"
# Terminal 3:
curl -w "\nTotal: %{time_total}s\n" -X POST "http://localhost:8000/chat/async?model=cohere&delay=3"

# ALL THREE should complete in ~3 seconds total (not 9)!

# ── Test 4: Quick concurrency test with background curl ──
time (
  curl -s -X POST "http://localhost:8000/chat/async?model=claude-3&delay=2" &
  curl -s -X POST "http://localhost:8000/chat/async?model=gpt-4&delay=2" &
  curl -s -X POST "http://localhost:8000/chat/async?model=cohere&delay=2" &
  wait
)
# Should print ~2 seconds total wall-clock time
```

### What You Should See

```bash
# Test 1: Single request
{"model":"claude-3","response":"Simulated response from claude-3","latency":"2.0s","endpoint":"async","elapsed":"2.002s"}

# Test 4: Three concurrent requests — server logs show overlap:
2024-01-15 10:30:00 | LLM call START | model=claude-3 (will take 2.0s)
2024-01-15 10:30:00 | LLM call START | model=gpt-4 (will take 2.0s)
2024-01-15 10:30:00 | LLM call START | model=cohere (will take 2.0s)
2024-01-15 10:30:02 | LLM call END   | model=claude-3
2024-01-15 10:30:02 | LLM call END   | model=gpt-4
2024-01-15 10:30:02 | LLM call END   | model=cohere

# Notice: All 3 START at the same time, all 3 END at the same time.
# They ran concurrently, not sequentially.
```

### Key Takeaway

The async endpoint handles multiple concurrent requests without blocking. When one request awaits an LLM response, the event loop picks up the next request. This is why async is essential for GenAI backends — your users shouldn't have to wait in line behind each other for LLM responses.

---

## Exercise 2: Parallel LLM Calls with asyncio.gather() (30 minutes)

### Goal

Use `asyncio.gather()` to call multiple LLM providers in parallel within a single request. This is the pattern for multi-analysis endpoints (sentiment + summary + keywords) and consensus queries.

### Steps

**1. Create `projects/fastapi_concepts_hands_on/09_2_parallel_provider_calls.py`:**

```python
"""
Exercise 2: Parallel LLM Provider Calls with asyncio.gather()

This script demonstrates running multiple async operations concurrently:
- asyncio.gather() for parallel execution
- Collecting results from multiple providers
- return_exceptions=True for resilient multi-provider calls
- Measuring sequential vs parallel execution time

GenAI Context:
You often need to call multiple LLM providers in parallel:
- Get sentiment + summary + keywords from a single text
- Query multiple providers and compare results
- Generate embeddings for multiple chunks simultaneously

Concepts covered:
- asyncio.gather() for parallel coroutines
- return_exceptions=True for partial failure handling
- Comparing sequential vs parallel execution time
- Handling mixed success/failure results

Run: uv run uvicorn 09_2_parallel_provider_calls:app --reload --port 8000
"""

import time
import asyncio
import random
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Parallel LLM Calls with asyncio.gather()")


# ===== SIMULATED LLM PROVIDERS =====

PROVIDERS = {
    "anthropic": {"name": "Claude 3", "base_latency": 2.0},
    "openai": {"name": "GPT-4", "base_latency": 1.5},
    "cohere": {"name": "Command R+", "base_latency": 2.5},
}


async def call_provider(provider: str, prompt: str, fail_chance: float = 0.0) -> dict:
    """
    Simulate calling an LLM provider with realistic latency.

    Args:
        provider: Provider name (anthropic, openai, cohere)
        prompt: The user's prompt
        fail_chance: Probability of simulated failure (0.0 to 1.0)
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")

    config = PROVIDERS[provider]
    # Add some random variation to latency (±0.5s)
    latency = config["base_latency"] + random.uniform(-0.5, 0.5)

    logger.info(f"  → Calling {provider} ({config['name']})... ({latency:.1f}s)")

    # Simulate random failures
    if random.random() < fail_chance:
        await asyncio.sleep(latency * 0.3)  # Fail fast
        raise ConnectionError(f"{provider} is currently unavailable")

    await asyncio.sleep(latency)

    return {
        "provider": provider,
        "model": config["name"],
        "response": f"Response from {config['name']}: Analysis of '{prompt[:50]}...'",
        "latency": f"{latency:.1f}s",
    }


# ===== DATA MODELS =====

class AnalysisRequest(BaseModel):
    text: str
    providers: list[str] = ["anthropic", "openai"]


# ===== ENDPOINTS =====

@app.get("/health")
def health():
    return {"status": "healthy", "providers": list(PROVIDERS.keys())}


@app.post("/analyze/sequential")
async def analyze_sequential(body: AnalysisRequest):
    """
    Call providers SEQUENTIALLY (slow).
    Total time = sum of all provider latencies.
    """
    start = time.time()
    results = []

    logger.info(f"Sequential analysis: {body.providers}")

    for provider in body.providers:
        try:
            result = await call_provider(provider, body.text)
            results.append(result)
        except Exception as e:
            results.append({"provider": provider, "error": str(e)})

    elapsed = time.time() - start

    return {
        "mode": "sequential",
        "total_time": f"{elapsed:.3f}s",
        "results": results,
        "provider_count": len(body.providers),
    }


@app.post("/analyze/parallel")
async def analyze_parallel(body: AnalysisRequest):
    """
    Call providers IN PARALLEL (fast).
    Total time = max of all provider latencies.
    Uses asyncio.gather() with return_exceptions=True for resilience.
    """
    start = time.time()

    logger.info(f"Parallel analysis: {body.providers}")

    # Create coroutines for all providers
    tasks = [
        call_provider(provider, body.text)
        for provider in body.providers
    ]

    # Run all in parallel — return_exceptions prevents one failure from losing all results
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results: separate successes from failures
    results = []
    for i, result in enumerate(raw_results):
        if isinstance(result, Exception):
            results.append({
                "provider": body.providers[i],
                "error": str(result),
                "status": "failed",
            })
        else:
            results.append({**result, "status": "success"})

    elapsed = time.time() - start

    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]

    return {
        "mode": "parallel",
        "total_time": f"{elapsed:.3f}s",
        "results": results,
        "summary": {
            "total": len(results),
            "successful": len(successful),
            "failed": len(failed),
        },
    }


@app.post("/analyze/parallel-risky")
async def analyze_parallel_risky(body: AnalysisRequest):
    """
    Call providers in parallel WITH simulated failures.
    Demonstrates how return_exceptions=True handles partial failures.
    Each provider has a 30% chance of failing.
    """
    start = time.time()

    logger.info(f"Parallel (risky) analysis: {body.providers}")

    tasks = [
        call_provider(provider, body.text, fail_chance=0.3)
        for provider in body.providers
    ]

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for i, result in enumerate(raw_results):
        if isinstance(result, Exception):
            results.append({
                "provider": body.providers[i],
                "error": str(result),
                "status": "failed",
            })
            logger.warning(f"  ✗ {body.providers[i]} failed: {result}")
        else:
            results.append({**result, "status": "success"})
            logger.info(f"  ✓ {result['provider']} succeeded")

    elapsed = time.time() - start

    successful = [r for r in results if r.get("status") == "success"]

    if not successful:
        raise HTTPException(
            status_code=503,
            detail="All LLM providers failed. Please try again.",
        )

    return {
        "mode": "parallel-risky",
        "total_time": f"{elapsed:.3f}s",
        "results": results,
        "summary": {
            "total": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
        },
    }
```

**2. Run the server:**

```bash
cd FastAPI/projects/fastapi_concepts_hands_on
uv run uvicorn 09_2_parallel_provider_calls:app --reload --port 8000
```

### Test It

```bash
# ── Test 1: Sequential (slow) — 3 providers ──
curl -X POST http://localhost:8000/analyze/sequential \
  -H "Content-Type: application/json" \
  -d '{"text": "Async programming is essential for GenAI backends", "providers": ["anthropic", "openai", "cohere"]}'
# Total time: ~6 seconds (2 + 1.5 + 2.5)

# ── Test 2: Parallel (fast) — same 3 providers ──
curl -X POST http://localhost:8000/analyze/parallel \
  -H "Content-Type: application/json" \
  -d '{"text": "Async programming is essential for GenAI backends", "providers": ["anthropic", "openai", "cohere"]}'
# Total time: ~2.5 seconds (max of 2, 1.5, 2.5) — 2-3x faster!

# ── Test 3: Parallel with failures — run multiple times ──
curl -X POST http://localhost:8000/analyze/parallel-risky \
  -H "Content-Type: application/json" \
  -d '{"text": "Test resilience", "providers": ["anthropic", "openai", "cohere"]}'
# Some runs: all succeed. Some runs: 1-2 fail but you still get results.
# Rare: all fail → 503 error.

# ── Test 4: Compare timing directly ──
echo "=== Sequential ===" && \
curl -s -w "\nWall time: %{time_total}s\n" -X POST http://localhost:8000/analyze/sequential \
  -H "Content-Type: application/json" \
  -d '{"text": "Compare timing", "providers": ["anthropic", "openai", "cohere"]}' && \
echo "\n=== Parallel ===" && \
curl -s -w "\nWall time: %{time_total}s\n" -X POST http://localhost:8000/analyze/parallel \
  -H "Content-Type: application/json" \
  -d '{"text": "Compare timing", "providers": ["anthropic", "openai", "cohere"]}'
```

### What You Should See

```json
// Test 1: Sequential
{
  "mode": "sequential",
  "total_time": "6.123s",
  "results": [
    {"provider": "anthropic", "model": "Claude 3", "latency": "2.1s"},
    {"provider": "openai", "model": "GPT-4", "latency": "1.6s"},
    {"provider": "cohere", "model": "Command R+", "latency": "2.4s"}
  ],
  "provider_count": 3
}

// Test 2: Parallel — same results, ~2.5x faster
{
  "mode": "parallel",
  "total_time": "2.534s",
  "results": [
    {"provider": "anthropic", "model": "Claude 3", "status": "success"},
    {"provider": "openai", "model": "GPT-4", "status": "success"},
    {"provider": "cohere", "model": "Command R+", "status": "success"}
  ],
  "summary": {"total": 3, "successful": 3, "failed": 0}
}
```

### Key Takeaway

`asyncio.gather()` turns sequential LLM calls into parallel ones, reducing total latency from the sum of all calls to the max of all calls. With `return_exceptions=True`, one flaky provider doesn't crash the entire request — you get partial results and can handle failures gracefully. This is the workhorse pattern for any multi-provider or multi-analysis GenAI endpoint.

---

## Exercise 3: Timeout Handling for LLM Requests (30 minutes)

### Goal

Implement timeout handling with `asyncio.wait_for()` to prevent LLM calls from hanging indefinitely. Add provider failover when the primary provider times out.

### Steps

**1. Create `projects/fastapi_concepts_hands_on/09_3_timeout_handling.py`:**

```python
"""
Exercise 3: Timeout Handling for LLM Requests

This script demonstrates preventing hanging requests with timeouts:
- asyncio.wait_for() to add timeouts to any async call
- Catching asyncio.TimeoutError for user-friendly responses
- Provider failover when the primary times out
- Configurable timeouts per operation type

GenAI Context:
LLM providers occasionally respond very slowly or hang entirely.
Without timeouts, your users wait indefinitely and your server
accumulates stuck connections until it crashes. Timeouts ensure
predictable response times even when providers misbehave.

Concepts covered:
- asyncio.wait_for(coroutine, timeout=N)
- asyncio.TimeoutError handling
- Sequential failover with timeouts
- Per-operation timeout configuration

Run: uv run uvicorn 09_3_timeout_handling:app --reload --port 8000
"""

import time
import asyncio
import random
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Timeout Handling for LLM Calls")


# ===== TIMEOUT CONFIGURATION =====

TIMEOUTS = {
    "chat": 10.0,       # Chat completion: 10 second max
    "embedding": 5.0,    # Embedding generation: 5 second max
    "health": 2.0,       # Health check: 2 second max
}


# ===== SIMULATED LLM PROVIDERS =====

async def call_anthropic(prompt: str) -> dict:
    """Simulate Anthropic API — sometimes slow"""
    latency = random.choice([1.5, 2.0, 3.0, 8.0, 15.0])  # Occasionally very slow
    logger.info(f"  Anthropic: starting (will take {latency}s)")
    await asyncio.sleep(latency)
    return {
        "provider": "anthropic",
        "model": "Claude 3",
        "response": f"Claude's analysis of: {prompt[:50]}",
        "latency": f"{latency:.1f}s",
    }


async def call_openai(prompt: str) -> dict:
    """Simulate OpenAI API — generally faster but can fail"""
    latency = random.choice([1.0, 1.5, 2.0, 5.0, 12.0])  # Usually fast
    logger.info(f"  OpenAI: starting (will take {latency}s)")
    await asyncio.sleep(latency)
    return {
        "provider": "openai",
        "model": "GPT-4",
        "response": f"GPT-4's analysis of: {prompt[:50]}",
        "latency": f"{latency:.1f}s",
    }


async def call_cohere(prompt: str) -> dict:
    """Simulate Cohere API — backup provider"""
    latency = random.choice([1.0, 2.0, 3.0])  # Reliable
    logger.info(f"  Cohere: starting (will take {latency}s)")
    await asyncio.sleep(latency)
    return {
        "provider": "cohere",
        "model": "Command R+",
        "response": f"Command R+'s analysis of: {prompt[:50]}",
        "latency": f"{latency:.1f}s",
    }


# ===== DATA MODELS =====

class ChatRequest(BaseModel):
    prompt: str
    timeout: float | None = None  # Optional client-specified timeout


# ===== ENDPOINTS =====

@app.get("/health")
def health():
    return {"status": "healthy", "default_timeouts": TIMEOUTS}


@app.post("/chat/basic-timeout")
async def chat_basic_timeout(body: ChatRequest):
    """
    Basic timeout example.
    Wraps a single LLM call with asyncio.wait_for().
    """
    timeout = body.timeout or TIMEOUTS["chat"]
    start = time.time()

    logger.info(f"Chat request (timeout={timeout}s): {body.prompt[:50]}")

    try:
        result = await asyncio.wait_for(
            call_anthropic(body.prompt),
            timeout=timeout,
        )
        elapsed = time.time() - start

        return {
            **result,
            "timeout_setting": f"{timeout}s",
            "elapsed": f"{elapsed:.3f}s",
            "timed_out": False,
        }

    except asyncio.TimeoutError:
        elapsed = time.time() - start
        logger.warning(
            f"  ✗ Anthropic TIMED OUT after {elapsed:.1f}s (limit: {timeout}s)"
        )
        raise HTTPException(
            status_code=504,
            detail={
                "error": "llm_timeout",
                "message": f"LLM provider did not respond within {timeout}s. Please try again.",
                "timeout": timeout,
                "elapsed": f"{elapsed:.3f}s",
            },
        )


@app.post("/chat/failover")
async def chat_with_failover(body: ChatRequest):
    """
    Timeout with failover.
    Tries providers in order: Anthropic → OpenAI → Cohere.
    Each provider gets its own timeout.
    """
    timeout = body.timeout or TIMEOUTS["chat"]
    start = time.time()

    # Provider chain: try each in order
    providers = [
        ("anthropic", call_anthropic),
        ("openai", call_openai),
        ("cohere", call_cohere),
    ]

    errors = []

    for name, call_func in providers:
        try:
            logger.info(f"  Trying {name} (timeout={timeout}s)...")
            result = await asyncio.wait_for(
                call_func(body.prompt),
                timeout=timeout,
            )

            elapsed = time.time() - start
            return {
                **result,
                "failover_chain": [e["provider"] for e in errors] + [name],
                "attempts": len(errors) + 1,
                "total_elapsed": f"{elapsed:.3f}s",
                "timed_out": False,
            }

        except asyncio.TimeoutError:
            elapsed = time.time() - start
            logger.warning(f"  ✗ {name} timed out after {timeout}s")
            errors.append({
                "provider": name,
                "error": "timeout",
                "elapsed": f"{elapsed:.3f}s",
            })

        except Exception as e:
            elapsed = time.time() - start
            logger.warning(f"  ✗ {name} failed: {e}")
            errors.append({
                "provider": name,
                "error": str(e),
                "elapsed": f"{elapsed:.3f}s",
            })

    # All providers failed
    total_elapsed = time.time() - start
    logger.error(f"  ✗ ALL PROVIDERS FAILED after {total_elapsed:.1f}s")

    raise HTTPException(
        status_code=503,
        detail={
            "error": "all_providers_failed",
            "message": "All LLM providers failed or timed out. Please try again later.",
            "attempts": errors,
            "total_elapsed": f"{total_elapsed:.3f}s",
        },
    )


@app.post("/chat/fast-timeout")
async def chat_fast_timeout(body: ChatRequest):
    """
    Short timeout (3s) — demonstrates aggressive timeout for quick responses.
    Good for use cases where you'd rather show an error than wait.
    """
    timeout = 3.0  # Very aggressive timeout
    start = time.time()

    logger.info(f"Fast chat (timeout={timeout}s): {body.prompt[:50]}")

    try:
        result = await asyncio.wait_for(
            call_anthropic(body.prompt),
            timeout=timeout,
        )
        elapsed = time.time() - start
        return {**result, "timeout": f"{timeout}s", "elapsed": f"{elapsed:.3f}s"}

    except asyncio.TimeoutError:
        elapsed = time.time() - start
        logger.warning(f"  ✗ Fast timeout hit at {elapsed:.1f}s")
        raise HTTPException(
            status_code=504,
            detail=f"Response not available within {timeout}s. Try a simpler prompt.",
        )
```

**2. Run the server:**

```bash
cd FastAPI/projects/fastapi_concepts_hands_on
uv run uvicorn 09_3_timeout_handling:app --reload --port 8000
```

### Test It

```bash
# ── Test 1: Basic timeout — run multiple times, some will timeout ──
curl -X POST http://localhost:8000/chat/basic-timeout \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain async programming"}'
# Run 3-4 times: some succeed (fast latency), some timeout (slow latency)

# ── Test 2: Custom short timeout ──
curl -X POST http://localhost:8000/chat/basic-timeout \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Quick question", "timeout": 2.0}'
# Very likely to timeout — only the fastest responses (1.5s) make it

# ── Test 3: Failover — provider chain ──
curl -X POST http://localhost:8000/chat/failover \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Analyze this text", "timeout": 4.0}'
# If Anthropic times out, falls back to OpenAI, then Cohere.
# Response shows which providers were tried.

# ── Test 4: Fast timeout (3s aggressive) ──
curl -X POST http://localhost:8000/chat/fast-timeout \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Quick response needed"}'
# Only succeeds when Anthropic is fast (1.5-2s). Frequent timeouts.

# ── Test 5: Generous timeout (should always succeed) ──
curl -X POST http://localhost:8000/chat/basic-timeout \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Take your time", "timeout": 30.0}'
# Even the slowest simulated response (15s) completes within 30s
```

### What You Should See

```json
// Test 1 — Success (fast response):
{
  "provider": "anthropic",
  "model": "Claude 3",
  "response": "Claude's analysis of: Explain async programming",
  "latency": "2.0s",
  "timeout_setting": "10.0s",
  "elapsed": "2.003s",
  "timed_out": false
}

// Test 1 — Timeout (slow response):
{
  "detail": {
    "error": "llm_timeout",
    "message": "LLM provider did not respond within 10.0s. Please try again.",
    "timeout": 10.0,
    "elapsed": "10.002s"
  }
}

// Test 3 — Failover (Anthropic timed out, OpenAI succeeded):
{
  "provider": "openai",
  "model": "GPT-4",
  "failover_chain": ["anthropic", "openai"],
  "attempts": 2,
  "total_elapsed": "5.503s",
  "timed_out": false
}
```

### Key Takeaway

`asyncio.wait_for()` is your safety net for LLM calls. Without timeouts, a provider outage cascades into server-wide failure as connections pile up. With timeouts plus failover, your API stays responsive even when providers misbehave. Always set timeouts — the only question is how long to wait.

---

## Exercise 4: Background Tasks for Non-Blocking Logging (25 minutes)

### Goal

Use FastAPI's `BackgroundTasks` to perform post-response work (logging token usage, updating analytics, caching) without delaying the user's response.

### Steps

**1. Create `projects/fastapi_concepts_hands_on/09_4_background_tasks.py`:**

```python
"""
Exercise 4: Background Tasks for Non-Blocking Logging

This script demonstrates FastAPI's BackgroundTasks for post-response work:
- Logging token usage after the response is sent
- Updating analytics counters in the background
- Caching responses for future requests
- Comparing response times with and without background tasks

GenAI Context:
After processing a chat request, you need to:
- Log token usage to the database
- Update per-user analytics (requests, tokens, cost)
- Cache the response for identical future queries
- Send webhook notifications

None of this should delay the user's response. Background tasks
run AFTER the response is sent.

Concepts covered:
- FastAPI BackgroundTasks injection
- background_tasks.add_task() for scheduling work
- async vs sync background task functions
- Measuring the impact on response time

Run: uv run uvicorn 09_4_background_tasks:app --reload --port 8000
"""

import time
import asyncio
import logging
from datetime import datetime

from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Background Tasks for GenAI Logging")


# ===== IN-MEMORY STORAGE (simulates database) =====

usage_log: list[dict] = []
analytics: dict = {"total_requests": 0, "total_tokens": 0, "total_cost": 0.0}
response_cache: dict[str, dict] = {}


# ===== BACKGROUND TASK FUNCTIONS =====

async def log_token_usage(
    request_id: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
):
    """
    Log token usage — runs AFTER the response is sent.
    In production, this would write to a database.
    """
    # Simulate database write latency
    await asyncio.sleep(0.5)

    entry = {
        "request_id": request_id,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "timestamp": datetime.now().isoformat(),
    }
    usage_log.append(entry)

    logger.info(
        f"  [BG] Token usage logged: {input_tokens}+{output_tokens}="
        f"{input_tokens + output_tokens} tokens ({model})"
    )


async def update_analytics(model: str, tokens: int, cost: float):
    """
    Update analytics counters — runs AFTER the response is sent.
    In production, this would update Redis or a database.
    """
    await asyncio.sleep(0.3)

    analytics["total_requests"] += 1
    analytics["total_tokens"] += tokens
    analytics["total_cost"] += cost

    logger.info(
        f"  [BG] Analytics updated: requests={analytics['total_requests']} "
        f"tokens={analytics['total_tokens']} cost=${analytics['total_cost']:.4f}"
    )


async def cache_response(cache_key: str, response_data: dict):
    """
    Cache the response — runs AFTER the response is sent.
    In production, this would write to Redis.
    """
    await asyncio.sleep(0.2)

    response_cache[cache_key] = {
        "data": response_data,
        "cached_at": datetime.now().isoformat(),
    }

    logger.info(f"  [BG] Response cached: key={cache_key}")


# ===== SIMULATED LLM CALL =====

async def simulate_llm_call(model: str, prompt: str) -> dict:
    """Simulate an LLM call with realistic latency"""
    await asyncio.sleep(1.5)  # 1.5s LLM latency

    input_tokens = len(prompt.split()) * 2  # Rough estimate
    output_tokens = 50  # Simulated output

    return {
        "response": f"Here is my analysis of: {prompt[:50]}",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


# ===== COST CALCULATION =====

COST_PER_1K_TOKENS = {
    "claude-3": 0.015,
    "gpt-4": 0.03,
}


# ===== DATA MODELS =====

class ChatRequest(BaseModel):
    model: str = "claude-3"
    prompt: str


# ===== ENDPOINTS =====

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat")
async def chat(body: ChatRequest, background_tasks: BackgroundTasks):
    """
    Chat endpoint with background tasks.

    The user gets the response immediately. Token logging,
    analytics, and caching happen in the background.
    """
    start = time.time()
    request_id = f"req-{int(time.time() * 1000)}"

    logger.info(f"Chat request: model={body.model} prompt='{body.prompt[:50]}'")

    # Check cache first
    cache_key = f"{body.model}:{body.prompt}"
    if cache_key in response_cache:
        logger.info(f"  Cache HIT for key={cache_key[:30]}")
        cached = response_cache[cache_key]
        return {
            **cached["data"],
            "cached": True,
            "cached_at": cached["cached_at"],
            "elapsed": f"{time.time() - start:.3f}s",
        }

    # Call LLM
    result = await simulate_llm_call(body.model, body.prompt)
    elapsed = time.time() - start

    # Calculate cost
    total_tokens = result["input_tokens"] + result["output_tokens"]
    cost_per_1k = COST_PER_1K_TOKENS.get(body.model, 0.01)
    cost = (total_tokens / 1000) * cost_per_1k

    # Schedule background tasks (these run AFTER the response is sent)
    background_tasks.add_task(
        log_token_usage,
        request_id,
        body.model,
        result["input_tokens"],
        result["output_tokens"],
    )
    background_tasks.add_task(
        update_analytics,
        body.model,
        total_tokens,
        cost,
    )
    background_tasks.add_task(
        cache_response,
        cache_key,
        result,
    )

    logger.info(f"  Response sent in {elapsed:.3f}s (background tasks scheduled)")

    # Response is sent NOW — background tasks haven't started yet
    return {
        **result,
        "request_id": request_id,
        "cost": f"${cost:.6f}",
        "elapsed": f"{elapsed:.3f}s",
        "cached": False,
        "background_tasks_scheduled": ["log_usage", "analytics", "cache"],
    }


@app.post("/chat/no-background")
async def chat_no_background(body: ChatRequest):
    """
    Chat endpoint WITHOUT background tasks.
    Logging, analytics, and caching happen BEFORE the response.
    Compare response time with /chat.
    """
    start = time.time()
    request_id = f"req-{int(time.time() * 1000)}"

    result = await simulate_llm_call(body.model, body.prompt)

    total_tokens = result["input_tokens"] + result["output_tokens"]
    cost_per_1k = COST_PER_1K_TOKENS.get(body.model, 0.01)
    cost = (total_tokens / 1000) * cost_per_1k

    # Do everything inline (BEFORE sending response)
    await log_token_usage(request_id, body.model, result["input_tokens"], result["output_tokens"])
    await update_analytics(body.model, total_tokens, cost)

    cache_key = f"{body.model}:{body.prompt}"
    await cache_response(cache_key, result)

    elapsed = time.time() - start

    return {
        **result,
        "request_id": request_id,
        "cost": f"${cost:.6f}",
        "elapsed": f"{elapsed:.3f}s",
        "note": "Logging/analytics/caching done BEFORE response (slower)",
    }


@app.get("/analytics")
def get_analytics():
    """View current analytics (populated by background tasks)"""
    return {
        "analytics": analytics,
        "usage_log_count": len(usage_log),
        "cache_entries": len(response_cache),
        "recent_usage": usage_log[-5:] if usage_log else [],
    }
```

**2. Run the server:**

```bash
cd FastAPI/projects/fastapi_concepts_hands_on
uv run uvicorn 09_4_background_tasks:app --reload --port 8000
```

### Test It

```bash
# ── Test 1: Chat WITH background tasks (fast response) ──
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3", "prompt": "Explain async in Python"}'
# elapsed: ~1.5s (just the LLM call)

# ── Test 2: Chat WITHOUT background tasks (slower response) ──
curl -X POST http://localhost:8000/chat/no-background \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3", "prompt": "Explain sync vs async"}'
# elapsed: ~2.5s (LLM call + logging + analytics + caching)

# ── Test 3: Check analytics (populated by background tasks) ──
# Wait 2 seconds for background tasks to complete, then:
curl http://localhost:8000/analytics
# Shows accumulated usage data from previous requests

# ── Test 4: Cache hit (same prompt as Test 1) ──
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3", "prompt": "Explain async in Python"}'
# elapsed: ~0.001s (cached — no LLM call needed!)

# ── Test 5: Multiple requests, then check analytics ──
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4", "prompt": "What is RAG?"}'

sleep 2  # Wait for background tasks

curl http://localhost:8000/analytics
# Shows total_requests, total_tokens, total_cost accumulated
```

### What You Should See

```json
// Test 1: Fast response (background tasks run after)
{
  "response": "Here is my analysis of: Explain async in Python",
  "model": "claude-3",
  "input_tokens": 8,
  "output_tokens": 50,
  "request_id": "req-1705312200000",
  "cost": "$0.000870",
  "elapsed": "1.503s",
  "cached": false,
  "background_tasks_scheduled": ["log_usage", "analytics", "cache"]
}

// Test 2: Slower response (everything inline)
{
  "elapsed": "2.505s",
  "note": "Logging/analytics/caching done BEFORE response (slower)"
}

// Test 4: Cache hit (instant)
{
  "cached": true,
  "cached_at": "2024-01-15T10:30:05",
  "elapsed": "0.001s"
}

// Server logs show background tasks running AFTER response:
10:30:03 | Chat request: model=claude-3 prompt='Explain async in Python'
10:30:04 |   Response sent in 1.503s (background tasks scheduled)
10:30:05 |   [BG] Token usage logged: 8+50=58 tokens (claude-3)
10:30:05 |   [BG] Analytics updated: requests=1 tokens=58 cost=$0.0009
10:30:05 |   [BG] Response cached: key=claude-3:Explain async in...
```

### Key Takeaway

Background tasks let you separate user-facing work from bookkeeping. The user gets their LLM response in 1.5s instead of 2.5s because logging, analytics, and caching happen after the response is sent. In a production GenAI app, this means faster perceived response times while still capturing all the operational data you need.

---

## Exercise 5: Multi-Provider Failover with asyncio.wait() (35 minutes)

### Goal

Build a provider-racing endpoint that sends the same prompt to multiple LLM providers simultaneously and returns whichever responds first. This maximizes responsiveness and provides automatic failover.

### Steps

**1. Create `projects/fastapi_concepts_hands_on/09_5_multi_provider_failover.py`:**

```python
"""
Exercise 5: Multi-Provider Failover with asyncio.wait()

This script demonstrates racing multiple providers:
- asyncio.create_task() to start providers concurrently
- asyncio.wait(FIRST_COMPLETED) to get the fastest response
- Cancelling slower providers after getting a winner
- Handling the case where the first finisher failed
- Complete provider management with cleanup

GenAI Context:
In production, LLM provider latency varies wildly. By racing
providers against each other, your users always get the fastest
available response. If one provider is having a bad day, the
others pick up the slack automatically.

Concepts covered:
- asyncio.create_task() for immediate task scheduling
- asyncio.wait() with FIRST_COMPLETED
- Task cancellation and cleanup
- Handling failed "winners" (first to finish but with an error)
- Combining wait() with wait_for() for overall timeout

Run: uv run uvicorn 09_5_multi_provider_failover:app --reload --port 8000
"""

import time
import asyncio
import random
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Provider Failover (Race Pattern)")


# ===== SIMULATED LLM PROVIDERS =====

async def call_anthropic(prompt: str) -> dict:
    """Anthropic Claude — variable latency, occasionally fails"""
    latency = random.choice([1.5, 2.0, 3.0, 5.0, 8.0])
    fail = random.random() < 0.15  # 15% failure rate

    logger.info(f"  [anthropic] Starting (latency={latency}s, will_fail={fail})")
    await asyncio.sleep(latency if not fail else latency * 0.3)

    if fail:
        raise ConnectionError("Anthropic: rate limit exceeded")

    return {
        "provider": "anthropic",
        "model": "Claude 3",
        "response": f"Claude's response to: {prompt[:40]}",
        "latency": f"{latency:.1f}s",
    }


async def call_openai(prompt: str) -> dict:
    """OpenAI GPT-4 — generally fast, occasionally slow"""
    latency = random.choice([1.0, 1.5, 2.0, 4.0, 10.0])
    fail = random.random() < 0.1  # 10% failure rate

    logger.info(f"  [openai] Starting (latency={latency}s, will_fail={fail})")
    await asyncio.sleep(latency if not fail else latency * 0.3)

    if fail:
        raise ConnectionError("OpenAI: service temporarily unavailable")

    return {
        "provider": "openai",
        "model": "GPT-4",
        "response": f"GPT-4's response to: {prompt[:40]}",
        "latency": f"{latency:.1f}s",
    }


async def call_cohere(prompt: str) -> dict:
    """Cohere Command R+ — reliable but slower"""
    latency = random.choice([2.0, 2.5, 3.0, 3.5])
    fail = random.random() < 0.05  # 5% failure rate

    logger.info(f"  [cohere] Starting (latency={latency}s, will_fail={fail})")
    await asyncio.sleep(latency if not fail else latency * 0.3)

    if fail:
        raise ConnectionError("Cohere: internal server error")

    return {
        "provider": "cohere",
        "model": "Command R+",
        "response": f"Command R+'s response to: {prompt[:40]}",
        "latency": f"{latency:.1f}s",
    }


# ===== DATA MODELS =====

class ChatRequest(BaseModel):
    prompt: str
    providers: list[str] = ["anthropic", "openai"]
    overall_timeout: float = 15.0


PROVIDER_MAP = {
    "anthropic": call_anthropic,
    "openai": call_openai,
    "cohere": call_cohere,
}


# ===== HELPER: RACE WITH CLEANUP =====

async def race_providers(prompt: str, provider_names: list[str]) -> dict:
    """
    Race multiple providers. Return the first successful response.
    Cancel all remaining providers after getting a winner.
    If the first finisher failed, wait for the next one.
    """
    # Create tasks for all providers
    tasks = {}
    for name in provider_names:
        if name in PROVIDER_MAP:
            task = asyncio.create_task(PROVIDER_MAP[name](prompt), name=name)
            tasks[task] = name

    remaining = set(tasks.keys())
    errors = []

    while remaining:
        # Wait for the first task to complete
        done, remaining = await asyncio.wait(
            remaining,
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in done:
            provider_name = tasks[task]

            # Check if the completed task succeeded
            if task.exception():
                error = task.exception()
                logger.warning(f"  [RACE] {provider_name} failed: {error}")
                errors.append({"provider": provider_name, "error": str(error)})
                continue

            # We have a winner!
            result = task.result()
            logger.info(f"  [RACE] Winner: {provider_name}")

            # Cancel all remaining tasks
            for pending_task in remaining:
                pending_name = tasks[pending_task]
                pending_task.cancel()
                logger.info(f"  [RACE] Cancelled: {pending_name}")

            return {
                **result,
                "race_winner": True,
                "providers_tried": [e["provider"] for e in errors] + [provider_name],
                "providers_cancelled": [tasks[t] for t in remaining],
                "errors": errors if errors else None,
            }

    # All providers failed
    raise HTTPException(
        status_code=503,
        detail={
            "error": "all_providers_failed",
            "message": "All LLM providers failed. Please try again.",
            "errors": errors,
        },
    )


# ===== ENDPOINTS =====

@app.get("/health")
def health():
    return {"status": "healthy", "providers": list(PROVIDER_MAP.keys())}


@app.post("/chat/race")
async def chat_race(body: ChatRequest):
    """
    Race providers — return the fastest successful response.

    Sends the prompt to all specified providers simultaneously.
    Returns whichever responds successfully first.
    Cancels the slower providers to save resources.
    """
    start = time.time()

    logger.info(f"Race: {body.providers} | prompt='{body.prompt[:40]}'")

    try:
        result = await asyncio.wait_for(
            race_providers(body.prompt, body.providers),
            timeout=body.overall_timeout,
        )

        elapsed = time.time() - start
        return {
            **result,
            "total_elapsed": f"{elapsed:.3f}s",
        }

    except asyncio.TimeoutError:
        elapsed = time.time() - start
        logger.error(f"  [RACE] Overall timeout after {elapsed:.1f}s")
        raise HTTPException(
            status_code=504,
            detail={
                "error": "overall_timeout",
                "message": f"No provider responded within {body.overall_timeout}s",
                "elapsed": f"{elapsed:.3f}s",
            },
        )


@app.post("/chat/race-all-three")
async def chat_race_all_three(body: ChatRequest):
    """
    Race ALL three providers — maximum resilience.
    Even if two providers fail, the third can still save the request.
    """
    body.providers = ["anthropic", "openai", "cohere"]

    start = time.time()

    logger.info(f"Race (all 3): prompt='{body.prompt[:40]}'")

    try:
        result = await asyncio.wait_for(
            race_providers(body.prompt, body.providers),
            timeout=body.overall_timeout,
        )

        elapsed = time.time() - start
        return {
            **result,
            "total_elapsed": f"{elapsed:.3f}s",
        }

    except asyncio.TimeoutError:
        elapsed = time.time() - start
        raise HTTPException(
            status_code=504,
            detail=f"No provider responded within {body.overall_timeout}s",
        )
```

**2. Run the server:**

```bash
cd FastAPI/projects/fastapi_concepts_hands_on
uv run uvicorn 09_5_multi_provider_failover:app --reload --port 8000
```

### Test It

```bash
# ── Test 1: Race two providers ──
curl -X POST http://localhost:8000/chat/race \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain the event loop", "providers": ["anthropic", "openai"]}'
# Returns whichever responds first. Shows which was cancelled.

# ── Test 2: Race all three providers ──
curl -X POST http://localhost:8000/chat/race-all-three \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is RAG?"}'
# Maximum resilience — up to 2 providers can fail and you still get a response.

# ── Test 3: Run multiple times to see different winners ──
for i in {1..5}; do
  echo "--- Run $i ---"
  curl -s -X POST http://localhost:8000/chat/race-all-three \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Quick test"}' | python3 -m json.tool | grep -E "provider|total_elapsed|winner|cancelled"
  echo ""
done
# Different providers win each time based on random latency

# ── Test 4: Short overall timeout ──
curl -X POST http://localhost:8000/chat/race \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test timeout", "providers": ["anthropic", "openai"], "overall_timeout": 2.0}'
# Likely to timeout — only very fast responses make it under 2s

# ── Test 5: Single provider (no racing, just timeout) ──
curl -X POST http://localhost:8000/chat/race \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Solo test", "providers": ["cohere"]}'
# Works with a single provider too — just adds timeout protection
```

### What You Should See

```json
// Test 1 — OpenAI wins the race:
{
  "provider": "openai",
  "model": "GPT-4",
  "response": "GPT-4's response to: Explain the event loop",
  "latency": "1.0s",
  "race_winner": true,
  "providers_tried": ["openai"],
  "providers_cancelled": ["anthropic"],
  "errors": null,
  "total_elapsed": "1.003s"
}

// Test 2 — Anthropic fails, OpenAI wins:
{
  "provider": "openai",
  "model": "GPT-4",
  "race_winner": true,
  "providers_tried": ["anthropic", "openai"],
  "providers_cancelled": ["cohere"],
  "errors": [{"provider": "anthropic", "error": "rate limit exceeded"}],
  "total_elapsed": "1.503s"
}

// Server logs show the race:
10:30:00 | Race (all 3): prompt='What is RAG?'
10:30:00 |   [anthropic] Starting (latency=3.0s, will_fail=False)
10:30:00 |   [openai] Starting (latency=1.5s, will_fail=False)
10:30:00 |   [cohere] Starting (latency=2.5s, will_fail=False)
10:30:01 |   [RACE] Winner: openai
10:30:01 |   [RACE] Cancelled: anthropic
10:30:01 |   [RACE] Cancelled: cohere
```

### Key Takeaway

The race pattern with `asyncio.wait(FIRST_COMPLETED)` gives your users the fastest possible response from any available provider. Combined with task cancellation, you don't waste resources on providers that finish after the winner. If a provider fails (the first one to "finish" has an exception), the race continues with the remaining providers. This is the most resilient pattern for production GenAI APIs where provider reliability varies.

---

## What's Next?

After completing these exercises, you have a solid foundation in async patterns for GenAI:

| Exercise | Pattern | When to Use |
|----------|---------|-------------|
| 1. Basic async | `async def` + `await` | Every endpoint with I/O |
| 2. Parallel calls | `asyncio.gather()` | Multi-analysis, batch embeddings |
| 3. Timeouts | `asyncio.wait_for()` | Every LLM call in production |
| 4. Background tasks | `BackgroundTasks` | Logging, analytics, caching |
| 5. Provider racing | `asyncio.wait(FIRST_COMPLETED)` | Minimum-latency multi-provider |

**Topic 6: Database Integration** — Apply async patterns to database operations with async SQLAlchemy. Combine async LLM calls with async database reads/writes for fully non-blocking GenAI endpoints.

Your async skills from this topic are the foundation of everything that follows!
