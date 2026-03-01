"""
This example demonstrates the difference between synchronous and asynchronous endpoints in FastAPI.
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

Key Takeaway:
- The async endpoint handles multiple concurrent requests without blocking. 
  When one request awaits an LLM response, the event loop picks up the next request. 
  This is why async is essential for GenAI backends — your users shouldn't have to wait in line behind each other for LLM responses.

"""

import time 
import asyncio
import logging 

from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Sync vs Async Example")


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