"""
This example demonstrates how to use asyncio.gather() to call multiple LLM providers in parallel.
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

Key Takeaway:
- asyncio.gather() turns sequential LLM calls into parallel ones, reducing total latency from the sum of all calls to the max of all calls. 
  With return_exceptions=True, one flaky provider doesn't crash the entire request — you get partial results and can handle failures gracefully. 
  This is the workhorse pattern for any multi-provider or multi-analysis GenAI endpoint.

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