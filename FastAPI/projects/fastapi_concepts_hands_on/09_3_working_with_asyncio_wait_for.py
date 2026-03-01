"""
This example demonstrates how to handle timeouts when calling LLM providers in FastAPI.
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

Key Takewaway:
- asyncio.wait_for() is your safety net for LLM calls. 
  Without timeouts, a provider outage cascades into server-wide failure as connections pile up. 
  With timeouts plus failover, your API stays responsive even when providers misbehave. Always set timeouts — the only question is how long to wait.




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