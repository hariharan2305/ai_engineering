"""
This example demonstrates how to use asyncio.create_task() and asyncio.wait() to race mulitple providers.
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

Key Takeaway:
- The race pattern with asyncio.wait(FIRST_COMPLETED) gives your users the fastest possible response from any available provider. 
  Combined with task cancellation, you don't waste resources on providers that finish after the winner. 
  If a provider fails (the first one to "finish" has an exception), the race continues with the remaining providers. 
  This is the most resilient pattern for production GenAI APIs where provider reliability varies.

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