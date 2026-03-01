"""
This example demonstrates how to use FastAPI's BackgroundTasks to perform post-response processing for GenAI applications.
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

Key Takeaway:
- Background tasks let you separate user-facing work from bookkeeping. 
  The user gets their LLM response in 1.5s instead of 2.5s because logging, analytics, and caching happen after the response is sent. 
  In a production GenAI app, this means faster perceived response times while still capturing all the operational data you need.

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