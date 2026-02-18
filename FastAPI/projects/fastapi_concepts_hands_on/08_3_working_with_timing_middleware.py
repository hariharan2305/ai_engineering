"""
In this example, we implement a custom TimingMiddleware to measure and log the processing time of each request.

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

Key takeaways:
- Timing middleware gives you automatic visibility into endpoint performance without adding timing code to every endpoint. 
  For GenAI APIs, this is essential because LLM calls are orders of magnitude slower than typical operations. 
  The X-Process-Time header lets frontends show "Response took 2.3s" to users, and the slow request warnings alert you to degraded performance in production.


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