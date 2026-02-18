"""
In this example, we set up CORS middleware in a FastAPI application to allow cross-origin requests from specific frontend origins. 

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

Understanding the CORS flow:
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

Key takeaways:
- CORS is a browser security feature, not a server-side concern. Your API must explicitly allow cross-origin requests by including CORS response headers. 
  Without CORSMiddleware, any React/Next.js frontend on a different port will be blocked by the browser. 
  The API response still works (curl proves it), but the browser refuses to pass it to JavaScript.    

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
