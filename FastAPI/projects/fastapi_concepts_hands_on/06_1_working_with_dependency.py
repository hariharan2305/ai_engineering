"""
This FastAPI application demonstrates how to use dependencies to protect endpoints with API key authentication.

This application includes:
1. A dependency function `verify_api_key` that checks for a valid API key in the request header and raises an HTTP 401 error if the key is invalid.
2. Protected endpoints (`/models`, `/chat`, and `/settings`) that require a valid API key to access. The API key is passed to the endpoint functions via the `Depends` mechanism.

NOTE: Dependencies in FastAPI are a powerful way to manage common logic that needs to be executed for multiple endpoints, such as authentication, database connections, or any shared functionality. By using the `Depends` function, you can easily inject dependencies into your endpoint functions, allowing for cleaner and more maintainable code. In this example, we use a dependency to validate API keys, ensuring that only authorized clients can access the protected endpoints.

Having dependencies ensures any request to the protected endpoints must go through the dependency function first. If the API key is invalid, the request is rejected before it reaches the endpoint logic, providing a secure way to protect your API.

"""

from fastapi import FastAPI, Depends, Header, HTTPException

# Create an instance of the FastAPI class
app = FastAPI()

# ====== Dependency Function ========
def verify_api_key(x_api_key: str = Header(...)) -> str:
    """Validate API key from the request header."""
    print(f"Dependency: Verifying API key: {x_api_key}")

    valid_keys = ["dev-key-123", "prod-key-456"]

    if x_api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    print("Dependency: API key is valid")
    return x_api_key

# ====== Protected Endpoints ========
@app.get("/models")
def list_models(api_key: str = Depends(verify_api_key)):
    """This endpoint lists the available models, but only if a valid API key is provided."""
    print(f"Endpoint: Received request with API key: {api_key}")
    return [
        {"id": "gpt-4", "provider": "openai", "max_tokens": 8192},
        {"id": "claude-opus-4-5", "provider": "anthropic", "max_tokens": 8192},
        {"id": "llama-3.1", "provider": "meta", "max_tokens": 4096},
    ]

@app.post("/chat")
def chat(message: str, api_key: str = Depends(verify_api_key)):
    """This endpoint simulates a chat completion, but only if a valid API key is provided."""
    print(f"Endpoint: Received chat message: '{message}' with API key: {api_key}")
    return {
        "response": f"Echoing your message: '{message}'",
        "api_key_used": api_key
    }

@app.get("/settings")
def get_settings(api_key: str = Depends(verify_api_key)):
    """This endpoint returns user settings, but only if a valid API key is provided."""
    print(f"Endpoint: Fetching settings with API key: {api_key}")
    return {
        "theme": "dark",
        "notifications": True,
        "api_key_used": api_key
    }