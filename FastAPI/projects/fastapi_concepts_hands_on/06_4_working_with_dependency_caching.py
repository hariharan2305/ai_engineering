"""
A FastAPI application demonstrating the use of dependency caching to optimize performance.

This application includes:
1. A `Settings` class that defines application configuration using Pydantic's `BaseSettings`.
2. A dependency function `get_settings` that loads the settings and simulates an expensive operation (e.g., reading from a file or environment variables). This function is designed to be cached so that subsequent calls return the same instance of the settings without reloading.
3. Helper functions `get_api_key_for_model` and `validate_max_tokens` that depend on the settings to provide specific functionality (e.g., determining the correct API key based on the model name and validating requested max tokens against the configured limit).
4. An endpoint `/chat` that uses the settings and helper functions to simulate a chat completion, demonstrating how dependencies can be used to manage configuration and shared logic efficiently.

NOTE: Dependency caching in FastAPI allows you to optimize performance by ensuring that expensive operations (like loading configuration or establishing database connections) are only performed once per request, and the results are reused across multiple dependencies that require the same data. By using the `Depends` mechanism, you can easily manage and share dependencies across your endpoints, leading to cleaner and more efficient code.

Key observation: ⚙️ Loading settings only prints ONCE, even though get_settings() is used in three places!

Why This Matters:
- Without caching, database queries execute multiple times per request
- Request-scoped caching makes dependencies safe and efficient
- Perfect for expensive operations like loading configs or querying databases

Key Takeaway:
- FastAPI caches dependency results within a single request. Call the same dependency 10 times in different places, and it only executes once. This is like Spark's lazy evaluation—computed once, reused everywhere.

"""

from fastapi import FastAPI, Depends, HTTPException
from pydantic_settings import BaseSettings

# Create an instance of the FastAPI class
app = FastAPI()

# ====== Settings ========
class Settings(BaseSettings):
    app_name: str = "GenAI Backend"
    openai_api_key: str = "sk-test-key"
    anthropic_api_key: str = "ant-test-key"
    default_model: str = "gpt-4"
    max_tokens: int = 8192

# ====== Dependency Function ========
def get_settings() -> Settings:
    """Dependency function to provide application settings."""
    print("Dependency: Loading settings...")
    # This is an expensive operation (e.g., reading from a file, environment variables, or a remote config service), so we want to cache the result.
    return Settings()

# Helper functions using Settings
def get_api_key_for_model(model_name: str, settings: Settings = Depends(get_settings)) -> str:
    """Get the appropriate API key based on the model's provider."""
    print(f"Helper Function: Getting API key for model '{model_name}'")
    if model_name.startswith("gpt"):
        return settings.openai_api_key
    elif model_name.startswith("claude"):
        return settings.anthropic_api_key
    else:
        raise ValueError("Unsupported model")
    
def validate_max_tokens(requested_tokens: int, settings: Settings = Depends(get_settings)) -> int:
    """Validate the requested max tokens against the settings."""
    print(f"Helper Function: Validating requested max tokens: {requested_tokens}")
    if requested_tokens > settings.max_tokens:
        raise HTTPException(status_code=400, detail=f"Requested max tokens exceed the limit of {settings.max_tokens}")
    return requested_tokens

# ====== Endpoint ========
@app.post("/chat")
def chat(
    model_name: str, 
    max_tokens: int = 1000,
    settings: Settings = Depends(get_settings), # Direct use of the settings dependency to get configuration values
    api_key: str = Depends(get_api_key_for_model), # Using a helper function that depends on the settings to determine the correct API key for the requested model
    tokens_validated: int = Depends(validate_max_tokens) # Using another helper function that depends on the settings to validate the requested max tokens against the configured limit
):
    """This endpoint simulates a chat completion, using settings and helper functions to manage API keys and token limits."""
    print(f"Endpoint: Received chat request for model '{model_name}' with max tokens {max_tokens}")
    print(f"Using API key: {api_key}")
    return {
        "app_name": settings.app_name,
        "response": f"Simulated response from model '{model_name}' with max tokens {max_tokens}",
        "api_key_used": api_key,
        "tokens_validated": tokens_validated
    }