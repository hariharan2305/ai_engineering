"""
A simple FastAPI application demonstrating the use of query parameters.

This application includes:
1. An endpoint to list models with optional pagination and filtering by provider. (use of simple OPTIONAL query parameters with default values)
2. An endpoint to get models for a specific provider with an optional limit. (use of both Path parameters and OPTIONAL query parameters with default values)
3. An endpoint to get detailed models for a specific provider and model type with a required limit. (use of Path parameters and REQUIRED query parameters without default values)

NOTE: Query parameters are key-value pairs that are appended to the URL after a question mark (?). They are used to send additional data to the server and are typically used for filtering, sorting, pagination, and other optional parameters that modify the behavior of the endpoint. In FastAPI, query parameters can be defined in the function signature and can have default values, making them optional for clients to provide when making requests.

"""

from fastapi import FastAPI
from typing import Optional
from enum import Enum

#Create an instance of the FastAPI class
app = FastAPI()

# ====== Models ========
class ModelType(str, Enum):
    GENERAL = "general"
    CODING = "coding"
    CREATIVE = "creative"



# ===== Endpoints ========

# Define a route that accepts query parameters
@app.get("/models")
def list_models(
    skip: int = 0, # OPTIONAL query parameter with a default value of 0. This means that if the client does not provide this query parameter, it will default to 0.
    limit: int = 10, # OPTIONAL query parameter with a default value of 10. This means that if the client does not provide this query parameter, it will default to 10.
    provider: Optional[str] = None # OPTIONAL query parameter that can be None if not provided by the client. This allows clients to filter models by provider if they choose to, but it's not required.
):
    """This endpoint lists the available models with pagination and optional filtering by provider."""
    all_models = [
        {"id": "gpt-4", "provider": "openai", "max_tokens": 8192},
        {"id": "claude-opus-4-5", "provider": "anthropic", "max_tokens": 8192},
        {"id": "llama-3.1", "provider": "meta", "max_tokens": 4096},
        {"id": "gpt-5", "provider": "openai", "max_tokens": 16384},
        {"id": "claude-sonnet-4-5", "provider": "anthropic", "max_tokens": 16384},
    ]

    # Filter models by provider if the query parameter is provided
    if provider:
        all_models = [model for model in all_models if model["provider"] == provider]
    
    # Apply pagination
    models = all_models[skip : skip + limit]
    return {
        "total": len(all_models),
        "skip": skip,
        "limit": limit,
        "models": models
    }

# Combining path and query parameters
@app.get("/providers/{provider_id}/models")
def get_provider_models(
    provider_id: str,
    limit: int = 10 # OPTIONAL query parameter with a default value of 10. This means that if the client does not provide this query parameter, it will default to 10.
):
    """Get models for a specific provider with an optional limit on the number of results."""
    provider_models = [
        {"id": "gpt-4", "provider": "openai", "max_tokens": 8192},
        {"id": "gpt-5", "provider": "openai", "max_tokens": 16384},
        {"id": "claude-opus-4-5", "provider": "anthropic", "max_tokens": 8192},
        {"id": "claude-sonnet-4-5", "provider": "anthropic", "max_tokens": 16384},
        {"id": "llama-3.1", "provider": "meta", "max_tokens": 4096},
    ]

    # Filter models by provider_id
    filtered_models = [model for model in provider_models if model["provider"] == provider_id]
    
    # Apply limit
    limited_models = filtered_models[:limit]
    
    return {
        "provider_id": provider_id,
        "total_models": len(filtered_models),
        "models": limited_models
    }

# Combining path with required and optional query parameters
@app.get("/providers/{provider_id}/models/{model_type}")
def get_detailed_provider_models(
    provider_id: str,
    model_type: ModelType,
    limit: int # REQUIRED query parameter without a default value. This means that the client must provide this query parameter when making a request to this endpoint.
):
    """Get models for a specific provider and model type with required limit."""
    provider_models = [
        {"id": "gpt-4", "provider": "openai", "max_tokens": 8192, "type": "general"},
        {"id": "gpt-5", "provider": "openai", "max_tokens": 16384, "type": "creative"},
        {"id": "claude-opus-4-5", "provider": "anthropic", "max_tokens": 8192, "type": "coding"},
        {"id": "claude-sonnet-4-5", "provider": "anthropic", "max_tokens": 16384, "type": "coding"},
        {"id": "llama-3.1", "provider": "meta", "max_tokens": 4096, "type": "general"},
    ]

    # Filter models by provider_id and model_type
    filtered_models = [model for model in provider_models if model["provider"] == provider_id and model["type"] == model_type]
    
    # Apply limit
    limited_models = filtered_models[:limit]
    
    return {
        "provider_id": provider_id,
        "total_models": len(filtered_models),
        "models": limited_models
    }