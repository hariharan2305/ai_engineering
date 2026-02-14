"""
A simple FastAPI application demonstrating the use of path parameters.

This application includes:
1. An endpoint to get model details using a path parameter with simple type validation.
2. An endpoint to get supported models using an enum-validated path parameter.
3. An endpoint to get user conversation details using multiple path parameters (NOTE: The position of the path parameters in the URL matters and should be captured in the same order).

NOTE: Path parameters are variables that are part of the URL path and are used to capture dynamic values from the URL. In FastAPI, path parameters are defined in the route string using curly braces (e.g., /items/{item_id}) and are passed as arguments to the endpoint function. Path parameters are required by default, meaning that clients must provide a value for them when making a request to the endpoint.

"""

from fastapi import FastAPI
from enum import Enum

# Create an instance of the FastAPI class
app = FastAPI()

# ====== Models ========
class ModelName(str, Enum):
    GPT4 = "gpt-4"
    CLAUDE = "claude-opus-4-5"
    LLAMA = "llama-3.1"


# ====== Endpoints ========
# Define a route that accepts a path parameter
@app.get("/models/{model_id}")
def get_model(model_id: str):
    return {"model_id": model_id, "type": "GenAI Model"}


# Enum validated path parameter
# Having a data model/schema for the path parameters allows us to validate the input and also provides better documentation in the OpenAPI schema.
@app.get("/supported-models/{model_name}")
def get_supported_model(model_name: ModelName):
    return {
        "model": model_name,
        "available": True
    }

# Define a route that accepts multiple path parameters
@app.get("/users/{user_id}/conversations/{conversation_id}")
def get_user_conversation(user_id: int, conversation_id: int):
    return {
        "user_id": user_id,
        "conversation_id": conversation_id
    }