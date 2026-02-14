"""
A simple FastAPI application demonstrating the use of response status codes.

This application includes:
1. An endpoint to create a new model that returns a 201 Created status code upon successful creation.
2. An endpoint to delete a model that returns a 204 No Content status code upon successful deletion.
3. An endpoint to get a model by its ID that returns a 404 Not Found status code if the model does not exist.

NOTE: Response status codes are standardized codes that indicate the result of an HTTP request. In FastAPI, you can specify the status code for a response using the `status_code` parameter in the route decorator. 
Common status codes includes:
- 200 OK for successful GET requests
- 201 Created for successful POST requests that create a resource
- 204 No Content for successful DELETE requests
- 404 Not Found when a requested resource cannot be found
- 400 Bad Request for invalid input data
- 401 Unauthorized for authentication errors
- 403 Forbidden for authorization errors
- 500 Internal Server Error for unexpected server errors

Using appropriate status codes helps clients understand the outcome of their requests and allows for better error handling.

"""

from fastapi import FastAPI
from fastapi import status, HTTPException

# Create an instance of the FastAPI class
app = FastAPI()

# ====== Endpoints ========
@app.post("/models", status_code=status.HTTP_201_CREATED)
def create_model(model_name: str):
    """Create a new model with the given name."""
    return {
        "id": "model-123",
        "name": model_name,
        "status": "created"
    }

@app.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(model_id: str):
    """Delete a model by its ID."""
    pass

@app.get("/models/{model_id}")
def get_model_or_404(model_id: str):
    """Get a model by its ID, or return a 404 error if not found.""" 
    models = {"gpt-4": {"name": "GPT-4", "status": "available"}}

    if model_id not in models:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {model_id} not found")
    
    return models[model_id]