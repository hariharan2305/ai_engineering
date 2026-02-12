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