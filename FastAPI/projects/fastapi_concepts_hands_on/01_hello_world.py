from fastapi import FastAPI

# Create an instance of the FastAPI class
app = FastAPI(
    title="GenAI Backend",
    version="0.1.0"
)

# Define a route for the root URL
@app.get("/")
def root():
    return {"message": "Welcome to GenAI Backend!"}

# Define a route for the /health endpoint
@app.get("/health")
def health():
    return {"status": "Healthy"}

