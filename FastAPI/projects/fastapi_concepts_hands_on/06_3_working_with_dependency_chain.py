"""
A simple FastAPI application demonstrating a dependency chain for API key validation, user retrieval, and token usage check.

This application includes:
1. A dependency function `verify_api_key` that checks for a valid API key in the request header and raises an HTTP 401 error if the key is invalid.
2. A dependency function `get_current_user` that retrieves the user associated with the validated API key.
3. A dependency function `check_token_usage` that checks if the user has exceeded their token limit and raises an HTTP 403 error if the limit is exceeded.
4. A protected endpoint `/chat` that simulates a chat completion, but only if the user has a valid API key and has not exceeded their token limit.
5. A protected endpoint `/profile` that returns the user's profile information, but only if they have a valid API key.

NOTE: This example demonstrates a dependency chain where each dependency builds upon the previous one. The `chat` endpoint depends on `check_token_usage`, which depends on `get_current_user`, which in turn depends on `verify_api_key`. This allows for a clean and modular way to handle complex authentication and authorization logic in FastAPI.

Dependencies execute in order (deepest first → shallowest → endpoint). If any step fails, later steps don't run. This creates a powerful pipeline: validate → fetch user → check limits → business logic.

"""

from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel

# Create an instance of the FastAPI class
app = FastAPI()

# ====== Models ========
class User(BaseModel):
    id: int
    email: str
    tokens_used: int
    token_limit: int

# ====== Mock Database ========
USERS = {
    "key-123": User(id=1, email="user1@example.com", tokens_used=250, token_limit=1000),
    "key-456": User(id=2, email="user2@example.com", tokens_used=1500, token_limit=5000),
}

# ====== Dependency Functions ========
# Level 1: API Key Validation
def verify_api_key(x_api_key: str = Header(...)) -> User:
    """Validate API key and return the validated key."""
    print(f"Dependency: Verifying API key: {x_api_key}")

    if x_api_key not in USERS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    return x_api_key

# Level 2: User Retrieval
def get_current_user(api_key: str = Depends(verify_api_key)) -> User:
    """Retrieve the user associated with the validated API key."""
    print(f"Dependency: Retrieving user for API key: {api_key}")
    user = USERS[api_key]
    print(f"Dependency: Retrieved user: {user.email}")
    return user

# Level 3: Token Usage Check
def check_token_usage(user: User = Depends(get_current_user)) -> User:
    """Check if the user has exceeded their token limit."""
    print(f"Dependency: Checking token usage for user: {user.email}")
    if user.tokens_used >= user.token_limit:
        raise HTTPException(status_code=403, detail="Token limit exceeded")
    print(f"Dependency: User {user.email} has used {user.tokens_used}/{user.token_limit} tokens")
    return user

# ====== Protected Endpoint ========
@app.post("/chat")
def chat(prompt: str, user: User = Depends(check_token_usage)):
    """This endpoint simulates a chat completion, but only if the user has a valid API key and has not exceeded their token limit."""
    print(f"Endpoint: Received chat prompt: '{prompt}' from user: {user.email}")
    
    # Simulate token usage for this request (for demonstration purposes)
    user.tokens_used += len(prompt.split()) + 10  # Assume each prompt uses tokens based on word count + response tokens
    remaining_tokens = user.token_limit - user.tokens_used    

    return {
        "response": f"Echoing your message: '{prompt}'",
        "user_email": user.email,
        "tokens_used": user.tokens_used,
        "token_limit": user.token_limit,
        "remaining_tokens": remaining_tokens
    }

@app.get("/profile")
def get_profile(user: User = Depends(get_current_user)):
    """This endpoint returns the user's profile information, but only if they have a valid API key."""
    print(f"Endpoint: Fetching profile for user: {user.email}")
    return {
        "id": user.id,
        "email": user.email,
        "tokens_used": user.tokens_used,
        "token_limit": user.token_limit,
        "remaining_tokens": user.token_limit - user.tokens_used
    }