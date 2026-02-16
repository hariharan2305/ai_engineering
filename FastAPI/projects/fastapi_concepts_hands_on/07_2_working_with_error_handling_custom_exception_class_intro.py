"""
In this example, we will introduce a custom exception class to handle specific error scenarios related to token budget management in a GenAI application.

This script demonstrates:
- Creating exception classes that store context
- Raising custom exceptions in business logic
- Why custom exceptions are better than HTTPException for complex errors

The model here: Users have monthly token budgets for LLM API calls.
When they request a chat that would exceed their budget, we raise an exception.

Concepts covered:
- Creating exception classes with __init__
- Storing context as attributes (user_id, tokens_used, tokens_limit)
- Raising exceptions in business logic (not just in endpoints)
- Exception as a communication mechanism between layers

Benefits of custom exceptions:
- Context stored as attributes: exc.user_id, exc.tokens_used, etc.
- Type-specific catching: except TokenBudgetExceededError vs generic except Exception
- Business logic stays clean: Validation function doesn't know about HTTP
- Separation of concerns: Business logic (check budget) separate from API layer (convert to HTTP)

Key Takeaway:
- Custom exceptions let you carry context through your code. The business logic checks budgets and raises exceptions. Later, exception handlers convert those exceptions to HTTP responses. This separation makes code cleaner and more testable.

"""

from typing import Optional

# ===== Custom Exception Class =====
class GenAIException(Exception):
    """Custom exception class for handling GenAI-related errors."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

class TokenBudgetExceededError(GenAIException):
    """
    Raised when a user would exceed their monthly token budget.

    Attributes:
        user_id: The user who exceeded budget
        tokens_used: Tokens already used this month
        tokens_limit: User's monthly budget
        tokens_needed: Tokens required for current request
    """

    def __init__(
        self,
        user_id: str,
        tokens_used: int,
        tokens_limit: int,
        tokens_needed: Optional[int] = None
    ):
        self.user_id = user_id
        self.tokens_used = tokens_used
        self.tokens_limit = tokens_limit
        self.tokens_needed = tokens_needed

        # Build user-friendly message
        message = (
            f"Token budget exceeded. "
            f"Used {tokens_used}/{tokens_limit} tokens this month. "
        )
        
        if tokens_needed:
            remaining = tokens_limit - tokens_used
            message += f"Current request needs {tokens_needed}, but only {remaining} remaining."

        super().__init__(message)


# ===== BUSINESS LOGIC =====

def estimate_tokens(message: str) -> int:
    """
    Rough token estimation.
    In reality, use tiktoken library for accurate counts.
    """
    # Very rough: 4 characters ≈ 1 token
    return len(message) // 4

def validate_token_budget(user_id: str, tokens_used: int, tokens_limit: int, request_message: str) -> None:
    """
    Check if user can make this request without exceeding budget.

    Raises:
        TokenBudgetExceededError: If request would exceed budget
    """
    tokens_needed = estimate_tokens(request_message)
    tokens_remaining = tokens_limit - tokens_used

    if tokens_needed > tokens_remaining:
        raise TokenBudgetExceededError(
            user_id=user_id,
            tokens_used=tokens_used,
            tokens_limit=tokens_limit,
            tokens_needed=tokens_needed
        )


# ===== USAGE EXAMPLES =====

def test_valid_request():
    """User has budget - no exception"""
    try:
        # User has used 5000/10000 tokens, needs 100 more
        validate_token_budget(
            user_id="alice",
            tokens_used=5000,
            tokens_limit=10000,
            request_message="Hello" * 50  # ~200 tokens
        )
        print("✅ Valid request: User has sufficient budget")
    except TokenBudgetExceededError as e:
        print(f"❌ {e}")


def test_budget_exceeded():
    """User exceeds budget - exception raised"""
    try:
        # User has used 9500/10000 tokens, needs 500 more (exceeds limit)
        validate_token_budget(
            user_id="bob",
            tokens_used=9500,
            tokens_limit=10000,
            request_message="Hello" * 500  # ~2000 tokens
        )
        print("Request succeeded")
    except TokenBudgetExceededError as e:
        print(f"✅ Exception caught: {e.message}")
        print(f"   User: {e.user_id}")
        print(f"   Tokens used: {e.tokens_used}/{e.tokens_limit}")
        print(f"   Tokens needed: {e.tokens_needed}")


def test_exact_budget():
    """User has exactly enough budget"""
    try:
        # User has used 9900/10000, needs exactly 100
        validate_token_budget(
            user_id="charlie",
            tokens_used=9900,
            tokens_limit=10000,
            request_message="Hello" * 25  # ~100 tokens
        )
        print("✅ Valid request: User has exactly enough budget")
    except TokenBudgetExceededError as e:
        print(f"❌ {e}")


if __name__ == "__main__":
    print("Testing custom exceptions...\n")
    test_valid_request()
    print()
    test_budget_exceeded()
    print()
    test_exact_budget()