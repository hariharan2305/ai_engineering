# FastAPI for GenAI Builders: Fast Track Roadmap

**Your Goal:** Ship GenAI applications. Not become a senior backend engineer.

**Your Background:** ML engineer who wants to build end-to-end GenAI systems—potentially as an entrepreneur.

**Time:** ~6 weeks (8-10 hours/week)

---

## Philosophy: Ship First, Optimize Later

This roadmap makes opinionated tradeoffs:

| We Optimize For | We Sacrifice |
|-----------------|--------------|
| Shipping working products | Perfect code architecture |
| GenAI-specific patterns | Generic backend knowledge |
| Simple deployment (Railway/Render) | NGINX + Kubernetes mastery |
| Practical testing | 80%+ test coverage |
| Learning by building | Comprehensive theory |

**The GenAI landscape moves fast.** You need to ship, learn from users, and iterate—not spend 12 weeks building perfect infrastructure for an app that might not find product-market fit.

---

## Quick Overview

```
Phase 1: FastAPI Foundation (Week 1)
├─ Topic 1: HTTP/REST Essentials (condensed)
├─ Topic 2: FastAPI Core Concepts
└─ Topic 3: Error Handling Basics

Phase 2: Backend Essentials (Week 2)
├─ Topic 4: Middleware & Request Lifecycle
├─ Topic 5: Async FastAPI (critical for LLMs)
└─ Topic 6: Database Integration (PostgreSQL + Redis basics)

Phase 3: LLM Integration (Week 3)
├─ Topic 7: Integrating LLM Providers
├─ Topic 8: Streaming Responses (SSE)
└─ Mini-Project 1: Multi-Provider Chat API

Phase 4: Security & Cost Control (Week 4)
├─ Topic 9: Authentication (API Keys + JWT)
├─ Topic 10: Rate Limiting & Usage Tracking
└─ Mini-Project 2: Secure Chat with Usage Limits

Phase 5: GenAI Patterns (Week 5)
├─ Topic 11: GenAI-Specific Patterns
├─ Topic 12: Function Calling & Tool Use
├─ Topic 13: Vector Databases & RAG
└─ Mini-Project 3: RAG System or Agent

Phase 6: Ship It (Week 6)
├─ Topic 14: Docker for Deployment
├─ Topic 15: Cloud Deployment (Railway/Render/Fly.io)
└─ Final Project: Deployed GenAI Application
```

**Total:** 15 Topics + 3 Mini-Projects + 1 Final Project

### 📊 Current Progress

| Phase | Topics | Status |
|-------|--------|--------|
| Phase 1 | Topics 1-3 | Topic 1 ✅, Topic 2 ✅, Topic 3 🔧 Next |
| Phase 2 | Topics 4-6 | 📋 Pending |
| Phase 3 | Topics 7-8 + Mini-Project 1 | 📋 Pending |
| Phase 4 | Topics 9-10 + Mini-Project 2 | 📋 Pending |
| Phase 5 | Topics 11-13 + Mini-Project 3 | 📋 Pending |
| Phase 6 | Topics 14-15 + Final Project | 📋 Pending |

**Next Action:** Complete Topic 3 (Error Handling) to finish Phase 1

---

## Phase 1: FastAPI Foundation

**Goal:** Get productive with FastAPI in one week.

---

### ✅ Topic 1: HTTP/REST Essentials (Condensed) — COMPLETED

**Target Files:**
- `concepts/01_HTTP_REST_API_Basics.md` ✅ (comprehensive coverage)

**Learning Objectives:**

**Part 1: HTTP Methods & CRUD Operations**
- Understanding the HTTP request/response cycle
- Mapping HTTP methods to database operations (GET→Read, POST→Create, PUT→Update, DELETE→Delete)
- When to use POST vs PUT vs PATCH
- Idempotency and why it matters

**Part 2: Status Codes That Matter**
- 2xx success codes (200 OK, 201 Created, 204 No Content)
- 4xx client error codes (400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found, 422 Validation Error, 429 Rate Limited)
- 5xx server error codes (500 Internal Error, 502 Bad Gateway, 503 Service Unavailable)
- Choosing the right status code for your API responses

**Part 3: REST Principles for GenAI**
- Why statelessness matters (load balancer routing, horizontal scaling)
- Why LLM APIs include full conversation history in every request
- Resource-based URL design patterns
- Query parameters vs request body decisions

**What You'll Be Able to Build:**
- RESTful endpoint design for chat applications
- Proper error responses with correct status codes
- URL structures that follow industry conventions

**Key Questions You Should Be Able to Answer:**
1. Why do LLM APIs require the full conversation history in every request?
2. What's the difference between 401 and 403 status codes?
3. When should you use 422 vs 400 for validation errors?
4. Why is statelessness important for horizontally scaled GenAI apps?

**Hands-On Exercises:**
- Exercise 1: Design RESTful URLs for a chat API
- Exercise 2: Map HTTP methods to conversation operations
- Exercise 3: Create a status code reference for your app

**GenAI Application:** Understanding statelessness explains why every Claude/OpenAI API call includes the full message history—servers don't remember previous requests, enabling seamless load balancing.

**Estimated Time:** 1 hour concepts + 1 hour hands-on

---

### ✅ Topic 2: FastAPI Core Concepts — COMPLETED

**Target Files:**
- `concepts/02_FastAPI_Core_Concepts.md` ✅
- `concepts/02_FastAPI_Hands_On_Practice.md` ✅ (6 exercises completed)

> **Note:** Dependency injection is covered at a basic level. For advanced patterns (chaining, request-scoped dependencies, sub-dependencies), see the Master Roadmap's advanced extensions.

**Learning Objectives:**

**Part 1: Application Setup**
- Creating a FastAPI application instance
- Running with Uvicorn (development server)
- Understanding ASGI and why it matters for async

**Part 2: Path Operations**
- Decorators for HTTP methods (@app.get, @app.post, etc.)
- Path parameters with automatic type validation
- Query parameters with defaults and constraints
- Combining path and query parameters

**Part 3: Pydantic Models for Validation**
- Request body models with automatic validation
- Field constraints (min_length, max_length, ge, le)
- Optional fields with default values
- Nested models for complex data structures

**Part 4: Response Handling**
- Response models for consistent output
- HTTP status codes in decorators
- Automatic JSON serialization

**Part 5: Dependency Injection**
- Creating reusable dependencies
- Sharing logic across endpoints (auth, database sessions)
- Dependency hierarchy and chaining

**Part 6: Automatic Documentation**
- Interactive Swagger UI at /docs
- ReDoc alternative at /redoc
- OpenAPI schema generation

**What You'll Be Able to Build:**
- A basic chat API with validated request/response models
- Endpoints with proper input validation
- Reusable dependencies for common operations

**Key Questions You Should Be Able to Answer:**
1. How does FastAPI use Python type hints for validation?
2. What's the difference between path parameters and query parameters?
3. When should you use a dependency vs putting logic in the endpoint?
4. Why does FastAPI generate documentation automatically?

**Hands-On Exercises:**
- Exercise 1: Build a `/health` endpoint
- Exercise 2: Create Pydantic models for chat messages (ChatRequest, ChatResponse)
- Exercise 3: Add query parameter pagination to a list endpoint
- Exercise 4: Use dependencies for shared validation logic
- Exercise 5: Explore the auto-generated `/docs` page

**GenAI Application:** Pydantic models ensure that message content, model names, and temperature values are validated before reaching your LLM integration code. Invalid data never touches your business logic.

**Estimated Time:** 3 hours concepts + 3 hours hands-on

---

### 🔧 Topic 3: Error Handling Basics — NEXT

**Target Files:**
- `concepts/03_Error_Handling.md` (to be created)
- `concepts/03_Error_Handling_Practice.md` (to be created)

**Learning Objectives:**

**Part 1: HTTPException**
- Using HTTPException for expected errors
- Structuring error detail payloads
- Common status code patterns (404 for not found, 400 for bad requests)

**Part 2: Custom Exception Classes**
- Creating domain-specific exceptions (LLMProviderError, RateLimitExceeded, TokenBudgetExceeded)
- Storing context in exception attributes
- Raising custom exceptions in business logic

**Part 3: Global Exception Handlers**
- Registering handlers with @app.exception_handler
- Converting custom exceptions to JSON responses
- Adding response headers (Retry-After for rate limits)
- Catch-all handlers for unexpected errors

**Part 4: Logging Errors**
- Logging exceptions for debugging
- Separating user-facing messages from internal details
- Not exposing stack traces to users

**What You'll Be Able to Build:**
- Consistent error responses across your API
- Custom exceptions for GenAI-specific failures (provider errors, token limits)
- Centralized error logging for debugging

**Key Questions You Should Be Able to Answer:**
1. When should you use HTTPException vs a custom exception class?
2. How do global exception handlers improve code organization?
3. What information should you expose to API users vs log internally?
4. How do you add headers like Retry-After to error responses?

**Hands-On Exercises:**
- Exercise 1: Add HTTPException for missing resources (404)
- Exercise 2: Create a custom `TokenBudgetExceeded` exception
- Exercise 3: Build a global exception handler for LLM provider errors
- Exercise 4: Add error logging that captures request context
- Exercise 5: Test error responses with different scenarios

**GenAI Application:** LLM API calls fail frequently (rate limits, timeouts, provider outages). Proper error handling ensures users get actionable feedback and your system degrades gracefully.

**Estimated Time:** 2 hours concepts + 2 hours hands-on

---

## Phase 2: Backend Essentials

**Goal:** Learn the patterns that make real GenAI apps work.

---

### 📋 Topic 4: Middleware & Request Lifecycle

**Target Files:**
- `concepts/04_Middleware.md`
- `concepts/04_Middleware_Practice.md`

**Learning Objectives:**

**Part 1: Request/Response Lifecycle**
- Understanding how requests flow through FastAPI
- Middleware execution order (first in, last out)
- When to use middleware vs dependencies

**Part 2: Built-in Middleware**
- CORS middleware for web frontend integration
- Configuring allowed origins, methods, and headers
- Why CORS matters for browser-based GenAI apps

**Part 3: Custom Middleware Patterns**
- Request ID injection for distributed debugging
- Request timing to identify slow endpoints
- Basic request/response logging
- Accessing request state across middleware

**Part 4: Practical Middleware for GenAI**
- Adding request IDs to track LLM calls
- Logging slow requests (especially LLM calls that take 10+ seconds)
- Passing context through the request lifecycle

**What You'll Be Able to Build:**
- CORS-enabled API for web frontends
- Request tracking with unique IDs
- Performance monitoring via timing middleware

**Key Questions You Should Be Able to Answer:**
1. Why is CORS middleware required for browser-based GenAI apps?
2. How do you pass data between middleware and endpoints using request.state?
3. What's the execution order when you have multiple middleware?
4. When should you use middleware vs a dependency for cross-cutting concerns?

**Hands-On Exercises:**
- Exercise 1: Configure CORS for a React/Next.js frontend
- Exercise 2: Create request ID middleware
- Exercise 3: Add timing middleware that logs slow requests
- Exercise 4: Build logging middleware for debugging

**GenAI Application:** Request IDs are essential for debugging production issues. When a user reports "my request failed," you need to trace that specific request through your logs and LLM provider calls.

**Estimated Time:** 2 hours concepts + 2 hours hands-on

---

### 📋 Topic 5: Async FastAPI (Critical for LLMs)

**Target Files:**
- `concepts/05_Async_FastAPI.md`
- `concepts/05_Async_FastAPI_Practice.md`

**Learning Objectives:**

**Part 1: Why Async Matters for GenAI**
- LLM API calls take 1-30 seconds
- Sync endpoints block other users during I/O wait
- Async enables concurrent request handling
- Understanding the event loop concept

**Part 2: async/await Fundamentals**
- Converting sync functions to async
- When to use async vs sync in FastAPI
- The asyncio module basics
- Async context managers

**Part 3: Concurrent Operations**
- asyncio.gather() for parallel API calls
- asyncio.wait() with FIRST_COMPLETED for racing providers
- Cancelling pending tasks
- Error handling in concurrent operations

**Part 4: Async Patterns for GenAI**
- Timeouts for LLM calls with asyncio.wait_for()
- Parallel embedding generation
- Background tasks for non-blocking logging
- Async HTTP clients (httpx)

**What You'll Be Able to Build:**
- Concurrent LLM provider calls (try multiple, return fastest)
- Timeouts that prevent hanging requests
- Background logging that doesn't slow responses

**Key Questions You Should Be Able to Answer:**
1. Why is async critical for LLM-powered applications?
2. What's the difference between asyncio.gather() and asyncio.wait()?
3. How do you add timeouts to prevent LLM calls from hanging forever?
4. When should you use background tasks vs awaiting?

**Hands-On Exercises:**
- Exercise 1: Convert a sync endpoint to async
- Exercise 2: Make parallel LLM calls with asyncio.gather()
- Exercise 3: Implement timeout handling for LLM requests
- Exercise 4: Use background tasks for logging
- Exercise 5: Build a multi-provider call that returns the fastest response

**GenAI Application:** Without async, one slow LLM call blocks all other users. With async, 100 users can wait for LLM responses simultaneously. This is the foundation of scalable GenAI backends.

**Estimated Time:** 3 hours concepts + 3 hours hands-on

---

### 📋 Topic 6: Database Integration

**Target Files:**
- `concepts/06_Database_Integration.md`
- `concepts/06_Database_Integration_Practice.md`

**Learning Objectives:**

**Part 1: PostgreSQL with Async SQLAlchemy**
- Why PostgreSQL for GenAI apps (structured data, JSON support, reliability)
- Async SQLAlchemy setup with asyncpg driver
- Database session management with dependencies
- Connection pooling basics

**Part 2: GenAI Data Models**
- User model (authentication, preferences, token limits)
- Conversation model (metadata, system prompts, model selection)
- Message model (role, content, token counts, latency tracking)
- Relationships and foreign keys

**Part 3: CRUD Operations**
- Creating records (async session.add, commit, refresh)
- Reading with filters and pagination
- Updating existing records
- Soft deletes vs hard deletes

**Part 4: Redis for GenAI**
- Redis setup with redis-py async
- Caching LLM responses (when appropriate)
- Session/conversation context storage
- Rate limiting data (covered more in Topic 10)

**What You'll Be Able to Build:**
- Persistent conversation storage
- User management with token tracking
- Message history with pagination
- Response caching layer

**Key Questions You Should Be Able to Answer:**
1. Why use async SQLAlchemy instead of sync for GenAI apps?
2. What data should you track per message for analytics?
3. When should you cache LLM responses, and when shouldn't you?
4. How do you paginate conversation history efficiently?

**Hands-On Exercises:**
- Exercise 1: Set up async SQLAlchemy with PostgreSQL
- Exercise 2: Create User, Conversation, and Message models
- Exercise 3: Build CRUD operations for conversations
- Exercise 4: Add pagination to message retrieval
- Exercise 5: Implement Redis caching for recent conversations
- Exercise 6: Track token usage per user

**GenAI Application:** Every production chat app needs to store conversations, track token usage, and cache expensive operations. This topic builds the data layer that all GenAI features depend on.

**Estimated Time:** 4 hours concepts + 4 hours hands-on

---

## Phase 3: LLM Integration

**Goal:** Connect to real LLM providers and handle streaming.

---

### 📋 Topic 7: Integrating LLM Providers

**Target Files:**
- `concepts/07_LLM_Integration.md`
- `concepts/07_LLM_Integration_Practice.md`

**Learning Objectives:**

**Part 1: Provider SDKs**
- Installing and configuring Anthropic SDK
- Installing and configuring OpenAI SDK
- API key management (environment variables, secrets)
- Making basic API calls
- Handling responses and extracting content
- Error handling (rate limits, timeouts, API errors)

**Part 2: Unified Provider Interface**
- Creating a provider abstraction layer (base class pattern)
- Normalizing request/response formats across providers
- Parameter translation (max_tokens, temperature differences)
- Factory pattern for provider selection
- Provider-specific quirks and gotchas

**Part 3: Multi-Provider Resilience**
- Implementing provider failover
- Error classification (retriable vs non-retriable)
- Timeout handling for LLM calls
- Logging provider performance

**What You'll Be Able to Build:**
- Chat endpoint that works with multiple LLM providers
- Provider switching via API parameter
- Automatic failover when one provider fails

**Key Questions You Should Be Able to Answer:**
1. Why should you create a provider abstraction layer instead of using SDKs directly?
2. How do you handle rate limits from LLM providers gracefully?
3. What's the difference between Anthropic's and OpenAI's message format?
4. How do you implement timeout for potentially slow LLM calls?

**Hands-On Exercises:**
- Exercise 1: Anthropic SDK integration
- Exercise 2: OpenAI SDK integration
- Exercise 3: Creating unified provider interface
- Exercise 4: Provider selection endpoint
- Exercise 5: Error handling for API failures
- Exercise 6: Multi-provider chat API with fallback

**GenAI Application:** LLM provider integration is the core of any GenAI app. A unified interface lets you switch providers for cost optimization, handle outages, and compare model outputs without changing your application code.

**Estimated Time:** 3 hours concepts + 4 hours hands-on

---

### 📋 Topic 8: Streaming Responses (SSE)

**Target Files:**
- `concepts/08_Streaming_SSE.md`
- `concepts/08_Streaming_SSE_Practice.md`

**Learning Objectives:**

**Part 1: Why Streaming Matters**
- User experience: words appearing in real-time vs waiting 10+ seconds
- Perceived latency reduction
- Memory efficiency for long responses
- When streaming is required vs optional

**Part 2: Server-Sent Events (SSE)**
- SSE format (data: {json}\n\n)
- StreamingResponse in FastAPI
- Async generators (yield in async functions)
- Setting correct headers (text/event-stream, Cache-Control)

**Part 3: LLM Streaming Integration**
- Streaming with Anthropic SDK (messages.stream)
- Streaming with OpenAI SDK (stream=True)
- Extracting tokens from stream
- Handling stream errors mid-response

**Part 4: Streaming with Persistence**
- Accumulating streamed content for database storage
- Getting final token counts after stream completes
- Saving messages after streaming finishes
- Error recovery during streaming

**Part 5: Frontend Consumption**
- JavaScript EventSource / fetch with ReadableStream
- Parsing SSE chunks on the client
- Handling stream completion

**What You'll Be Able to Build:**
- Streaming chat endpoints with real-time word output
- Combined streaming + database persistence
- Error-resilient streaming implementation

**Key Questions You Should Be Able to Answer:**
1. Why does streaming dramatically improve user experience for LLM apps?
2. What's the SSE message format and why is it used?
3. How do you save a streamed response to the database after completion?
4. How do you handle errors that occur mid-stream?

**Hands-On Exercises:**
- Exercise 1: Basic StreamingResponse with async generator
- Exercise 2: Stream from Anthropic SDK
- Exercise 3: Stream from OpenAI SDK
- Exercise 4: Add database persistence after stream completes
- Exercise 5: Handle mid-stream errors gracefully
- Exercise 6: Test streaming with a simple frontend

**GenAI Application:** Streaming is table-stakes for chat interfaces. Users expect to see responses appear word-by-word like ChatGPT/Claude, not wait 10 seconds for a wall of text.

**Estimated Time:** 2 hours concepts + 3 hours hands-on

---

### Mini-Project 1: Multi-Provider Chat API

**Target Directory:** `genai_projects/01_multi_provider_chat/`

**Objective:** Build a chat API with multiple LLM provider support, streaming responses, and conversation history.

**Features:**
- Multi-provider support (Anthropic, OpenAI)
- Streaming responses with SSE
- Conversation history storage (PostgreSQL)
- Provider failover (try next provider if one fails)
- Token usage tracking per message
- Message persistence during streaming

**Endpoints:**
- `POST /conversations` - Create new conversation
- `GET /conversations` - List user's conversations
- `GET /conversations/{id}` - Get conversation with messages
- `POST /conversations/{id}/chat` - Send message (non-streaming)
- `POST /conversations/{id}/stream` - Send message (streaming)
- `DELETE /conversations/{id}` - Delete conversation

**Tech Stack:**
- FastAPI with async
- PostgreSQL + async SQLAlchemy
- Anthropic SDK + OpenAI SDK
- Redis for caching (optional)
- Docker Compose for local development

**Deliverables:**
- Working multi-provider chat API
- Database schema with migrations
- Streaming endpoint implementation
- README with setup instructions
- docker-compose.yml for local development

**Estimated Time:** 6-8 hours

---

## Phase 4: Security & Cost Control

**Goal:** Protect your API and control LLM costs.

---

### 📋 Topic 9: Authentication (API Keys + JWT)

**Target Files:**
- `concepts/09_Authentication.md`
- `concepts/09_Authentication_Practice.md`

**Learning Objectives:**

**Part 1: Authentication Strategies**
- API keys for programmatic access (apps calling your API)
- JWT for user sessions (web/mobile apps)
- When to use each approach
- Stateless authentication for horizontal scaling

**Part 2: API Key Implementation**
- Generating secure API keys (secrets.token_urlsafe)
- Hashing keys before storage (never store raw keys)
- Header-based authentication (X-API-Key or Authorization: Bearer)
- Validating API keys with database lookup

**Part 3: JWT Implementation**
- JWT structure (header, payload, signature)
- Creating tokens with python-jose
- Setting expiration times
- Validating tokens in dependencies
- Handling expired/invalid tokens

**Part 4: FastAPI Security Utilities**
- OAuth2PasswordBearer for JWT
- APIKeyHeader for API keys
- Dependency injection patterns for auth
- Protecting endpoints with Depends()

**What You'll Be Able to Build:**
- User registration and login system
- API key generation for developers
- Protected endpoints requiring authentication
- Token refresh flows

**Key Questions You Should Be Able to Answer:**
1. Why should you hash API keys before storing them?
2. What's the difference between API key and JWT authentication?
3. How do JWT tokens enable stateless authentication?
4. When should you use API keys vs JWT tokens?

**Hands-On Exercises:**
- Exercise 1: Generate and hash API keys
- Exercise 2: Create API key validation dependency
- Exercise 3: Implement JWT token creation
- Exercise 4: Build JWT validation with expiration handling
- Exercise 5: Create login and signup endpoints
- Exercise 6: Protect chat endpoints with authentication

**GenAI Application:** Authentication identifies who is making requests—essential for usage tracking, rate limiting by user, and billing. API keys are common for developer integrations, while JWT works for end-user applications.

**Estimated Time:** 3 hours concepts + 3 hours hands-on

---

### 📋 Topic 10: Rate Limiting & Usage Tracking

**Target Files:**
- `concepts/10_Rate_Limiting.md`
- `concepts/10_Rate_Limiting_Practice.md`

**Learning Objectives:**

**Part 1: Why Rate Limiting Is Critical**
- LLM API calls cost real money ($0.003-$0.06 per 1K tokens)
- One malicious user can cost you $1000s in hours
- Fair resource allocation across users
- Preventing abuse while allowing legitimate use

**Part 2: Rate Limiting Algorithms**
- Fixed window (simple but bursty)
- Sliding window with Redis (smoother limiting)
- Token bucket (allows bursting within limits)
- Choosing the right algorithm for your use case

**Part 3: Redis Implementation**
- Sliding window rate limiting with sorted sets
- Atomic operations for accurate counting
- Rate limit headers (X-RateLimit-Remaining, Retry-After)
- Handling 429 Too Many Requests responses

**Part 4: Token Budget Management**
- Tracking tokens used per user (daily/monthly)
- Estimating tokens before making LLM calls
- Enforcing token budgets
- Usage analytics endpoint

**Part 5: Cost Calculation**
- Token costs by provider and model
- Calculating cost per request
- Budget enforcement and alerts
- Exposing usage data to users

**What You'll Be Able to Build:**
- Rate limiting by requests per hour per user
- Token budget enforcement (monthly limits)
- Usage tracking dashboard data
- Cost estimation per request

**Key Questions You Should Be Able to Answer:**
1. Why is token-based rate limiting more important than request-based for GenAI?
2. How does sliding window rate limiting work with Redis?
3. What headers should you return when rate limiting a request?
4. How do you estimate tokens before making an LLM call?

**Hands-On Exercises:**
- Exercise 1: Implement fixed window rate limiting
- Exercise 2: Build sliding window limiter with Redis
- Exercise 3: Add rate limit headers to responses
- Exercise 4: Create token tracking per user
- Exercise 5: Implement cost calculation
- Exercise 6: Build usage analytics endpoint

**GenAI Application:** Rate limiting and usage tracking are non-negotiable for production GenAI apps. Without them, a single user can bankrupt your LLM API budget in hours.

**Estimated Time:** 3 hours concepts + 3 hours hands-on

---

### Mini-Project 2: Secure Chat with Usage Limits

**Target Directory:** `genai_projects/02_secure_chat/`

**Objective:** Add authentication, rate limiting, and usage tracking to your multi-provider chat API.

**Features:**
- API key authentication
- JWT authentication for web users
- Rate limiting (requests per hour)
- Token budget per user
- Usage tracking and analytics
- Cost estimation per request

**New Endpoints:**
- `POST /auth/register` - Create account
- `POST /auth/login` - Get JWT token
- `POST /auth/api-keys` - Generate API key
- `GET /usage` - Get usage statistics
- `GET /usage/history` - Get usage over time

**Tech Stack:**
- FastAPI with OAuth2PasswordBearer
- JWT with python-jose
- Redis for rate limiting
- PostgreSQL for users, API keys, usage logs
- bcrypt for password hashing

**Deliverables:**
- Secure chat API with auth
- Rate limiting system
- Usage tracking and analytics
- API documentation with auth examples
- Test suite with auth fixtures

**Estimated Time:** 6-8 hours

---

## Phase 5: GenAI Patterns

**Goal:** Build sophisticated GenAI features.

---

### 📋 Topic 11: GenAI-Specific Patterns

**Target Files:**
- `concepts/11_GenAI_Patterns.md`
- `concepts/11_GenAI_Patterns_Practice.md`

**Learning Objectives:**

**Part 1: Context Window Management**
- Token counting with tiktoken
- Calculating tokens for conversations
- Truncating conversations to fit context windows
- Keeping system prompts while trimming history
- Strategies for long conversations

**Part 2: Response Caching**
- When to cache LLM responses (low temperature, deterministic)
- Creating cache keys from request parameters
- Cache invalidation strategies
- TTL decisions for different use cases

**Part 3: Prompt Templates**
- Using Jinja2 for dynamic prompts
- System prompt management
- Context injection for RAG
- Template versioning

**Part 4: Conversation Patterns**
- Maintaining conversation state
- System prompt best practices
- Multi-turn conversation handling
- Conversation summarization for long contexts

**What You'll Be Able to Build:**
- Context window manager that truncates intelligently
- Caching layer for deterministic queries
- Templated prompt system
- Long conversation handling

**Key Questions You Should Be Able to Answer:**
1. How do you count tokens for a conversation?
2. When is it safe to cache LLM responses?
3. Why should you always keep system prompts when truncating?
4. How do you handle conversations that exceed the context window?

**Hands-On Exercises:**
- Exercise 1: Implement token counting for messages
- Exercise 2: Build conversation truncation that preserves system prompt
- Exercise 3: Create request hashing for cache keys
- Exercise 4: Implement response caching with Redis
- Exercise 5: Build a prompt template system with Jinja2
- Exercise 6: Handle long conversations with summarization

**GenAI Application:** Context window limits are a constant constraint. Smart truncation and caching strategies are essential for production apps that handle long conversations.

**Estimated Time:** 3 hours concepts + 3 hours hands-on

---

### 📋 Topic 12: Function Calling & Tool Use

**Target Files:**
- `concepts/12_Function_Calling.md`
- `concepts/12_Function_Calling_Practice.md`

**Learning Objectives:**

**Part 1: What Is Function Calling?**
- LLMs deciding to call functions you define
- Use cases: database queries, API calls, calculations
- OpenAI function calling format
- Anthropic tool use format

**Part 2: Defining Tools**
- Tool/function schemas (name, description, parameters)
- JSON Schema for parameter definitions
- Required vs optional parameters
- Good tool descriptions that help the LLM

**Part 3: Implementing the Execution Layer**
- Detecting when the LLM wants to use a tool
- Executing tools and formatting results
- Returning tool results to the LLM
- Error handling in tool execution

**Part 4: Multi-Turn Tool Use**
- Agentic loops (call LLM → execute tool → call LLM again)
- Maximum iteration limits (preventing infinite loops)
- Conversation state with tool call history
- Detecting when the task is complete

**Part 5: Common Tool Patterns**
- Database query tools
- API calling tools (weather, search)
- Calculator tools
- File/document retrieval tools

**What You'll Be Able to Build:**
- Chat endpoint with tool use
- Multi-step reasoning agent
- Database query assistant
- API-integrated chatbot

**Key Questions You Should Be Able to Answer:**
1. What's the difference between function calling and RAG?
2. How do you prevent infinite loops in agentic systems?
3. What makes a good tool description?
4. How do you handle errors when tools fail?

**Hands-On Exercises:**
- Exercise 1: Define a calculator tool
- Exercise 2: Implement OpenAI function calling
- Exercise 3: Implement Anthropic tool use
- Exercise 4: Build a tool execution layer
- Exercise 5: Create a multi-turn agentic loop
- Exercise 6: Handle tool errors gracefully
- Exercise 7: Build a database query tool

**GenAI Application:** Function calling transforms LLMs from text generators into agents that can take actions. This enables database queries, API integrations, and multi-step workflows.

**Estimated Time:** 3 hours concepts + 4 hours hands-on

---

### 📋 Topic 13: Vector Databases & RAG

**Target Files:**
- `concepts/13_Vector_Databases_RAG.md`
- `concepts/13_Vector_Databases_RAG_Practice.md`

**Learning Objectives:**

**Part 1: RAG Fundamentals**
- What is Retrieval-Augmented Generation?
- When to use RAG (your own data, reducing hallucination)
- The RAG pipeline: query → retrieve → augment → generate
- RAG vs fine-tuning vs function calling

**Part 2: Embeddings**
- What are embeddings (vector representations of text)?
- Embedding models (OpenAI text-embedding-3-small, etc.)
- Generating embeddings via API
- Embedding dimensions and similarity metrics

**Part 3: Vector Database Integration**
- Qdrant basics (self-hosted or cloud)
- Creating collections
- Upserting vectors with metadata
- Similarity search queries
- Filtering with metadata

**Part 4: Document Processing**
- Chunking strategies (by sentences, paragraphs, tokens)
- Overlap for context preservation
- Metadata extraction
- Indexing documents with chunks

**Part 5: RAG Implementation**
- Embedding user queries
- Retrieving relevant chunks
- Injecting context into prompts
- Source citations in responses

**What You'll Be Able to Build:**
- Document upload and indexing pipeline
- Semantic search endpoint
- RAG-powered Q&A chat
- Source citation in responses

**Key Questions You Should Be Able to Answer:**
1. When should you use RAG vs fine-tuning?
2. How do you choose chunk size and overlap?
3. Why do you embed the query the same way as documents?
4. How do you cite sources in RAG responses?

**Hands-On Exercises:**
- Exercise 1: Generate embeddings with OpenAI API
- Exercise 2: Set up Qdrant and create a collection
- Exercise 3: Index documents with chunking
- Exercise 4: Implement similarity search
- Exercise 5: Build RAG prompt template with context injection
- Exercise 6: Create RAG chat endpoint
- Exercise 7: Add source citations to responses

**GenAI Application:** RAG enables your chatbot to answer questions about your documents, products, or knowledge base. It's the most common pattern for enterprise GenAI applications.

**Estimated Time:** 4 hours concepts + 4 hours hands-on

---

### Mini-Project 3: RAG System or Agent

**Target Directory:** `genai_projects/03_rag_or_agent/`

**Objective:** Build either a RAG document Q&A system or an agentic assistant with tools.

**Option A: RAG Document Q&A**

**Features:**
- Document upload and indexing
- Chunking and embedding generation
- Semantic search
- Context-augmented chat
- Source citations in responses

**Option B: Agentic Assistant**

**Features:**
- Tool definitions (weather, search, calculator)
- Agentic loop with tool execution
- Multi-turn conversations with tools
- Tool execution logging
- Iteration limits for safety

**Both Options Include:**
- All features from Mini-Projects 1 & 2
- PostgreSQL for metadata/history
- Redis for caching
- Authentication and rate limiting
- Comprehensive test suite

**Tech Stack:**
- FastAPI with async
- Qdrant for vector search (Option A)
- OpenAI embeddings
- Anthropic/OpenAI for generation
- PostgreSQL + Redis
- pytest

**Estimated Time:** 8-10 hours

---

## Phase 6: Ship It

**Goal:** Get your app running in production.

---

### 📋 Topic 14: Docker for Deployment

**Target Files:**
- `concepts/14_Docker_Deployment.md`
- `concepts/14_Docker_Deployment_Practice.md`

**Learning Objectives:**

**Part 1: Production Dockerfile**
- Choosing base images (python:3.11-slim)
- Environment variables for configuration
- Installing dependencies efficiently
- Copying application code
- Non-root user for security
- Health check instructions

**Part 2: Docker Best Practices**
- .dockerignore to exclude unnecessary files
- Layer ordering for caching (dependencies before code)
- Avoiding secrets in images
- Image size optimization

**Part 3: Docker Compose for Development**
- Multi-service setup (app, database, redis)
- Environment variable management
- Volume mounts for development
- Service dependencies (depends_on)
- Port mapping

**Part 4: Health Checks**
- Why health checks matter
- FastAPI health check endpoint
- Checking database and Redis connectivity
- Docker HEALTHCHECK instruction

**What You'll Be Able to Build:**
- Production-ready Dockerfile
- Local development environment with Docker Compose
- Health check endpoint for container orchestration

**Key Questions You Should Be Able to Answer:**
1. Why run containers as non-root user?
2. How does layer ordering affect build caching?
3. What should a health check endpoint verify?
4. How do you manage secrets without putting them in the image?

**Hands-On Exercises:**
- Exercise 1: Write a production Dockerfile
- Exercise 2: Create .dockerignore file
- Exercise 3: Build docker-compose.yml for local development
- Exercise 4: Implement health check endpoint
- Exercise 5: Test local development with Docker Compose

**GenAI Application:** Docker packages your app with all dependencies, ensuring it runs the same way locally and in production. It's the standard for deploying FastAPI applications.

**Estimated Time:** 2 hours concepts + 2 hours hands-on

---

### 📋 Topic 15: Cloud Deployment

**Target Files:**
- `concepts/15_Cloud_Deployment.md`
- `concepts/15_Cloud_Deployment_Practice.md`

**Learning Objectives:**

**Part 1: Platform Comparison**
- Railway: easiest, good free tier
- Render: simple, good for small apps
- Fly.io: more control, great performance
- Choosing based on your needs

**Part 2: Environment Configuration**
- Pydantic Settings for environment variables
- Required vs optional configuration
- Validating config on startup
- Multi-environment support (dev, staging, prod)

**Part 3: Deployment Process**
- Connecting to git repository
- Setting environment variables securely
- Database provisioning
- Redis setup
- Custom domains

**Part 4: Pre-Deployment Checklist**
- Environment variables set
- Database migrations run
- Health check endpoint working
- CORS configured for frontend
- Rate limiting enabled
- Error logging configured
- API keys have usage limits

**Part 5: Monitoring & Debugging**
- Viewing logs in production
- Debugging deployment failures
- Rollback strategies
- Basic alerting

**What You'll Be Able to Build:**
- Deployed GenAI application on Railway/Render
- Production environment configuration
- CI/CD from git push to deployment

**Key Questions You Should Be Able to Answer:**
1. How do you manage secrets in cloud deployments?
2. What should you check before deploying to production?
3. How do you debug a deployment that's failing?
4. What's the simplest path from development to production?

**Hands-On Exercises:**
- Exercise 1: Set up Railway CLI
- Exercise 2: Deploy a FastAPI app to Railway
- Exercise 3: Configure environment variables
- Exercise 4: Provision PostgreSQL and Redis
- Exercise 5: Set up custom domain (optional)
- Exercise 6: Verify health checks and logging

**GenAI Application:** Simple cloud deployment lets you ship fast. Railway and Render handle infrastructure so you can focus on your GenAI features, not DevOps.

**Estimated Time:** 2 hours concepts + 2 hours hands-on

---

### Final Project: Deployed GenAI Application

**Target Directory:** `genai_projects/final_project/`

**Objective:** Build and deploy a complete GenAI application—your portfolio piece.

**Core Features:**
- Multi-provider LLM chat (Anthropic + OpenAI)
- Streaming responses
- Conversation history
- User authentication (JWT + API keys)
- Rate limiting and usage tracking
- RAG or tool use (your choice)

**Production Requirements:**
- Deployed on Railway/Render/Fly.io
- PostgreSQL database
- Redis for caching/rate limiting
- Health check endpoint
- Environment-based configuration
- Basic error logging

**Deliverables:**
1. Working deployed application
2. API documentation (auto-generated /docs)
3. README with setup instructions
4. Architecture diagram (simple)

**Success Criteria:**
- Application is publicly accessible
- All core features working
- Can handle multiple concurrent users
- Errors are handled gracefully
- Usage is tracked per user

**This is your portfolio piece.** Make it something you'd be proud to show to potential employers or investors.

**Estimated Time:** 10-12 hours

---

## What's NOT in This Roadmap (And Why)

| Topic | Why It's Not Included |
|-------|----------------------|
| **Kubernetes** | Railway/Render handle scaling. Learn K8s when you actually need it. |
| **NGINX** | Cloud platforms provide load balancing. Don't add complexity. |
| **Prometheus/Grafana** | Start with basic logging. Add observability when you have users. |
| **80%+ test coverage** | Test critical paths. Ship fast, fix bugs as users find them. |
| **Multi-stage Docker builds** | Optimization you don't need until your images are huge. |
| **Distributed tracing** | Overkill for single-service apps. Add when you go microservices. |
| **LangChain/LangServe** | Adds abstraction you don't need. Raw SDKs are simpler to debug. |
| **CI/CD pipelines** | Railway/Render auto-deploy from git. Add GitHub Actions later. |

**The Goal:** Ship a working product. These topics become relevant when you have users and need to scale.

---

## After This Roadmap

**When you have users and need to level up:**

| When You Need | Learn |
|---------------|-------|
| Handle 1000+ requests/second | Kubernetes, horizontal scaling |
| Debug production issues | Distributed tracing (OpenTelemetry) |
| Complex deployment pipelines | CI/CD, blue-green deployments |
| Detailed performance metrics | Prometheus + Grafana |
| Multiple services | API gateway, service mesh |

**But don't learn these before you need them.** Premature optimization is the root of all evil.

---

## Essential Patterns Summary

This section provides a high-level overview of patterns covered in the concept guides. See the individual concept files for implementation details.

### LLM API Call Pattern
- Use async clients for non-blocking calls
- Add timeouts to prevent hanging (30s default)
- Handle rate limits with exponential backoff
- Wrap provider-specific errors in custom exceptions

### Streaming Response Pattern
- Use StreamingResponse with async generators
- SSE format: `data: {json}\n\n`
- Accumulate content for database persistence after streaming
- Handle client disconnection gracefully

### Protected Endpoint Pattern
- Verify authentication (API key or JWT)
- Check rate limits before processing
- Check token budget before LLM calls
- Update usage tracking after successful responses

### Provider Abstraction Pattern
- Base class defining common interface
- Concrete implementations per provider
- Factory for provider instantiation
- Normalized request/response models

---

## Success Metrics

By the end of this roadmap, you will have:

- [ ] Built a multi-provider LLM chat API
- [ ] Implemented streaming responses
- [ ] Set up PostgreSQL + Redis
- [ ] Added authentication and rate limiting
- [ ] Built a RAG system or agentic assistant
- [ ] Deployed to production
- [ ] A portfolio piece to show employers/investors

**Total Time:** ~60-80 hours over 6 weeks

**You're ready to ship GenAI products. Go build something.**

---

## Resources

**Official Docs:**
- [FastAPI](https://fastapi.tiangolo.com/)
- [Anthropic API](https://docs.anthropic.com/)
- [OpenAI API](https://platform.openai.com/docs/)
- [Qdrant](https://qdrant.tech/documentation/)

**Deployment:**
- [Railway](https://railway.app/)
- [Render](https://render.com/)
- [Fly.io](https://fly.io/)

**This roadmap is opinionated.** It optimizes for shipping, not mastery. When you need deeper knowledge, the Master Roadmap is there. But first—ship something.
