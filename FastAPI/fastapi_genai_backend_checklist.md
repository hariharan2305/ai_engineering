# FastAPI for GenAI Backends: Learning Checklist

## Purpose
Master building production-grade backends for GenAI applications using FastAPI, with multi-provider LLM support (Anthropic, OpenAI, LiteLLM, OpenRouter), all deployable via Docker.

---

# SECTION 1: FastAPI Fundamentals

## 1.1 HTTP & REST API Basics (Prerequisite)

**Learning Objectives:**
- Understand HTTP methods (GET, POST, PUT, DELETE) and when to use each
- Know HTTP status codes (200, 201, 400, 401, 404, 500) and their meanings
- Understand request/response structure (headers, body, query params, path params)
- Know what REST is and RESTful API design principles

**Checklist:**
- [ ] Can explain the difference between GET and POST
- [ ] Can describe what a 401 vs 403 vs 404 error means
- [ ] Can explain path parameters vs query parameters vs request body
- [ ] Can describe what makes an API "RESTful"

**Resources:**
- MDN HTTP Overview: https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview
- REST API Tutorial: https://restfulapi.net/

---

## 1.2 FastAPI Core Concepts

**Learning Objectives:**
- Create a FastAPI application instance
- Define path operations (endpoints) with decorators
- Handle path parameters, query parameters, and request bodies
- Return proper HTTP responses with status codes

**Subtopics:**

### 1.2.1 Application Setup
- [ ] Creating a FastAPI app instance
- [ ] Running with Uvicorn (ASGI server)
- [ ] Understanding automatic interactive docs (`/docs`, `/redoc`)

### 1.2.2 Path Operations
- [ ] `@app.get()`, `@app.post()`, `@app.put()`, `@app.delete()` decorators
- [ ] Path parameters: `/items/{item_id}`
- [ ] Query parameters: `/items?skip=0&limit=10`
- [ ] Combining path and query parameters

### 1.2.3 Request Body Handling
- [ ] Accepting JSON request bodies
- [ ] Multiple body parameters
- [ ] Optional and required fields

### 1.2.4 Response Handling
- [ ] Returning dictionaries (auto-converted to JSON)
- [ ] Setting status codes
- [ ] Custom response types

**After completing, you can answer:**
- How do I create an endpoint that accepts user input?
- How do I return different status codes for success vs error?
- What's the difference between path params, query params, and body?

**Resources:**
- FastAPI Tutorial - First Steps: https://fastapi.tiangolo.com/tutorial/first-steps/
- FastAPI Tutorial - Path Parameters: https://fastapi.tiangolo.com/tutorial/path-params/
- FastAPI Tutorial - Query Parameters: https://fastapi.tiangolo.com/tutorial/query-params/
- FastAPI Tutorial - Request Body: https://fastapi.tiangolo.com/tutorial/body/

---

## 1.3 Pydantic Models & Validation

**Learning Objectives:**
- Define data models using Pydantic BaseModel
- Add field validation (min/max, regex, custom validators)
- Use models for request validation and response serialization
- Handle validation errors gracefully

**Subtopics:**

### 1.3.1 Basic Models
- [ ] Creating a class that inherits from `BaseModel`
- [ ] Defining fields with type hints
- [ ] Optional vs required fields
- [ ] Default values

### 1.3.2 Field Validation
- [ ] Using `Field()` for constraints (min, max, regex)
- [ ] Custom validators with `@field_validator`
- [ ] Model-level validation with `@model_validator`

### 1.3.3 Nested Models
- [ ] Models containing other models
- [ ] Lists of models
- [ ] Optional nested models

### 1.3.4 Request/Response Models
- [ ] Using models as request body type hints
- [ ] Using `response_model` parameter
- [ ] Different models for input vs output

**After completing, you can answer:**
- How do I ensure a field is between 0 and 100?
- How do I validate that an email field is actually an email?
- How do I define complex nested JSON structures?

**Resources:**
- Pydantic V2 Documentation: https://docs.pydantic.dev/latest/
- FastAPI - Request Body with Pydantic: https://fastapi.tiangolo.com/tutorial/body/
- FastAPI - Response Model: https://fastapi.tiangolo.com/tutorial/response-model/

---

## 1.4 Dependency Injection

**Learning Objectives:**
- Understand what dependency injection is and why it matters
- Create reusable dependencies with `Depends()`
- Chain dependencies (dependencies that depend on other dependencies)
- Use dependencies for common patterns (auth, database connections)

**Subtopics:**

### 1.4.1 Basic Dependencies
- [ ] Creating a function dependency
- [ ] Using `Depends()` in path operation parameters
- [ ] Dependencies with parameters

### 1.4.2 Dependency Chains
- [ ] Sub-dependencies (dependencies of dependencies)
- [ ] Sharing dependencies across multiple endpoints

### 1.4.3 Common Patterns
- [ ] Database session dependency
- [ ] Current user dependency (for auth)
- [ ] Configuration dependency

**After completing, you can answer:**
- How do I share a database connection across endpoints?
- How do I create a reusable authentication check?
- How do I avoid repeating the same setup code in every endpoint?

**Resources:**
- FastAPI - Dependencies: https://fastapi.tiangolo.com/tutorial/dependencies/
- FastAPI - Dependencies with yield: https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/

---

# SECTION 2: Async Programming for AI Backends

## 2.1 Python Async Fundamentals

**Learning Objectives:**
- Understand the difference between sync and async code
- Know when async provides benefits (I/O-bound vs CPU-bound)
- Write async functions with `async def` and `await`
- Run multiple async operations concurrently

**Subtopics:**

### 2.1.1 Sync vs Async Mental Model
- [ ] What is blocking I/O and why it's slow
- [ ] How async enables concurrency (not parallelism)
- [ ] The event loop concept
- [ ] When async helps vs when it doesn't

### 2.1.2 async/await Syntax
- [ ] Defining async functions with `async def`
- [ ] Calling async functions with `await`
- [ ] Why you can't call `await` in regular functions

### 2.1.3 Concurrent Operations
- [ ] `asyncio.gather()` for running multiple tasks
- [ ] `asyncio.create_task()` for background tasks
- [ ] Handling exceptions in concurrent operations

**After completing, you can answer:**
- When should I use `async def` vs regular `def` in FastAPI?
- How do I call multiple LLM APIs at the same time?
- Why is async good for API calls but not for math calculations?

**Resources:**
- FastAPI - Async: https://fastapi.tiangolo.com/async/
- Real Python - Async IO: https://realpython.com/async-io-python/
- Python asyncio docs: https://docs.python.org/3/library/asyncio.html

---

## 2.2 Async HTTP Clients (httpx)

**Learning Objectives:**
- Make async HTTP requests with httpx
- Handle timeouts and errors
- Manage client lifecycle (connection pooling)
- Stream responses from external APIs

**Subtopics:**

### 2.2.1 Basic Async Requests
- [ ] Creating an `AsyncClient`
- [ ] Making GET, POST requests
- [ ] Passing headers, JSON body, query params

### 2.2.2 Client Lifecycle
- [ ] Using `async with` context manager
- [ ] Connection pooling benefits
- [ ] Reusing clients vs creating new ones

### 2.2.3 Error Handling
- [ ] Setting timeouts
- [ ] Handling `TimeoutException`
- [ ] Handling `HTTPStatusError`
- [ ] Retry strategies

### 2.2.4 Streaming Responses
- [ ] Using `client.stream()` for large responses
- [ ] Iterating over response chunks
- [ ] Memory-efficient processing

**After completing, you can answer:**
- How do I make an async API call with a 30-second timeout?
- How do I reuse HTTP connections efficiently?
- How do I handle streaming responses from an API?

**Resources:**
- httpx Documentation: https://www.python-httpx.org/
- httpx Async: https://www.python-httpx.org/async/

---

## 2.3 FastAPI Async Patterns

**Learning Objectives:**
- Use async in FastAPI path operations correctly
- Manage async resources with lifespan events
- Implement background tasks
- Avoid common async pitfalls

**Subtopics:**

### 2.3.1 Async Path Operations
- [ ] When FastAPI runs `async def` vs `def` differently
- [ ] Mixing async and sync in the same app
- [ ] Avoiding blocking the event loop

### 2.3.2 Lifespan Events
- [ ] Using `@asynccontextmanager` with `lifespan` parameter
- [ ] Startup: initializing async resources (HTTP clients, DB pools)
- [ ] Shutdown: cleaning up resources properly

### 2.3.3 Background Tasks
- [ ] Using `BackgroundTasks` for fire-and-forget operations
- [ ] Limitations of background tasks
- [ ] When to use a task queue instead

**After completing, you can answer:**
- Where do I create my httpx client so it's shared across requests?
- How do I run something after returning a response to the user?
- What happens if I use a blocking library in an async endpoint?

**Resources:**
- FastAPI - Lifespan Events: https://fastapi.tiangolo.com/advanced/events/
- FastAPI - Background Tasks: https://fastapi.tiangolo.com/tutorial/background-tasks/

---

# SECTION 3: LLM Provider Integration

## 3.1 Anthropic Claude SDK

**Learning Objectives:**
- Set up the Anthropic Python SDK
- Make basic completion requests
- Handle streaming responses
- Understand Claude-specific features (system prompts, message format)

**Subtopics:**

### 3.1.1 SDK Setup
- [ ] Installing `anthropic` package
- [ ] API key configuration (environment variables)
- [ ] Creating a client instance

### 3.1.2 Messages API
- [ ] Message format (role: user/assistant)
- [ ] System prompts
- [ ] Setting `max_tokens`, `temperature`
- [ ] Response structure

### 3.1.3 Streaming
- [ ] Using `client.messages.stream()`
- [ ] Iterating over `text_stream`
- [ ] Handling stream events (message_start, content_block_delta, message_stop)

### 3.1.4 Error Handling
- [ ] Rate limit errors (429)
- [ ] API errors
- [ ] Timeout handling

**After completing, you can answer:**
- How do I get Claude to respond in a streaming fashion?
- How do I set a system prompt for Claude?
- What does the streaming event format look like?

**Resources:**
- Anthropic Python SDK: https://github.com/anthropics/anthropic-sdk-python
- Anthropic API Docs: https://docs.anthropic.com/en/api/messages
- Streaming Docs: https://docs.anthropic.com/en/api/streaming

---

## 3.2 OpenAI SDK

**Learning Objectives:**
- Set up the OpenAI Python SDK
- Make chat completion requests
- Handle streaming responses
- Understand OpenAI-specific features (function calling, vision)

**Subtopics:**

### 3.2.1 SDK Setup
- [ ] Installing `openai` package
- [ ] API key configuration
- [ ] Creating a client instance

### 3.2.2 Chat Completions API
- [ ] Message format (role: system/user/assistant)
- [ ] Model selection (gpt-4, gpt-4-turbo, gpt-3.5-turbo)
- [ ] Parameters (max_tokens, temperature, top_p)

### 3.2.3 Streaming
- [ ] Using `stream=True` parameter
- [ ] Iterating over chunks
- [ ] Extracting delta content

### 3.2.4 OpenAI-Specific Features
- [ ] Function calling / tool use
- [ ] Vision (image inputs)
- [ ] JSON mode

**After completing, you can answer:**
- How is OpenAI's message format different from Anthropic's?
- How do I stream tokens from GPT-4?
- How do I use function calling?

**Resources:**
- OpenAI Python SDK: https://github.com/openai/openai-python
- OpenAI API Reference: https://platform.openai.com/docs/api-reference/chat

---

## 3.3 LiteLLM (Multi-Provider Abstraction)

**Learning Objectives:**
- Understand why provider abstraction is valuable
- Set up LiteLLM for multi-provider access
- Configure fallbacks and retries
- Use the Router for load balancing

**Subtopics:**

### 3.3.1 Why LiteLLM
- [ ] Single interface to 100+ providers
- [ ] Easy provider switching
- [ ] Automatic fallback on failures
- [ ] Cost optimization

### 3.3.2 Basic Usage
- [ ] Installing `litellm`
- [ ] `litellm.completion()` function
- [ ] Model naming convention (provider/model)
- [ ] Async support with `acompletion()`

### 3.3.3 Router Configuration
- [ ] Setting up multiple providers
- [ ] Configuring fallbacks
- [ ] Retry settings
- [ ] Load balancing strategies

### 3.3.4 Streaming with LiteLLM
- [ ] `stream=True` parameter
- [ ] Iterating over chunks (same format for all providers)
- [ ] Async streaming

**After completing, you can answer:**
- How do I call Claude and GPT-4 with the same code?
- How do I automatically fall back to GPT-4 if Claude fails?
- How do I add OpenRouter models to my application?

**Resources:**
- LiteLLM Documentation: https://docs.litellm.ai/
- LiteLLM Router: https://docs.litellm.ai/docs/routing

---

## 3.4 OpenRouter Integration

**Learning Objectives:**
- Understand OpenRouter's value proposition (200+ models)
- Make API calls to OpenRouter
- Handle OpenRouter-specific requirements
- Compare cost across models

**Subtopics:**

### 3.4.1 OpenRouter Basics
- [ ] What models are available
- [ ] API key setup
- [ ] Required headers (HTTP-Referer)

### 3.4.2 Direct API Usage
- [ ] Endpoint URL structure
- [ ] Request format (OpenAI-compatible)
- [ ] Model naming

### 3.4.3 Via LiteLLM
- [ ] Model prefix (`openrouter/`)
- [ ] Configuration in Router

**After completing, you can answer:**
- How do I access Llama, Mistral, or other open models via API?
- What's the cost difference between providers for the same model?
- How do I add OpenRouter to my LiteLLM setup?

**Resources:**
- OpenRouter Documentation: https://openrouter.ai/docs
- OpenRouter Models: https://openrouter.ai/models

---

# SECTION 4: Streaming Responses

## 4.1 Server-Sent Events (SSE) Fundamentals

**Learning Objectives:**
- Understand what SSE is and how it differs from WebSockets
- Know the SSE message format
- Implement SSE endpoints in FastAPI
- Handle SSE on the client side

**Subtopics:**

### 4.1.1 SSE vs WebSocket
- [ ] One-way (server->client) vs bidirectional
- [ ] When to use SSE (LLM streaming, notifications)
- [ ] When to use WebSocket (chat, real-time collaboration)
- [ ] SSE advantages: simpler, HTTP-based, auto-reconnect

### 4.1.2 SSE Format
- [ ] `text/event-stream` content type
- [ ] Message format: `data: {json}\n\n`
- [ ] Event types with `event:` field
- [ ] Keeping connections alive

### 4.1.3 FastAPI StreamingResponse
- [ ] Using `StreamingResponse` class
- [ ] Async generators for streaming
- [ ] Setting correct headers
- [ ] Error handling in streams

**After completing, you can answer:**
- Why do ChatGPT and Claude use SSE instead of WebSockets?
- What's the exact format of an SSE message?
- How do I send incremental updates to the client?

**Resources:**
- MDN Server-Sent Events: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
- FastAPI StreamingResponse: https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse

---

## 4.2 Implementing LLM Token Streaming

**Learning Objectives:**
- Stream tokens from any LLM provider through FastAPI
- Handle streaming errors gracefully
- Implement proper connection cleanup
- Test streaming endpoints

**Subtopics:**

### 4.2.1 Stream Architecture
- [ ] Endpoint receives request -> calls LLM with stream=True -> yields tokens -> client displays
- [ ] Async generator pattern
- [ ] Back-pressure handling

### 4.2.2 Error Handling in Streams
- [ ] Try/except inside generator
- [ ] Sending error events to client
- [ ] Partial response recovery
- [ ] Timeout handling

### 4.2.3 Testing Streams
- [ ] Using `curl` to test SSE
- [ ] Python client for SSE
- [ ] Browser EventSource testing

**After completing, you can answer:**
- How do I handle an error midway through streaming?
- How do I test a streaming endpoint?
- What happens if the client disconnects mid-stream?

**Resources:**
- Anthropic Streaming: https://docs.anthropic.com/en/api/streaming
- httpx Streaming: https://www.python-httpx.org/quickstart/#streaming-responses

---

## 4.3 WebSocket Basics (For Bidirectional Needs)

**Learning Objectives:**
- Implement WebSocket endpoints in FastAPI
- Handle connection lifecycle (connect, message, disconnect)
- Manage multiple concurrent connections
- Know when WebSocket is actually needed vs SSE

**Subtopics:**

### 4.3.1 WebSocket in FastAPI
- [ ] `@app.websocket()` decorator
- [ ] Accepting connections
- [ ] Sending and receiving messages
- [ ] Handling disconnections

### 4.3.2 Connection Management
- [ ] Tracking active connections
- [ ] Broadcasting to multiple clients
- [ ] Per-user connections

### 4.3.3 When to Use WebSocket
- [ ] Real-time typing indicators
- [ ] Multi-user chat rooms
- [ ] Collaborative editing
- [ ] NOT needed for simple LLM streaming

**After completing, you can answer:**
- How do I maintain persistent bidirectional communication?
- How do I broadcast a message to all connected users?
- Should I use WebSocket or SSE for my use case?

**Resources:**
- FastAPI WebSockets: https://fastapi.tiangolo.com/advanced/websockets/

---

# SECTION 5: Production Patterns

## 5.1 Authentication & Authorization

**Learning Objectives:**
- Implement JWT-based authentication
- Protect endpoints with auth dependencies
- Handle different user roles/permissions
- Secure API keys properly

**Subtopics:**

### 5.1.1 JWT Basics
- [ ] What is a JWT (header, payload, signature)
- [ ] Creating tokens (encoding)
- [ ] Verifying tokens (decoding)
- [ ] Token expiration

### 5.1.2 FastAPI Security
- [ ] `HTTPBearer` security scheme
- [ ] Creating auth dependency
- [ ] Protecting endpoints with `Depends()`

### 5.1.3 Best Practices
- [ ] Never hardcode secrets
- [ ] Environment variables for API keys
- [ ] Token refresh strategies
- [ ] CORS configuration

**After completing, you can answer:**
- How do I protect my API endpoints from unauthorized access?
- How do I extract user information from a JWT?
- How do I handle expired tokens?

**Resources:**
- FastAPI Security: https://fastapi.tiangolo.com/tutorial/security/
- PyJWT Documentation: https://pyjwt.readthedocs.io/

---

## 5.2 Rate Limiting

**Learning Objectives:**
- Understand why rate limiting is critical for AI APIs
- Implement rate limiting with SlowAPI
- Configure different limits for different user tiers
- Handle rate limit errors gracefully

**Subtopics:**

### 5.2.1 Why Rate Limit AI APIs
- [ ] Cost protection (LLM APIs are expensive)
- [ ] Abuse prevention
- [ ] Fair usage across users
- [ ] Upstream provider limits

### 5.2.2 SlowAPI Implementation
- [ ] Setting up SlowAPI with FastAPI
- [ ] Rate limit formats (`100/hour`, `10/minute`)
- [ ] Key functions (by IP, by user)
- [ ] Storage backends (memory, Redis)

### 5.2.3 Tiered Rate Limiting
- [ ] Free vs premium limits
- [ ] Dynamic limits based on user
- [ ] Returning proper 429 responses

**After completing, you can answer:**
- How do I limit users to 100 requests per hour?
- How do I give premium users higher limits?
- What should I return when a user is rate limited?

**Resources:**
- SlowAPI Documentation: https://github.com/laurentS/slowapi

---

## 5.3 Error Handling & Retries

**Learning Objectives:**
- Handle errors from LLM providers gracefully
- Implement retry logic with exponential backoff
- Return meaningful error responses to clients
- Log errors for debugging

**Subtopics:**

### 5.3.1 Common LLM API Errors
- [ ] Rate limits (429)
- [ ] Server errors (500, 502, 503)
- [ ] Timeouts
- [ ] Invalid requests (400)

### 5.3.2 Tenacity for Retries
- [ ] Configuring retry attempts
- [ ] Exponential backoff
- [ ] Which errors to retry
- [ ] Logging retries

### 5.3.3 Error Response Design
- [ ] Consistent error format
- [ ] Appropriate status codes
- [ ] Helpful error messages
- [ ] Retry-After headers

**After completing, you can answer:**
- How do I automatically retry failed LLM calls?
- How long should I wait between retries?
- What errors should I retry vs return immediately?

**Resources:**
- Tenacity Documentation: https://tenacity.readthedocs.io/

---

## 5.4 Logging & Observability

**Learning Objectives:**
- Implement structured logging
- Add request tracing with correlation IDs
- Monitor key metrics for AI APIs
- Debug issues in production

**Subtopics:**

### 5.4.1 Structured Logging
- [ ] JSON logs vs text logs
- [ ] Using structlog
- [ ] Log levels (debug, info, warning, error)
- [ ] What to log (requests, responses, errors, timings)

### 5.4.2 Request Tracing
- [ ] Correlation IDs (request IDs)
- [ ] Middleware for automatic ID injection
- [ ] Passing IDs to LLM calls
- [ ] Finding logs by request ID

### 5.4.3 Key Metrics
- [ ] Request latency
- [ ] Token usage
- [ ] Error rates
- [ ] Cost tracking

**After completing, you can answer:**
- How do I trace a request through my entire system?
- How do I know which requests are slow?
- How do I track my LLM API costs?

**Resources:**
- Structlog Documentation: https://www.structlog.org/

---

# SECTION 6: Docker for Development & Deployment

## 6.1 Docker Fundamentals

**Learning Objectives:**
- Understand containers vs virtual machines
- Write Dockerfiles for Python applications
- Build and run Docker images
- Manage container lifecycle

**Subtopics:**

### 6.1.1 Core Concepts
- [ ] Images vs containers
- [ ] Layers and caching
- [ ] Registries (Docker Hub)

### 6.1.2 Dockerfile for Python/FastAPI
- [ ] Base image selection (`python:3.11-slim`)
- [ ] WORKDIR, COPY, RUN commands
- [ ] Installing dependencies
- [ ] Exposing ports
- [ ] CMD vs ENTRYPOINT

### 6.1.3 Build & Run
- [ ] `docker build` command
- [ ] `docker run` command
- [ ] Port mapping (`-p`)
- [ ] Environment variables (`-e`)
- [ ] Volume mounts (`-v`)

**After completing, you can answer:**
- How do I package my FastAPI app into a container?
- How do I pass API keys to a container?
- How do I access my app running in a container?

**Resources:**
- Docker Getting Started: https://docs.docker.com/get-started/
- Docker Python Guide: https://docs.docker.com/language/python/

---

## 6.2 Docker Compose for Multi-Service Apps

**Learning Objectives:**
- Define multi-container applications
- Manage service dependencies
- Handle environment configuration
- Use volumes for development

**Subtopics:**

### 6.2.1 docker-compose.yml Structure
- [ ] Services definition
- [ ] Port mappings
- [ ] Environment variables
- [ ] Depends_on for ordering

### 6.2.2 Development Workflow
- [ ] Volume mounts for hot reload
- [ ] Watching logs
- [ ] Rebuilding images
- [ ] Stopping and cleaning up

### 6.2.3 Common Services for AI Apps
- [ ] FastAPI application
- [ ] Redis for caching
- [ ] PostgreSQL for data

**After completing, you can answer:**
- How do I run my app with Redis using one command?
- How do I enable hot reload in Docker?
- How do I view logs from all services?

**Resources:**
- Docker Compose Documentation: https://docs.docker.com/compose/

---

# PROJECT CHECKLISTS

## Project 1: FastAPI Hello World (Dockerized)

**Directory:** `ai_engineering/FastAPI/projects/01_fastapi_hello/`

**Goal:** Basic FastAPI app running in Docker

**Components:**
- [ ] FastAPI app with `/health` endpoint returning `{"status": "ok"}`
- [ ] `/echo` POST endpoint that returns the input message
- [ ] Pydantic model for echo request/response
- [ ] Dockerfile
- [ ] docker-compose.yml
- [ ] requirements.txt
- [ ] README with run instructions

**Learning Validation:**
- [ ] Can run `docker-compose up` and access `/docs`
- [ ] Can make requests to both endpoints
- [ ] Can explain what each line in Dockerfile does

---

## Project 2: Single Provider Chat API

**Directory:** `ai_engineering/FastAPI/projects/02_single_provider_chat/`

**Goal:** Chat endpoint calling one LLM provider (Anthropic)

**Components:**
- [ ] `/v1/chat` POST endpoint
- [ ] ChatRequest model (message, max_tokens, temperature)
- [ ] ChatResponse model (content, usage, model)
- [ ] Anthropic SDK integration
- [ ] Environment variable for API key
- [ ] Error handling for API failures
- [ ] Dockerfile with env var support
- [ ] docker-compose.yml with .env file

**Learning Validation:**
- [ ] Can send a message and get a response
- [ ] Can explain the Anthropic message format
- [ ] Understands how API keys are passed to containers

---

## Project 3: Multi-Provider Chat with LiteLLM

**Directory:** `ai_engineering/FastAPI/projects/03_multi_provider_chat/`

**Goal:** Chat endpoint supporting multiple providers with fallback

**Components:**
- [ ] LiteLLM Router configuration
- [ ] Support for Anthropic, OpenAI, OpenRouter
- [ ] Automatic fallback on failure
- [ ] `/v1/chat` endpoint using router
- [ ] Response includes which provider was used
- [ ] `/v1/models` endpoint listing available models
- [ ] Environment variables for all API keys

**Learning Validation:**
- [ ] Can switch providers by changing model name
- [ ] Fallback works when primary provider fails
- [ ] Understands LiteLLM model naming convention

---

## Project 4: Streaming Chat API

**Directory:** `ai_engineering/FastAPI/projects/04_streaming_chat/`

**Goal:** SSE streaming endpoint for real-time token delivery

**Components:**
- [ ] `/v1/chat/stream` POST endpoint returning SSE
- [ ] StreamingResponse with async generator
- [ ] Proper SSE format (`data: {json}\n\n`)
- [ ] Stream completion event
- [ ] Error handling within stream
- [ ] Works with all configured providers
- [ ] Simple HTML test page to visualize streaming

**Learning Validation:**
- [ ] Can see tokens appear one by one
- [ ] Understands SSE message format
- [ ] Can test streaming with curl
- [ ] Knows difference between SSE and WebSocket

---

## Project 5: Production-Ready Chat Backend

**Directory:** `ai_engineering/FastAPI/projects/05_production_chat/`

**Goal:** Full-featured chat API with production patterns

**Components:**
- [ ] All endpoints from previous projects
- [ ] JWT authentication
- [ ] Rate limiting (different tiers)
- [ ] Retry logic with exponential backoff
- [ ] Structured logging with request IDs
- [ ] `/health` and `/ready` endpoints
- [ ] Proper error responses
- [ ] Redis for rate limit storage
- [ ] docker-compose with Redis service

**Learning Validation:**
- [ ] Protected endpoints reject unauthenticated requests
- [ ] Rate limits work correctly
- [ ] Failed requests retry automatically
- [ ] Can trace a request through logs

---

## Project 6: Chat with Conversation History

**Directory:** `ai_engineering/FastAPI/projects/06_chat_with_history/`

**Goal:** Multi-turn conversations with persistence

**Components:**
- [ ] Conversation storage (in-memory to start, then Redis)
- [ ] `/v1/conversations` - create new conversation
- [ ] `/v1/conversations/{id}/messages` - add message to conversation
- [ ] Full conversation context sent to LLM
- [ ] Conversation listing and deletion
- [ ] Token counting for context management

**Learning Validation:**
- [ ] Multi-turn conversations maintain context
- [ ] Understands how conversation history affects tokens/cost
- [ ] Can implement basic memory management

---

## Project 7: Capstone - Complete GenAI Backend

**Directory:** `ai_engineering/FastAPI/projects/07_genai_backend_capstone/`

**Goal:** Production-grade backend combining all concepts

**Components:**
- [ ] Multi-provider support with LiteLLM
- [ ] Both sync and streaming endpoints
- [ ] JWT authentication
- [ ] Tiered rate limiting
- [ ] Conversation management
- [ ] Retry logic
- [ ] Structured logging
- [ ] Health checks
- [ ] OpenAPI documentation
- [ ] docker-compose with all services
- [ ] Comprehensive README

**Learning Validation:**
- [ ] Can explain every component and why it exists
- [ ] Can debug issues using logs
- [ ] Can extend with new features
- [ ] Ready to build on this for RAG/agents

---

# RECOMMENDED LEARNING ORDER

1. **Start:** Section 1 (FastAPI Fundamentals) -> Project 1
2. **Then:** Section 6.1-6.2 (Docker) -> Ensure Project 1 works in Docker
3. **Then:** Section 3.1 (Anthropic SDK) -> Project 2
4. **Then:** Section 3.3 (LiteLLM) + Section 3.2, 3.4 -> Project 3
5. **Then:** Section 2 (Async) + Section 4 (Streaming) -> Project 4
6. **Then:** Section 5 (Production Patterns) -> Project 5
7. **Then:** Projects 6 and 7

---

# RESOURCES SUMMARY

## Official Documentation
- FastAPI: https://fastapi.tiangolo.com/
- Pydantic: https://docs.pydantic.dev/
- Anthropic: https://docs.anthropic.com/
- OpenAI: https://platform.openai.com/docs/
- LiteLLM: https://docs.litellm.ai/
- httpx: https://www.python-httpx.org/
- Docker: https://docs.docker.com/

## Tutorials & Guides
- FastAPI Full Course (freeCodeCamp): YouTube
- Real Python asyncio: https://realpython.com/async-io-python/
- Docker for Python Developers: https://docs.docker.com/language/python/

## GitHub Repositories to Study
- LangServe (FastAPI for LangChain): https://github.com/langchain-ai/langserve
- FastChat (Multi-model serving): https://github.com/lm-sys/FastChat
- LiteLLM: https://github.com/BerriAI/litellm
