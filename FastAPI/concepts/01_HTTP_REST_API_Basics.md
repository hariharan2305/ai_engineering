# HTTP & REST API Basics: A Senior Engineer's Guide

## Table of Contents

1. [HTTP: The Universal Language](#http-the-universal-language-of-the-web)
2. [HTTP Methods](#http-methods-the-verbs-of-web-communication)
3. [HTTP Status Codes](#http-status-codes-the-servers-response-language)
4. [Request/Response Structure](#requestresponse-structure-the-anatomy)
5. [REST Principles](#rest-architectural-principles)
6. [Putting It Together](#putting-it-together-a-genai-api-example)
7. [Quick Reference](#quick-reference-card)

---

## HTTP: The Universal Language of the Web

HTTP (HyperText Transfer Protocol) is **the standardized contract between any two systems communicating over the internet**.

### Mechanical Engineering Analogy
HTTP is like the ISO standards for pipe fittings. Doesn't matter if you're connecting a pump made in Germany to a valve made in Japan—if both follow ISO standards, they'll work together. HTTP is that standard for web communication.

```
┌─────────────┐                      ┌─────────────┐
│   Client    │ ──── HTTP Request ──▶│   Server    │
│ (Your code, │                      │  (FastAPI,  │
│  browser,   │ ◀── HTTP Response ───│   Flask,    │
│  Postman)   │                      │   Django)   │
└─────────────┘                      └─────────────┘
```

Every HTTP interaction is a **request-response cycle**. Client asks, server answers. Always. There's no persistent connection maintained—each request is independent and complete.

### Key Characteristics

- **Stateless**: Server doesn't maintain memory of previous requests
- **Synchronous**: Client waits for response before continuing
- **Reliable**: Built on TCP (connection-oriented, guaranteed delivery)
- **Versioned**: HTTP/1.1 (widely used), HTTP/2 (optimized), HTTP/3 (latest)

---

## HTTP Methods: The Verbs of Web Communication

HTTP methods tell the server **what operation you want to perform**. Think of them as SQL operations mapped to web requests:

| HTTP Method | SQL Equivalent | Intent | Idempotent? | Safe? |
|-------------|---------------|--------|-------------|-------|
| `GET` | `SELECT` | Retrieve data | ✅ Yes | ✅ Yes |
| `POST` | `INSERT` | Create new resource | ❌ No | ❌ No |
| `PUT` | `UPDATE` (full replace) | Replace entire resource | ✅ Yes | ❌ No |
| `PATCH` | `UPDATE` (partial) | Modify specific fields | ❌ No | ❌ No |
| `DELETE` | `DELETE` | Remove resource | ✅ Yes | ❌ No |
| `HEAD` | `SELECT (metadata only)` | Like GET, but no body | ✅ Yes | ✅ Yes |

### Key Concepts

**Idempotency**: If your network hiccups and retries a request multiple times, you get the same result (no side effects).
- `GET`, `PUT`, `DELETE` are idempotent
- `POST`, `PATCH` are not
- **Why it matters**: Payment systems use idempotency keys to make `POST` operations effectively idempotent, preventing duplicate charges on network retries.

**Safety**: The method doesn't modify server state.
- `GET`, `HEAD`, `OPTIONS` are safe
- Everything else modifies state

### When to Use Each

#### GET: Fetching Data
Safe, idempotent, no side effects. Can be cached.

```
GET /models                    # List all models
GET /models/gpt-4              # Get details of a specific model
GET /models?provider=openai    # List models filtered by provider
GET /models?limit=10&offset=0  # Pagination
```

#### POST: Creating Something New
Not idempotent. Each call creates a new resource. Use when you're instructing the server to create/generate something.

```
POST /chat/completions         # Create a new chat completion
POST /embeddings               # Generate embeddings
POST /api-keys                 # Create a new API key
```

**Real-world example**: When you call OpenAI's API to generate a response, you're POSTing your conversation history because you want the model to *create* a new completion object.

#### PUT: Full Replacement
Idempotent. Send the ENTIRE object—it will be completely replaced.

```
PUT /models/my-custom-model    # Replace all config for this model
# Body: {"name": "my-custom-model", "provider": "openai", "max_tokens": 4096, "temperature": 0.7}
```

If you PUT the same data 100 times, the result is identical to putting it once.

#### PATCH: Partial Update
Not idempotent in general. Send only the fields that changed.

```
PATCH /models/my-custom-model  # Update just the max_tokens
# Body: {"max_tokens": 8192}
```

#### DELETE: Remove
Idempotent. Remove the resource.

```
DELETE /models/my-custom-model  # Delete this model config
```

First call returns 200. Subsequent calls return 404 (not found), but the effect is the same—resource is gone.

### Databricks/Spark Analogy

Think of it like your data engineering workflows:
- `GET` = Reading from a Delta table (no mutations, can be cached)
- `POST` = Appending new data to a table (creates new records)
- `PUT` = `OVERWRITE` mode—replace the entire partition/table
- `PATCH` = `MERGE` statement—update specific rows by key
- `DELETE` = Dropping a table or deleting specific rows

---

## HTTP Status Codes: The Server's Response Language

Status codes are **3-digit numbers** that tell you what happened. They're grouped by the first digit:

```
1xx - Informational (requests for more info, rare in practice)
2xx - Success ✅ (everything worked)
3xx - Redirection ↪️ (client needs to do something else)
4xx - Client Error ❌ (you messed up, fix your request)
5xx - Server Error 💥 (they messed up, their problem)
```

### The Essential Ones for API Development

| Code | Name | When to Use | Real Example |
|------|------|-------------|--------------|
| **200** | OK | Request succeeded, here's your data | `GET /models/gpt-4` returns model info |
| **201** | Created | New resource created successfully | `POST /chat/completions` created a completion |
| **204** | No Content | Success, but nothing to return | `DELETE /models/123` succeeded, no response body |
| **400** | Bad Request | Malformed request, invalid JSON, missing required fields | Sending `{"prompt": null}` when prompt is required |
| **401** | Unauthorized | No credentials provided or invalid credentials | Missing API key header |
| **403** | Forbidden | Valid credentials, but insufficient permissions | Free tier trying to access premium model |
| **404** | Not Found | Resource doesn't exist | `GET /models/nonexistent-model` |
| **422** | Unprocessable Entity | Valid JSON, but semantic validation failed | `{"temperature": 5.0}` when max is 2.0 |
| **429** | Too Many Requests | Rate limit exceeded | Exceeded API quota or request limit |
| **500** | Internal Server Error | Server crashed, unhandled exception | Null pointer, database connection failed |
| **502** | Bad Gateway | Upstream service failed to respond | Your FastAPI can't reach OpenAI's servers |
| **503** | Service Unavailable | Server overloaded or in maintenance | Model server down, too many concurrent requests |

### 401 vs 403 (Common Confusion)

- **401 Unauthorized** = "Who are you?" (Authentication failed)
  - Missing credentials
  - Invalid/expired token
  - The identity check failed

- **403 Forbidden** = "I know who you are, but you can't do this" (Authorization failed)
  - Valid credentials, but lacks permission
  - Free tier trying to access premium feature
  - User doesn't own this resource

### 400 vs 422 (FastAPI Context)

- **400 Bad Request** = Can't even parse your request
  - Malformed JSON (missing closing brace)
  - Wrong content-type header
  - Server can't understand the format

- **422 Unprocessable Entity** = Parsed it successfully, but validation failed
  - FastAPI's default for Pydantic validation errors
  - Value is wrong type: `"temperature": "hot"` instead of `"temperature": 0.7`
  - Value out of range: `max_tokens: 50000` when max is 4096
  - Semantic validation: required field missing after parsing

---

## Request/Response Structure: The Anatomy

### Complete HTTP Request

```
┌──────────────────────────────────────────────────────────┐
│ POST /v1/chat/completions?stream=true HTTP/1.1           │ ← Request Line
│                                                          │   (method, path, HTTP version)
├──────────────────────────────────────────────────────────┤
│ Host: api.anthropic.com                                  │
│ Authorization: Bearer sk-ant-...                         │
│ Content-Type: application/json                           │ ← Headers
│ X-Request-ID: abc-123                                    │   (metadata, auth, config)
│ User-Agent: PythonRequests/2.31.0                        │
├──────────────────────────────────────────────────────────┤
│ {                                                        │
│   "model": "claude-sonnet-4-20250514",                   │
│   "max_tokens": 1024,                                    │ ← Body (Request payload)
│   "messages": [                                          │   (JSON, form-data, etc.)
│     {"role": "user", "content": "Explain REST APIs"}     │
│   ]                                                      │
│ }                                                        │
└──────────────────────────────────────────────────────────┘
```

### Complete HTTP Response

```
┌──────────────────────────────────────────────────────────┐
│ HTTP/1.1 200 OK                                          │ ← Status Line
├──────────────────────────────────────────────────────────┤
│ Content-Type: application/json                           │
│ Content-Length: 256                                      │
│ X-Request-ID: abc-123                                    │ ← Headers
│ Cache-Control: no-cache                                  │   (metadata about response)
├──────────────────────────────────────────────────────────┤
│ {                                                        │
│   "id": "msg_123",                                       │
│   "type": "message",                                     │ ← Body (Response payload)
│   "content": [                                           │
│     {"type": "text", "text": "REST APIs follow..."}      │
│   ],                                                     │
│   "model": "claude-sonnet-4-20250514",                   │
│   "usage": {"input_tokens": 10, "output_tokens": 150}    │
│ }                                                        │
└──────────────────────────────────────────────────────────┘
```

### Breaking Down the URL

```
https://api.example.com/v1/models/gpt-4/completions?temperature=0.7&max_tokens=100
└─┬──┘ └──────┬───────┘└────────────┬──────────────┘ └────────────┬────────────────┘
scheme       host            path                         query params
            (domain)     (identifies WHICH resource)   (modifies HOW to process)

Full URL components:
- Scheme: https (protocol)
- Host: api.example.com (domain + port)
- Path: /v1/models/gpt-4/completions (specific resource hierarchy)
- Query: ?temperature=0.7&max_tokens=100 (optional parameters)
```

### Four Ways to Send Data in HTTP

| Location | Use Case | Example | FastAPI |
|----------|----------|---------|---------|
| **Path Parameters** | Identify specific resource | `/models/{model_id}` | `def get_model(model_id: str)` |
| **Query Parameters** | Filtering, pagination, options | `?limit=10&offset=0` | `def list(limit: int = 10)` |
| **Headers** | Auth, metadata, content negotiation | `Authorization: Bearer ...` | `def endpoint(x_api_key: str = Header())` |
| **Body** | Complex data, creating/updating resources | JSON payload | `def create(data: ModelConfig)` |

### ML/Data Pipeline Analogy

Think of it in terms of your Spark workflows:

```
Path Params     → Specifying which Delta table: /tables/sales_2024
Query Params    → Your WHERE clause filters: ?region=US&year=2024
Headers         → Your Databricks workspace token, cluster config
Body            → The actual DataFrame you're writing, transformation logic
```

### Content Negotiation

Headers tell both client and server what format the data is in:

```http
Request:
Content-Type: application/json        # I'm sending JSON
Accept: application/json              # I want JSON back

Response:
Content-Type: application/json        # I'm responding with JSON
```

---

## REST: Architectural Principles

REST (REpresentational State Transfer) isn't a protocol—it's a **design philosophy** for APIs. Think of it as "best practices that make APIs predictable, scalable, and maintainable."

### The 6 REST Constraints

#### 1. Client-Server Separation
Frontend and backend are independent components. Your FastAPI doesn't care if the client is React, mobile app, Python script, or cURL. They communicate solely through HTTP.

**Benefit**: Scale frontend and backend independently. Replace UI without touching API.

#### 2. Statelessness
**Each request contains ALL information needed to process it.** The server doesn't remember previous requests or maintain session state.

```python
# ❌ Stateful (bad):
POST /login  → Server stores session in memory
GET /models  → Server checks stored session: "Oh, this is John"

# ✅ Stateless (good):
GET /models
Headers: Authorization: Bearer eyJhbGc...  # Token contains user info
# Server: "Decode token, verify signature, extract user ID, process request"
```

**Why it matters for GenAI backends:**
When you scale to multiple FastAPI instances behind a load balancer:
- Request 1 hits Server A
- Request 2 hits Server D (different server)
- Request 3 hits Server B (different server again)

If Server A stored session state, but the next request goes to Server D, that server wouldn't have the context. Statelessness = horizontal scalability. This is identical to how you'd scale Spark jobs across a cluster—each task is independent.

**Token-based auth example:**
```
Client → POST /login with credentials
Server → Returns JWT token containing user_id, permissions
Client → Every subsequent request includes token in Authorization header
Server → Verifies token signature (no DB lookup needed), extracts user_id
```

#### 3. Cacheability
Responses should declare if they're cacheable. HTTP defines caching rules.

```http
Response:
Cache-Control: max-age=3600        # Cache for 1 hour
Cache-Control: no-cache            # Don't cache
Cache-Control: private             # Only client caches, not proxies
```

**Implication**: `GET /models` might be cached, but `POST /chat/completions` won't be (it creates new data).

#### 4. Uniform Interface
Predictable, consistent patterns across the API.

```python
# Consistent resource naming
/providers      # Collection
/providers/{id}  # Specific item

# Consistent methods
GET    /providers       → List
GET    /providers/{id}  → Get one
POST   /providers       → Create
PUT    /providers/{id}  → Replace
PATCH  /providers/{id}  → Update
DELETE /providers/{id}  → Delete

# Consistent response structure
{
  "success": true,
  "data": {...},
  "error": null
}
```

#### 5. Layered System
Client doesn't know (or care) if it's hitting your server directly or going through load balancers, caches, CDNs, reverse proxies, API gateways, etc.

```
Client → Load Balancer → Cache → FastAPI Instance 1
                       ↘ FastAPI Instance 2
                       ↘ FastAPI Instance 3
```

Client just sees one endpoint; the architecture behind it is invisible.

#### 6. Code on Demand (Optional)
Server can send executable code to client. Rarely used (JavaScript in browsers is an exception). Skip this in practice.

### RESTful URL Design Patterns

#### ✅ Correct: Resources (Nouns)

```python
GET    /models                    # List all models
GET    /models/gpt-4              # Get specific model
POST   /models                    # Create new model
PUT    /models/gpt-4              # Replace model
PATCH  /models/gpt-4              # Update model
DELETE /models/gpt-4              # Delete model

# Nested resources for relationships
GET    /providers/openai/models           # All models from OpenAI
GET    /users/123/api-keys                # All API keys for user 123
POST   /conversations/abc/messages        # Add message to conversation
GET    /chat/completions/xyz/feedback     # Get feedback on completion
```

#### ❌ Wrong: Actions (Verbs)

```python
GET    /getModels           # Wrong
POST   /createModel         # Wrong
POST   /deleteModel/gpt-4   # Wrong
GET    /listUsers           # Wrong
```

The HTTP method already specifies the action. Resources should be nouns.

#### Handling Complex Operations

Sometimes you need to perform actions that don't fit standard CRUD:

```python
# Option 1: Treat result as a resource
POST /models/gpt-4/validate    # Validate this model
POST /embeddings               # Generate embeddings (create new embedding resource)

# Option 2: Query parameter for sub-action
GET  /models?format=json       # Different format of same resource
POST /messages?action=resend   # Non-standard action on resource

# Option 3: Separate endpoint for workflow
POST /batch-processing         # Start a batch job (creates new resource)
GET  /batch-processing/123     # Check job status
```

---

## Putting It Together: A GenAI API Example

### Real Example: Calling Anthropic's API

**Request:**
```http
POST /v1/messages HTTP/1.1
Host: api.anthropic.com
Authorization: Bearer sk-ant-...
Content-Type: application/json
X-Request-ID: req-abc-123
User-Agent: python-anthropic/0.15.0

{
  "model": "claude-sonnet-4-20250514",
  "max_tokens": 1024,
  "system": "You are a helpful assistant",
  "messages": [
    {
      "role": "user",
      "content": "Explain REST APIs to a machine learning engineer"
    }
  ]
}
```

**Breakdown:**
- **Method**: POST → Creating a new message/completion resource
- **Path**: /v1/messages → Creating a message in the v1 API
- **Headers**: Authorization token identifies which account, Content-Type tells server it's JSON
- **Body**: All context needed—model choice, system prompt, full conversation history
- **Stateless**: Server doesn't remember this conversation; client sends everything each time

**Response:**
```http
HTTP/1.1 200 OK
Content-Type: application/json
X-Request-ID: req-abc-123
Usage: input_tokens=15; cache_creation_input_tokens=0; cache_read_input_tokens=0; output_tokens=250

{
  "id": "msg_8awGwONIFfNWV4MUGMxRXVTy",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "REST APIs follow these key principles...",
      "index": 0
    }
  ],
  "model": "claude-sonnet-4-20250514",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 15,
    "output_tokens": 250
  }
}
```

**Status code 200**: Success. The completion was generated.

**Why stateless matters here:**
If your ML pipeline calls this API 10 times in a row with different prompts, each call is completely independent. The server doesn't remember the conversation from call 1 by the time call 2 arrives. That's why you see the full conversation history in the request body—if you want continuity, you manage it on the client side.

---

## Quick Reference Card

### HTTP Methods at a Glance

```
GET     = Read (SELECT)         - Idempotent, safe, cacheable
POST    = Create (INSERT)       - Not idempotent, creates new resource
PUT     = Replace (UPDATE all)  - Idempotent, full replacement
PATCH   = Modify (UPDATE some)  - Not idempotent, partial update
DELETE  = Remove (DELETE)       - Idempotent, removes resource
HEAD    = Like GET, no body     - Idempotent, safe, cacheable
```

### Status Code Ranges

```
2xx = Success               3xx = Redirect
  200 OK                      301 Moved Permanently
  201 Created                 302 Found
  204 No Content              304 Not Modified

4xx = Client Error          5xx = Server Error
  400 Bad Request             500 Internal Server Error
  401 Unauthorized            502 Bad Gateway
  403 Forbidden               503 Service Unavailable
  404 Not Found
  422 Unprocessable Entity
  429 Too Many Requests
```

### Data Location Strategies

```
Path Params:   /models/{model_id}           → Which resource
Query Params:  ?limit=10&offset=20          → How to process
Headers:       Authorization: Bearer ...     → Auth + metadata
Body:          {"field": "value", ...}      → Complex data
```

### REST Design Checklist

```
✅ Resources are nouns: /models, /users, /api-keys
✅ HTTP methods indicate actions: GET=read, POST=create, PUT=replace, DELETE=remove
✅ Each request is stateless (contains all needed context)
✅ Consistent URL patterns across the API
✅ Meaningful status codes and error messages
✅ Stateless authentication via tokens
❌ URLs with verbs: /getModels, /createUser, /deletePost
❌ RPC-style endpoints: /api/doSomething
❌ Server storing session state
```

---

## Key Insights for GenAI Backends

1. **Statelessness Enables Scale**: Multiple servers, load balancers, horizontal scaling all work because requests are independent. This is why LLM APIs send full conversation history.

2. **Idempotency for Reliability**: Network retries on idempotent operations (GET, PUT, DELETE) are safe. On non-idempotent operations (POST), you need idempotency keys to prevent duplicates.

3. **Fast API Maps Directly**: When you write FastAPI routes, you're implementing REST principles:
   ```python
   @app.post("/models")              # POST method to /models resource
   def create_model(config: ModelConfig):  # Request body in Pydantic model
       return {"id": 123}            # Response body (converted to JSON)
   ```

4. **HTTP isn't just for web**: Every API—REST, GraphQL, gRPC over HTTP—ultimately uses HTTP. Understanding HTTP deeply is foundational.

---