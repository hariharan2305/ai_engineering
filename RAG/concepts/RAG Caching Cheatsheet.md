# Caching & Performance – Code Cheat‑Sheet (2026)

This cheat‑sheet covers the **semantic caching** components from your stack guide, plus Redis where it is actively used in industry for this purpose:

- GPTCache (semantic cache library from Zilliz)
- LangChain GPTCache integration
- Redis semantic cache in LangChain (`RedisSemanticCache`)
- Redis Vector Library (RedisVL) `SemanticCache` for LLMs
- Practical patterns: exact‑match + semantic caching, invalidation & TTL

Sources: GPTCache docs & repo, LangChain cache reference, and Redis semantic caching guides.[^1][^2][^3][^4][^5][^6][^7][^8][^9][^10][^11]

> All these caches sit between your **client** and the **LLM API**: they see a prompt, decide whether there is a cache hit, and either return the cached response or call the LLM and update the cache.

***

## 1. GPTCache – Standalone Semantic Cache

GPTCache provides a general semantic cache with pluggable **embedding backends** and **storage backends** (SQLite, FAISS, Milvus/Qdrant, etc.), plus tight integration with LangChain and LlamaIndex.[^3][^4][^6][^12]

### 1.1 Install

```bash
pip install gptcache
```

Make sure Python ≥ 3.8.1 per GPTCache docs.[^7]

### 1.2 Quick start – basic exact‑match cache for OpenAI

From the GPTCache Quick Start:[^3]

```python
from gptcache.core import cache

# 1) Initialize the cache with default settings
cache.init()

# 2) (Optional) Set OpenAI key via GPTCache helper
# If you usually do: openai.api_key = "..."
# GPTCache recommends:
cache.set_openai_key()

# 3) Use GPTCache with OpenAI

from gptcache.adapter import openai as gptcache_openai

resp = gptcache_openai.ChatCompletion.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
)

print(resp.choices.message["content"])
```

The `gptcache_openai.ChatCompletion.create` wrapper hides cache lookups and updates; the first call hits the LLM, later calls may be served from cache depending on configuration.[^3]

### 1.3 Enable semantic caching with custom embeddings & vector store

From GPTCache usage examples:[^3]

```python
from gptcache.core import cache
from gptcache.adapter import openai as gptcache_openai
from gptcache.manager import manager_factory
from gptcache.embedding import openai as openai_embed
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

# 1) Configure embedding function
embedding_func = openai_embed.OpenAI()  # uses OpenAI embeddings

# 2) Configure storage backends (SQLite + FAISS here, but can be others)

data_manager = manager_factory(
    # where to store raw data (prompts/answers)
    data_path="./gptcache_data.sqlite",
    # vector store configuration
    vector_params={
        "vector_store": "faiss",
        "dimension": 1536,  # must match your embedding dim
    },
)

# 3) Initialize cache with semantic evaluation (distance-based)

cache.init(
    embedding_func=embedding_func.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
)

cache.set_openai_key()

# 4) Use OpenAI via GPTCache adapter as before

response = gptcache_openai.ChatCompletion.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)

print(response.choices.message["content"])
```

This configuration lets GPTCache **embed prompts**, store them in FAISS, and serve semantically similar queries from cache.[^3]

### 1.4 Using GPTCache with other LLMs and frameworks

GPTCache supports:

- Multiple embedding backends (OpenAI, Cohere, Hugging Face, ONNX, SentenceTransformers).[^3]
- LangChain and LlamaIndex integrations (see sections below) so you can plug GPTCache into any RAG pipeline built with those frameworks.[^4][^6]

***

## 2. GPTCache with LangChain – `GPTCache` LLM Cache

LangChain’s community cache module includes a `GPTCache` wrapper that uses GPTCache as the underlying LLM cache backend.[^10]

### 2.1 Install

```bash
pip install langchain-community gptcache
```

### 2.2 Set GPTCache as the global LLM cache in LangChain

From LangChain’s `GPTCache` reference:[^10]

```python
from langchain_community.cache import GPTCache as LangChainGPTCache
from langchain_community.globals import set_llm_cache

from gptcache.adapter.langchain import init_gptcache

# Initialize GPTCache using helper
init_func = init_gptcache(
    embedding_func="openai",   # or custom as per GPTCache docs
    data_manager="sqlite_faiss",  # shorthand in helper
)

semantic_cache = LangChainGPTCache(init_func)

set_llm_cache(semantic_cache)

# Any LangChain LLM call now uses GPTCache underneath
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

print(llm.invoke("Explain retrieval-augmented generation in 2 sentences."))
```

This will transparently apply GPTCache for prompts sent through LangChain LLMs.[^10]

***

## 3. Redis as Semantic Cache – Industry Usage

Redis is widely used as a **semantic cache** for LLM apps in production, backed by Redis’ vector search (via Redis Stack / RedisVL):

- Redis created a dedicated **semantic cache** pattern leveraging vector search to store question–answer pairs and reuse them for semantically similar queries.[^13][^2][^5][^9][^11]
- LangChain exposes `RedisSemanticCache` to use Redis as a semantic LLM cache.[^8][^1]

Below are two main options: **LangChain RedisSemanticCache** and **Redis Vector Library SemanticCache**.

***

### 3.1 LangChain `RedisSemanticCache`

`RedisSemanticCache` uses Redis as a vector DB under the hood and integrates with the LangChain LLM cache API.[^1][^8]

#### 3.1.1 Install

```bash
pip install langchain-community redis
```

#### 3.1.2 Configure RedisSemanticCache

From the LangChain reference:[^8][^1]

```python
from langchain_community.globals import set_llm_cache
from langchain_community.cache import RedisSemanticCache
from langchain_openai import OpenAIEmbeddings

redis_url = "redis://localhost:6379"  # or your Redis/Redis Stack URL

semantic_cache = RedisSemanticCache(
    redis_url=redis_url,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    score_threshold=0.2,  # similarity threshold for cache hits
)

set_llm_cache(semantic_cache)

# Now all LangChain LLM calls will check Redis semantic cache first
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini")

answer = llm.invoke("What is the capital of France?")
print(answer)
```

`score_threshold` controls how similar a query must be to hit the cache: lower threshold = stricter match.[^1][^8]

***

### 3.2 Redis Vector Library (RedisVL) – `SemanticCache` for LLMs

RedisVL is a higher‑level Python library that provides an explicit **SemanticCache** interface over Redis’ vector search.[^2][^5][^9]

#### 3.2.1 Install

```bash
pip install redisvl
```

#### 3.2.2 Initialize SemanticCache

From RedisVL semantic cache docs:[^9]

```python
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer

# Configure SemanticCache
llmcache = SemanticCache(
    name="llmcache",                          # Redis index name
    redis_url="redis://localhost:6379",      # Redis connection
    distance_threshold=0.1,                   # similarity threshold
    vectorizer=HFTextVectorizer("redis/langcache-embed-v1"),  # embedding model
)
```

This creates a Redis index to store embeddings + answers for cache entries.[^9]

#### 3.2.3 Using SemanticCache in a chat loop

The RedisVL guide shows how to wrap an OpenAI‑based chat loop:[^2][^9]

```python
from openai import OpenAI

client = OpenAI()

# Simple helper
def ask_llm(question: str) -> str:
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=question,
        max_tokens=200,
    )
    return response.choices.text.strip()


def ask_with_cache(question: str) -> str:
    # 1) Check semantic cache
    cached = llmcache.get(question)
    if cached is not None:
        print("[Cache hit]")
        return cached["response"]

    # 2) Otherwise call LLM
    print("[Cache miss]")
    answer = ask_llm(question)

    # 3) Store in cache
    llmcache.add(question, answer)

    return answer

print(ask_with_cache("What is the capital of France?"))
print(ask_with_cache("Which city is the capital of France?"))  # likely cache hit
```

This pattern shows **semantic caching** where differently worded but semantically equivalent queries reuse the same answer.[^5][^11][^9]

***

## 4. Caching Patterns for RAG

### 4.1 Combine exact‑match and semantic caching

Both GPTCache and RedisVL SemanticCache support:

- **Exact‑match**: queries that are byte‑for‑byte identical; easiest and safest form of caching.
- **Semantic cache**: queries that are different strings but embed to similar vectors.[^6][^2][^9][^3]

Typical pattern:

1. Check an **in‑memory / exact‑match cache** (e.g., Python dict, Redis string key) keyed by exact prompt.
2. If no hit, check **semantic cache** (GPTCache, Redis SemanticCache, LangChain RedisSemanticCache).
3. If still no hit, call LLM and update both caches.

This maximizes cache hits because exact matches are cheap to store and very precise, while semantic cache broadens coverage.[^11][^5][^2]

### 4.2 TTL, invalidation, and RAG updates

When your **underlying knowledge base** (docs, vectors) changes, cached answers may become stale.

Best practices:

- Use **TTL (time‑to‑live)** on cached entries (Redis supports this natively; GPTCache has eviction policies) so answers refresh periodically.[^5][^11]
- Consider **versioned cache keys**, including a **KB version hash** in the cache key or metadata, so a re‑index invalidates old entries.
- In RAG apps where content is highly dynamic (e.g., stock prices), restrict semantic caching to **low‑volatility queries** or **static content** and rely more on fresh retrieval for dynamic content.[^2][^5]

### 4.3 Where to put caching in RAC/RAG

- **Before retrieval**: rare; you normally want fresh retrieval, not cached retrieval results.
- **After retrieval, before generation (most common)**: cache the full **(query, retrieved context) → answer** mapping.
- **Per‑tool / per‑agent**: in agentic setups, each tool’s LLM calls can have their own semantic cache for repeat tool invocations.

Diagrammatically: `User Query → Retrieval → (Semantic Cache?) → LLM → Answer`.

Use this cheat‑sheet together with your **retrieval** and **LLM** cheat‑sheets to add caching layers that lower costs and latency without sacrificing freshness.

---

## References

1. [.RedisSemanticCache¶](https://api.python.langchain.com/en/latest/cache/langchain_community.cache.RedisSemanticCache.html)

2. [Improving RAG Applications with Semantic Caching and RAGAS](https://2024.allthingsopen.org/improving-rag-applications-with-semantic-caching-and-ragas) - Semantic caching is a way of boosting RAG performance by serving relevant, cached LLM responses, thu...

3. [GPTCache Quick Start](https://gptcache.readthedocs.io/en/latest/usage.html)

4. [GitHub - zilliztech/GPTCache: Semantic cache for LLMs. Fully ...](https://github.com/zilliztech/gptcache) - Semantic cache for LLMs. Fully integrated with LangChain and llama_index. - GitHub - zilliztech/GPTC...

5. [Level up RAG apps with Redis Vector Library](https://redis.io/blog/level-up-rag-apps-with-redis-vector-library/) - Semantic caching to save on LLM costs and accelerate responses; Semantic routing to send user reques...

6. [GPTCache : A Library for Creating Semantic Cache for LLM Queries](https://gptcache.readthedocs.io)

7. [GPTCache : A Library for Creating Semantic Cache for LLM Queries](https://gptcache.readthedocs.io/en/latest/)

8. [RedisSemanticCache | langchain_redis | LangChain Reference](https://reference.langchain.com/python/langchain-redis/cache/RedisSemanticCache) - Redis-based semantic cache implementation for LangChain. This class provides a semantic caching mech...

9. [Semantic Caching for LLMs#](https://docs.redisvl.com/en/v0.6.0/user_guide/03_llmcache.html)

10. [GPTCache — LangChain documentation](https://reference.langchain.com/v0.3/python/community/cache/langchain_community.cache.GPTCache.html) - ... ]). Return type: None. Examples using GPTCache. Model caches. On this page. GPTCache. __init__()...

11. [What is semantic caching? Guide to faster, smarter LLM apps - Redis](https://redis.io/blog/what-is-semantic-caching/) - The caching layer intercepts queries, checks for semantic matches in the cache, and only forwards ca...

12. [gptcache · PyPI](https://pypi.org/project/gptcache/) - GPTCache, a powerful caching library that can be used to speed up and lower the cost of chat applica...

13. [A Semantic Cache using LangChain](https://www.youtube.com/watch?v=LRswXEc5chE) - One common concern of developers building AI applications is how fast answers from LLMs will be serv...

