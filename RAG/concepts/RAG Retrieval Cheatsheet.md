# Retrieval Strategies & Libraries – Code Cheat‑Sheet (2026)

This cheat‑sheet shows how to implement **advanced retrieval strategies** using current libraries:

- LangChain retrievers
  - Basic vector store retriever
  - MultiQueryRetriever (multi‑query / RAG‑Fusion‑style)[^1][^2]
  - ContextualCompressionRetriever (LLM‑based contextual compression)[^3][^4]
- LlamaIndex query engines
  - Basic VectorStoreIndex query engine
  - RouterQueryEngine (routing across indices / retrieval strategies)[^5][^6][^7]
- Cloud / managed retrieval
  - Bedrock Knowledge Base `retrieve` (KB retrieval API)[^8]
  - Vertex AI Search retriever (as RAG backend)[^9]
- Advanced retrieval strategies
  - Hybrid Search (Dense + Sparse + RRF fusion)
  - HyDE (Hypothetical Document Embeddings)
  - Parent Document Retrieval
  - Query Decomposition
  - GraphRAG

All snippets follow **current docs and cookbook patterns as of early 2026**.[^10]

> These examples assume you already have embeddings and a vector DB or managed RAG backend, as covered in the Embeddings and Vector DB cheat‑sheets.

***

## 1. LangChain – Basic Vector Store Retriever

This is the building block most other LangChain retrieval strategies wrap.

### 1.1 Setup (example with FAISS)

```bash
pip install langchain-core langchain-openai faiss-cpu
```

```python
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Example docs
docs = [
    Document(page_content="LangChain makes it easy to build LLM apps."),
    Document(page_content="Retrieval augmented generation improves factuality."),
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Build vector store and retriever
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

results = retriever.invoke("What improves factuality?")
for r in results:
    print(r.page_content)
```

This pattern generalizes to Qdrant, Weaviate, Pinecone, pgvector, etc., by swapping the vector store class.

***

## 2. LangChain – MultiQueryRetriever (Multi‑Query / RAG‑Fusion)

`MultiQueryRetriever` uses an LLM to generate multiple reformulations of the user query, retrieves for each, and unions the results.[^11][^2][^1]

### 2.1 Install

```bash
pip install langchain-core langchain-openai langchain-community
```

### 2.2 Wrap an existing retriever with MultiQueryRetriever

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

# Assume you already have a base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

prompt = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are an AI assistant tasked with generating multiple search queries.\n"
        "Generate 3 different versions of the user question to retrieve relevant documents.\n"
        "Provide each alternative on its own line.\n\n"
        "Original question: {question}"
    ),
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    prompt=prompt,
    include_original=True,  # also search with the original query
)

query = "How can I reduce hallucinations in LLM responses?"
results = multi_query_retriever.invoke(query)

print("Retrieved", len(results), "docs")
for doc in results[:3]:
    print("-", doc.page_content[:200].replace("\n", " "))
```

This matches the documented `from_llm` usage in the current MultiQueryRetriever reference.[^2][^1]

***

## 3. LangChain – ContextualCompressionRetriever (LLM‑based Compression)

`ContextualCompressionRetriever` wraps a base retriever and a **DocumentCompressor** (like `LLMChainExtractor`) to trim retrieved docs down to only query‑relevant snippets.[^4][^12][^3]

### 3.1 Install

```bash
pip install langchain-core langchain-openai
```

### 3.2 Wrap base retriever with LLMChainExtractor

```python
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# base_retriever from previous example

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

compressor = LLMChainExtractor.from_llm(llm=llm)

compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor,
)

query = "What does this corpus say about RAG evaluation metrics?"

compressed_docs = compression_retriever.invoke(query)

for d in compressed_docs:
    print("--- Compressed Chunk ---")
    print(d.page_content[:400])
```

This pattern mirrors the official blog example: base retriever → LLMChainExtractor → ContextualCompressionRetriever.[^3][^4]

> You can swap `LLMChainExtractor` with other compressors (e.g., `EmbeddingsFilter`, `LLMTextSplitter`) for different behaviors.[^3]

***

## 4. LlamaIndex – Basic VectorStoreIndex Query Engine

LlamaIndex’s default retrieval pattern is to build a **VectorStoreIndex** over nodes and call `.as_query_engine()`.[^6]

### 4.1 Install core + OpenAI integrations (example)

```bash
pip install llama-index-core llama-index-embeddings-openai llama-index-llms-openai
```

### 4.2 Build index and query

```python
import os
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

os.environ["OPENAI_API_KEY"] = "sk-..."

llm = OpenAI(model="gpt-4.1-mini", temperature=0)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Simple documents
docs = [
    Document(text="LlamaIndex helps you build RAG over private data."),
    Document(text="RouterQueryEngine lets you route queries to different indices."),
]

index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)

query_engine = index.as_query_engine(
    similarity_top_k=5,
    llm=llm,
)

response = query_engine.query("How can I do RAG over my documents?")
print(response)
```

This is the canonical pattern from current LlamaIndex examples.[^6]

***

## 5. LlamaIndex – RouterQueryEngine (Routing Across Retrieval Modes)

`RouterQueryEngine` lets you route a query to one of several query engines (e.g., a **VectorStoreIndex** for detailed lookup vs a **SummaryIndex** for summarization).[^7][^5][^6]

### 5.1 Install (example with Mistral, but you can use OpenAI too)

```bash
pip install llama-index llama-index-llms-openai llama-index-embeddings-openai
```

### 5.2 Define indices and router query engine

Adapted from the RouterQueryEngine example:[^7][^6]

```python
import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors.llm_selectors import LLMSingleSelector
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

os.environ["OPENAI_API_KEY"] = "sk-..."

llm = LlamaOpenAI(model="gpt-4.1-mini", temperature=0)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Load documents from a directory
reader = SimpleDirectoryReader("./data")
documents = reader.load_data()

# Create indices over the same data
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
summary_index = SummaryIndex.from_documents(documents)

# Create individual query engines
vector_engine = vector_index.as_query_engine(similarity_top_k=5, llm=llm)
summary_engine = summary_index.as_query_engine(response_mode="tree_summarize", llm=llm)

# Wrap them as tools
tools = [
    QueryEngineTool(
        query_engine=summary_engine,
        metadata=ToolMetadata(
            name="summarizer",
            description="Good for summarization questions over the corpus.",
        ),
    ),
    QueryEngineTool(
        query_engine=vector_engine,
        metadata=ToolMetadata(
            name="retriever",
            description="Good for detailed, context-specific questions.",
        ),
    ),
]

router_engine = RouterQueryEngine.from_defaults(
    query_engine_tools=tools,
    selector=LLMSingleSelector.from_defaults(llm=llm),
)

response = router_engine.query("Give me a short overview of this repo.")
print(response)

response2 = router_engine.query("What does file X say about error handling?")
print(response2)
```

This pattern is straight from the RouterQueryEngine tutorials, using an LLM selector to route between engines.[^5][^6][^7]

***

## 6. Managed Retrieval – Bedrock Knowledge Bases `retrieve`

Bedrock Knowledge Bases handle ingestion, chunking, indexing, and retrieval; you call the `retrieve` API to get passages for RAG.[^8]

### 6.1 Prerequisites

- A knowledge base already created & associated with a vector store and data sources (see Vector DB cheat‑sheet for `create_knowledge_base`).[^8]

### 6.2 Python retrieval call

```bash
pip install boto3
```

```python
import boto3

bedrock_agent = boto3.client("bedrock-agent", region_name="us-east-1")

knowledge_base_id = "kb-xxxxxxxx"  # from create_knowledge_base response

query = "Summarize the refund policy for EU customers."

response = bedrock_agent.retrieve(
    knowledgeBaseId=knowledge_base_id,
    retrievalQuery={"text": query},
    retrievalConfiguration={
        "vectorSearchConfiguration": {
            "numberOfResults": 8,
        }
    },
)

for item in response["retrievalResults"]:
    content = item["content"]["text"]["text"]
    score = item.get("score")
    source = item.get("metadata", {}).get("sourceUri")
    print("Score:", score, "Source:", source)
    print(content[:300], "\n---\n")
```

You then pass these `content` snippets as context into your LLM prompt.

***

## 7. Managed Retrieval – Vertex AI Search (LangChain Retriever)

Vertex AI Search is a managed RAG backend that handles indexing and ranking; you query via a retriever wrapper.[^9]

### 7.1 Install & configure

```bash
pip install -U langchain-google-vertexai google-cloud-discoveryengine
```

```python
import os
from langchain_google_vertexai import VertexAISearchRetriever

PROJECT_ID = "your-gcp-project-id"
LOCATION_ID = "global"        # or region like "us-central1"
DATA_STORE_ID = "your-datastore-id"  # from Vertex AI Search config

retriever = VertexAISearchRetriever(
    project_id=PROJECT_ID,
    location_id=LOCATION_ID,
    data_store_id=DATA_STORE_ID,
    max_documents=5,
)

query = "What does our SSO onboarding guide say about Okta integration?"
results = retriever.invoke(query)

for doc in results:
    print(doc.metadata.get("source"))
    print(doc.page_content[:300], "\n---\n")
```

This code follows the current Vertex AI Search retriever docs in LangChain OSS.[^9]

***

## 8. Hybrid Search (Dense + Sparse + RRF)

Hybrid search combines **dense vector search** (semantic similarity) with **sparse BM25** (exact keyword match) and fuses the results.

**Why it matters:** Dense search misses exact terms — product codes, acronyms, proper nouns. BM25 misses paraphrases and conceptual matches. Hybrid catches both, making it the current standard for production RAG.

**Process:**
1. Run a dense ANN query against your vector store (top-K by cosine similarity)
2. Run a BM25 query over the same corpus (top-K by term frequency)
3. Fuse both ranked lists with **Reciprocal Rank Fusion (RRF)**
4. Return the top-K from the unified list to the LLM

**RRF formula:** For each document, its fused score = Σ `1 / (k + rank_i)` across all ranked lists, where `k = 60` is a smoothing constant. Higher score = better combined rank.

```python
def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
```

**When to use:** Almost always. Hybrid is the default for production — pure dense retrieval is a starting point, not the end state.

**Libraries:** Qdrant native hybrid API (sparse + dense), Weaviate hybrid search, LangChain `EnsembleRetriever` (combines any two retrievers with weighted fusion), `rank_bm25` for a custom BM25 layer.

***

## 9. HyDE (Hypothetical Document Embeddings)

Instead of embedding the user's short query directly, ask an LLM to generate a **hypothetical answer**, then embed that and retrieve documents similar to it.

**Why it matters:** A user query ("What causes transformer attention to fail on long sequences?") is short and sparse. A hypothetical answer lives in the same semantic space as real documents — longer, richer, better aligned with how the corpus is written. This narrows the gap between query space and document space.

**Process:**
1. User submits a query
2. LLM generates a plausible but possibly fictional answer (the "hypothetical document")
3. Embed the hypothetical document (not the original query)
4. Retrieve real documents by vector similarity to that embedding
5. Pass real retrieved docs (not the hypothetical) to the LLM for final generation

```python
# Step: generate the hypothetical document
response = llm.invoke(
    f"Write a short passage that directly answers this question:\n{query}"
)
hypothetical_doc = response.content
# Then embed hypothetical_doc and query your vector store with it
```

**Trade-off:** Adds one LLM call per query. Most effective when queries are short or phrased differently from how documents are written. Less useful on corpora where query and document vocabulary already align well.

***

## 10. Parent Document Retrieval

Index small, precise chunks for retrieval but return the larger **parent document** (or surrounding context) to the LLM.

**Why it matters:** Small chunks produce better embedding matches (focused semantics), but give the LLM too little context to answer well. Large chunks give full context but embed poorly (diluted semantics). Parent document retrieval gets the best of both.

**Process:**
1. **At index time:** split each document into small child chunks (e.g. 200 tokens). Store each child with a pointer back to its parent (e.g. 1000-token block or the full document).
2. **At query time:** retrieve the top-K child chunks by vector similarity (small = precise match).
3. **Before generation:** swap each child for its parent — feed the full parent context to the LLM.

**Variants:**
- **Small-to-big:** child is a sentence or short paragraph; parent is the full section
- **Sentence window:** retrieve a single sentence but expand to ±N surrounding sentences before passing to LLM

**LangChain implementation:** `ParentDocumentRetriever` from `langchain.retrievers` handles the child→parent mapping automatically using an in-memory or persistent docstore.

***

## 11. Query Decomposition

For complex, multi-part questions, use an LLM to break the query into 2–4 focused sub-questions, retrieve independently for each, then synthesise.

**Why it matters:** A single complex query ("Compare the refund policies of our EU and US stores, and flag any regulatory conflicts") retrieves poorly because no single chunk answers all parts. Decomposition lets each sub-question find the right chunk.

**Process:**
1. LLM receives the complex query and outputs N sub-questions
2. Each sub-question is run through the retriever independently (in parallel where possible)
3. Retrieved chunks are deduplicated and merged
4. LLM synthesises a final answer across all retrieved context

**Variants:**
- **Parallel decomposition:** all sub-questions are independent, run concurrently
- **Sequential decomposition (step-back):** answer sub-Q1 first, use that answer to refine sub-Q2 (useful for reasoning chains)

**When to use:** Multi-hop questions, comparison questions, questions that span multiple documents or topics. Not worth the extra LLM calls for simple factual lookups.

***

## 12. GraphRAG

GraphRAG replaces the flat vector index with a **knowledge graph** — entities and relationships extracted from documents become nodes and edges that can be traversed to answer relational questions.

**Why it matters:** Vector search finds semantically similar text but cannot reason about relationships ("How is Company A connected to the regulatory failure in Report B?"). A knowledge graph makes those connections explicit and traversable.

**Process:**
1. **Ingestion:** an LLM or NLP pipeline extracts entities (people, orgs, concepts) and relationships from documents and builds a graph
2. **At query time:** the query is parsed for entities → graph traversal collects the relevant subgraph (nodes + edges)
3. The subgraph is serialised as text context and passed to the LLM alongside (or instead of) vector-retrieved chunks

**When to use:**
- Global, "big-picture" questions across a large corpus ("What are the main themes?")
- Multi-hop relational questions ("Who approved the policy that led to X?")
- Domains with rich entity structure: legal, medical, financial, org knowledge bases

**Tools:** Microsoft GraphRAG (`github.com/microsoft/graphrag`), LlamaIndex `KnowledgeGraphIndex`, Neo4j + LangChain graph retriever.

**Trade-off:** High ingestion cost (LLM extractions per document). Best suited for corpora that are queried repeatedly and where relational questions matter. For straightforward factual retrieval, hybrid dense+sparse is cheaper and usually sufficient.

### Complete Indexing and Retrieval Flow — Every LLM Call

#### Index Time (the expensive part)

```
CORPUS FILES
     │
     ▼
① CHUNKING  (no LLM)
   Split each document into chunks (e.g. 1200 tokens, 100 overlap)
   3 docs × ~3 chunks each ≈ 9 chunks total
     │
     ▼
② ENTITY EXTRACTION  ← LLM call per chunk  [most expensive step]
   Each chunk → LLM:
     "Extract all entities (people, concepts, systems) and
      their relationships from this text"
   Output: JSON of {entities: [...], relationships: [...]}
   9 chunks → 9 LLM calls
     │
     ▼
③ ENTITY SUMMARIZATION  ← LLM call per entity
   Multiple chunks may mention the same entity (e.g. "BM25")
   GraphRAG merges all descriptions of that entity across chunks,
   then calls LLM: "Summarize these descriptions into one"
   ~30-50 unique entities → ~30-50 LLM calls
     │
     ▼
④ COMMUNITY DETECTION  (no LLM — graph algorithm)
   Leiden algorithm clusters densely-connected entities
   e.g. {BM25, TF-IDF, Robertson} → "Sparse Retrieval" community
   Output: 5-10 communities at each level (levels 0, 1, 2...)
     │
     ▼
⑤ COMMUNITY SUMMARIZATION  ← LLM call per community per level
   For each community → LLM:
     "Summarize the themes, entities, and relationships in this cluster"
   10 communities × 3 levels ≈ 30 LLM calls
     │
     ▼
⑥ EMBEDDING  (no LLM — embedding API calls, not chat calls)
   Embed: entity descriptions, community full content, text units
   ~50 entities + ~10 communities + ~9 text units = ~70 embedding calls
```

**Total index LLM calls: ~9 + ~50 + ~30 = ~90 chat completions**
vs standard hybrid RAG: **0 LLM calls at index time** (only embedding calls)

---

#### Query Time (cheap — same cost structure as other strategies)

```
USER QUERY: "How does BM25 handle term frequency?"
     │
     ├─── LOCAL SEARCH ──────────────────────────────────────────┐
     │                                                           │
     │   ① ENTITY EMBEDDING  (1 embedding call)                 │
     │      Embed the query with the configured embedding model  │
     │                                                           │
     │   ② ENTITY MATCHING  (vector search in lancedb)          │
     │      Find top-k entity descriptions closest to query      │
     │      Result: [BM25, term frequency, TF-IDF]               │
     │                                                           │
     │   ③ GRAPH TRAVERSAL  (no LLM — pure graph lookup)        │
     │      Walk edges from matched entities:                    │
     │        BM25 → IMPROVES_UPON → TF-IDF                     │
     │        BM25 → INTRODUCES → term freq saturation          │
     │        BM25 → DEVELOPED_BY → Robertson                   │
     │      Collect neighboring nodes + relationships            │
     │      Fetch source text_units referenced by those nodes    │
     │                                                           │
     │   ④ CONTEXT ASSEMBLY  (no LLM)                           │
     │      Pack: entity data + relationships + text units       │
     │      into a context window                                │
     │                                                           │
     │   ⑤ GENERATION  (1 LLM call)                             │
     │      LLM generates answer from assembled context          │
     │                                                           │
     │   Total: 1 embedding + 1 LLM call                        │
     └───────────────────────────────────────────────────────────┘

     ├─── GLOBAL SEARCH ─────────────────────────────────────────┐
     │                                                           │
     │   ① MAP PHASE  (1 LLM call per relevant community)       │
     │      For each community report:                           │
     │        "Does this community help answer the query?"       │
     │        "If yes, write a partial answer from this cluster" │
     │      ~10 communities → ~10 LLM calls                     │
     │                                                           │
     │   ② REDUCE PHASE  (1 LLM call)                           │
     │      Synthesise all partial answers into final answer     │
     │                                                           │
     │   Total: ~11 LLM calls per query (no embedding needed)   │
     └───────────────────────────────────────────────────────────┘
```

---

#### Cost Comparison vs Standard Hybrid RAG

| Phase | Standard Hybrid RAG | GraphRAG Local | GraphRAG Global |
|---|---|---|---|
| Index | 0 LLM calls, N embedding calls | ~90 LLM calls + N embedding calls | same |
| Query | 1 LLM call (generation) | 1 embedding + 1 LLM call | ~11 LLM calls |
| Index cost ratio | 1× | **10–50×** | **10–50×** |
| Query cost ratio | 1× | ~1× | ~10× |

The index cost is where GraphRAG pays its premium. Query time for **local search** is nearly identical to standard RAG. **Global search** is expensive at query time because the map-reduce touches every community on every question.

---

### Step-by-Step Walkthrough with Concrete Examples

#### Step ②: Entity Extraction — What Actually Happens Per Chunk

GraphRAG reads this chunk from a corpus about retrieval:

> *"BM25 is a probabilistic ranking function developed by Robertson. It improves upon TF-IDF by adding term frequency saturation. Elasticsearch uses BM25 as its default ranking algorithm."*

LLM extracts:
```
Entities:
  - BM25 (algorithm): "A probabilistic ranking function with term frequency saturation"
  - Robertson (person): "Researcher who developed BM25"
  - TF-IDF (algorithm): "Earlier ranking function that BM25 improves upon"
  - Elasticsearch (system): "Search engine that uses BM25 as default"

Relationships:
  - BM25 → DEVELOPED_BY → Robertson
  - BM25 → IMPROVES_UPON → TF-IDF
  - Elasticsearch → USES → BM25
```

A different chunk from the same corpus says:

> *"BM25 was originally called Okapi BM25, named after the Okapi IR system at City University London, where Robertson worked."*

LLM extracts another partial description of BM25 and Robertson. After processing all chunks, there are now **4 different partial descriptions of BM25** collected from 4 different chunks.

---

#### Step ③: Entity Summarization — Why It Is Needed

Without summarization, BM25 would exist as 4 fragmented nodes in the graph — each carrying a partial description. You can't answer "what is BM25?" cleanly because the answer is split across 4 places.

GraphRAG calls the LLM once per unique entity:

> *"Here are 4 descriptions of BM25 collected from different parts of the corpus. Write one unified description."*

LLM output:
```
BM25 (unified): "A probabilistic ranking function, also known as Okapi BM25,
developed by Robertson at City University London. It improves upon TF-IDF
by adding term frequency saturation (repeated terms gain diminishing returns),
IDF weighting (rare terms score higher), and document length normalization
(preventing long documents from winning by sheer size)."
```

Now there is **exactly one BM25 node** in the graph with a complete, accurate description. Without this step, every entity is a fragmented half-description and the graph is noisy.

---

#### Steps ④ + ⑤: Community Detection and Summarization — The Three Levels Explained

After entity summarization the full graph looks like this (simplified):

```
[BM25] ── IMPROVES_UPON ──► [TF-IDF]
[BM25] ── DEVELOPED_BY ──► [Robertson]
[BM25] ── USES ──────────► [IDF weighting]
[FAISS] ── IS_A ─────────► [vector index]
[ChromaDB] ── IS_A ──────► [vector index]
[cosine similarity] ── USED_IN ──► [dense retrieval]
[dense retrieval] ── CONTRASTS_WITH ──► [BM25]
[RAGAS] ── MEASURES ─────► [faithfulness]
[RAGAS] ── MEASURES ─────► [answer_relevancy]
```

The Leiden algorithm looks at density of connections — nodes heavily linked to each other form a natural cluster. It produces communities at multiple levels of granularity, like zooming in and out on a map:

**Level 0 — most granular (many small clusters):**
```
Community A: BM25, TF-IDF, IDF weighting, term freq saturation, Robertson
Community B: FAISS, ChromaDB, Qdrant, vector index, cosine similarity
Community C: RAGAS, faithfulness, answer_relevancy, answer_similarity
Community D: dense retrieval, sparse retrieval, hybrid search, RRF
Community E: chunking, chunk size, overlap, semantic chunker
... (~15–20 communities)
```

**Level 1 — medium granularity (clusters begin to merge):**
```
Community A+D: All retrieval methods (sparse + dense + hybrid merge)
Community B:   All vector stores
Community C:   All evaluation
Community E:   All preprocessing
... (~8–10 communities)
```

**Level 2 — coarsest (broadest themes):**
```
Community 1: "Retrieval and Ranking" (sparse + dense + hybrid + vector stores)
Community 2: "RAG Pipeline Quality" (evaluation + generation)
Community 3: "Document Processing" (chunking + embeddings + ingestion)
... (~3–5 communities)
```

Different questions need different granularity:
- "What is BM25?" → Level 0 answer (very specific cluster)
- "How do retrieval methods compare?" → Level 1 answer (medium cluster)
- "What are the main themes of this corpus?" → Level 2 answer (broadest clusters)

For each community at each level, the LLM is called once to write a community summary. **10 communities × 3 levels = 30 LLM calls.** These community summaries are what global search reads at query time.

---

#### Step ⑥: Embedding — What Exactly Gets Embedded and Why

By this point there are no embeddings yet. Steps ①–⑤ produced text: entity descriptions and community summaries. Embedding converts that text into vectors so that fast similarity search is possible at query time.

Three things get embedded, each serving a distinct purpose:

**1. Entity descriptions** → stored in LanceDB (`entity_description.lance`)

Each entity's unified description is embedded:
```
embed("A probabilistic ranking function, also known as Okapi BM25,
       developed by Robertson at City University London...")
→ [0.12, -0.34, 0.91, ...]  (1536-dim vector, one per entity)
```

Role: **primary retrieval — the entry point into the graph.** When a query arrives, it is embedded and compared against these vectors. The closest entities become the starting nodes for graph traversal. This is the only similarity search in local search's primary retrieval path.

**2. Community full content** → stored in LanceDB

Each community's full content is embedded. Role: **entry point for community-level search modes** (drift search, basic search). Global search does not use these embeddings — it reads community report text directly via map-reduce, no similarity search.

**3. Text units (source chunks)** → stored in LanceDB

The original raw chunks are embedded. Their role is **not** primary retrieval — and this is a nuance worth understanding precisely:

- In **local search**: after graph traversal, matched entities may collectively reference dozens of source chunks. The context window has a fixed token budget. Text unit embeddings are used to *rank which of those chunks are most query-relevant* — the query embedding is compared against candidate chunk embeddings and only the top-scoring ones fill the allocated context budget (`text_unit_prop`, default 50% of context). This is secondary selection, not primary retrieval. The chunks themselves are fetched by chunk_id from the node references — not by similarity search.
- In **global search**: text unit embeddings are not used at all. Community summaries are the context; raw chunks never appear.
- In **basic search** (GraphRAG's third mode, rarely discussed): text unit embeddings ARE the primary retrieval mechanism — this mode is essentially standard dense retrieval against chunks, no graph traversal at all. It exists as a fallback for simple factual queries.

**Role of text unit embeddings by search mode:**

| Search mode | Text unit embeddings |
|---|---|
| Local search | Secondary — ranks candidate chunks for context window selection |
| Global search | Not used |
| Basic search | Primary — standard vector similarity retrieval |

---

#### The Full Mental Model in One Diagram

```
INDEX TIME                            QUERY TIME (local search)
──────────                            ──────────────────────────
Chunks                                Query
  │                                     │
  ▼ LLM × chunks                        ▼ 1 embedding call
Entity extraction                    Embed query
  │                                     │
  ▼ LLM × entities                      ▼ vector search in lancedb
Entity summarization                 Match entity descriptions
  │                                  → find: [BM25, TF-IDF, Robertson]
  ▼ graph algorithm                      │
Graph construction                       ▼ no LLM — pure graph lookup
  │                                  Graph traversal from matched entities
  ▼ graph algorithm                  → collect subgraph + source chunks
Community detection                      │
  │                                      ▼ no LLM
  ▼ LLM × communities × levels      Assemble context window
Community summarization              (subgraph + relationships + chunks)
  │                                      │
  ▼ embedding API                        ▼ 1 LLM call
Embed entity descriptions,           Generate answer
  community content, text units
  (stored in LanceDB)
```

The embeddings in step ⑥ are the **bridge between index time and query time** — they allow "find the relevant starting point in the graph for this query" to work as a fast vector lookup rather than scanning every entity description with an LLM.

***

## 13. How to Use This Retrieval Cheat‑Sheet

1. **Start here — basic to production:**
   - Basic vector retriever → add **Hybrid Search** (BM25 + dense + RRF) → your retrieval is now production-grade
   - Add **MultiQueryRetriever** for better recall on ambiguous or synonym-heavy queries
   - Add **ContextualCompressionRetriever** to trim noisy context before it reaches the LLM
2. **When retrieval quality is still poor:**
   - Try **HyDE** if queries are short and don’t match document vocabulary
   - Try **Parent Document Retrieval** if answers feel incomplete or miss nuance
   - Try **Query Decomposition** if questions are multi-part or span multiple topics
3. **For multi‑index / multi‑strategy setups:**
   - Use **RouterQueryEngine** to route between summarization, vector retrieval, and keyword indices
4. **For relational or global questions:**
   - Use **GraphRAG** when entity relationships matter and vector search produces shallow answers
5. **For managed search/RAG:**
   - Use **Bedrock KB** or **Vertex AI Search** and focus on prompt-time fusion and evaluation, not low-level plumbing

Keep this Markdown alongside your other component cheat‑sheets so you and your agents can wire up or experiment with retrieval strategies quickly while building RAC/RAG systems.

---

## 14. Insights from Practice — What Experimentation Revealed

This section captures the non-obvious insights, cross-cutting realizations, and hard-won lessons from building and evaluating each retrieval strategy. These are the things that don't appear in documentation but become obvious only after you run the experiments.

---

### 14.1 The Root Problem All Retrieval Strategies Are Solving

> **The retrieval problem is fundamentally a representation alignment problem.**

Dense retrieval only works well when query and document representations live in compatible regions of the embedding space. Every advanced retrieval strategy — HyDE, multi-query, instruction-tuned embeddings, hybrid — is just a different approach to solving that alignment problem from a different angle:

| Strategy | What alignment problem it fixes |
|---|---|
| Multi-query | Broadens the query's coverage of embedding space — vocabulary gap |
| HyDE | Transforms the query into document-space before searching — format gap |
| Instruction prefixes (`query:` vs `passage:`) | Nudges query and doc vectors closer — asymmetric encoding gap |
| Fine-tuned embeddings | Trains the model to align them explicitly — domain gap |
| Hybrid (BM25 + dense) | Covers what dense misses with exact keyword matching — semantic vs lexical gap |

When a retrieval strategy underperforms, the first question is: which alignment gap is it failing to close on this corpus?

---

### 14.2 BM25 — When Keywords Beat Semantics

**Key insight:** BM25 has zero concept of synonyms — "fast" and "quick" are completely different tokens. But on corpora written in consistent technical vocabulary, this is a strength, not a weakness. Technical questions use the same terminology as the documents that answer them.

**Three signals BM25 uses:**
1. **Term Frequency (TF)** — how often the query token appears in the chunk, with saturation (doubling occurrences doesn't double the score; `k1=1.5`)
2. **Inverse Document Frequency (IDF)** — rare tokens across the corpus score higher. "embeddings" beats "the"
3. **Length normalization** — long chunks are penalized so they don't win purely by size (`b=0.75`)

**Experimental evidence (on a technical RAG corpus):**
- Dense baseline answer_similarity: 0.538
- BM25-only answer_similarity: 0.788 (+0.250)

BM25 substantially outperformed dense retrieval on a corpus where the test questions used the exact same technical vocabulary as the documents. This is not always the case — it's a corpus property.

**Where BM25 wins:** exact product codes, rare technical terms, proper nouns, version numbers, any query where the answer uses the exact same tokens as the question.

**Where BM25 fails:** synonyms, paraphrases, conceptual questions ("what improves factuality?" when the document says "grounding reduces hallucinations").

---

### 14.3 Hybrid Search — A Multiplier, Not a Fix

**The critical misunderstanding about hybrid retrieval:** switching from dense-only to hybrid does not guarantee improvement. Hybrid is a fusion layer — it can only amplify signal that already exists in the individual legs.

**Alpha-weighted fusion vs RRF — why RRF won:**

| Approach | Problem |
|---|---|
| Alpha-weighted (`score = α × dense + (1-α) × bm25`) | BM25 scores and cosine similarity have no stable shared scale. A BM25 score of 8.3 on one query might be the maximum; on another it's mediocre. Alpha tuning is fragile. |
| RRF (`score = Σ 1/(k + rank_i)`) | Only uses rank position — completely scale-independent. No normalization needed. No tuning. `k=60` works universally. |

**The film critic analogy:** if one critic scores 3–7 and another scores 7–9.5, averaging their raw scores is lying to you. RRF says: forget the scores, just look at which films each ranked highest. That consensus is the signal.

**The failure mode (experimental):**

| Retriever | answer_similarity |
|---|---|
| Dense only (baseline) | 0.538 |
| BM25 only | 0.788 |
| Hybrid (RRF) | 0.722 |

Hybrid scored **below BM25 alone**. Why: the dense leg (0.538) was weak. It was placing the correct chunk at rank 5 or 6, but confidently placing a false positive at rank 1. RRF has no way to know the dense leg's rank-1 is noise — it treats it as valid signal. The false positive accumulated enough RRF score to outrank BM25's correct result.

**The production readiness test for hybrid:**
1. Fix chunking → both legs benefit simultaneously
2. Upgrade embedding model → strengthen the dense leg
3. Improve BM25 tokenization (stopwords, stemming) → strengthen the sparse leg
4. Run a complementarity audit: for each test question, did dense get it right? Did BM25? You want a meaningful percentage of questions where one got it and the other missed — **complementary failure modes**
5. Only then fuse — now RRF is combining two strong, complementary signals

> Hybrid earns its place only when both legs are actually good — and failing on different questions.

---

### 14.4 Multi-Query — Vocabulary Gap vs Knowledge Gap

**The distinction that matters:**
- **Multi-query** generates rephrasings of the same question → covers **vocabulary gaps** (the corpus uses different words than the query)
- **Query decomposition** breaks a complex question into sub-questions → covers **knowledge gaps** (no single chunk can answer all parts)

These solve completely different problems. Using multi-query for a multi-part question doesn't help — you're just hitting the same knowledge gap from different angles.

**LangChain vs LlamaIndex vs custom — what actually differs:**

| Implementation | Fusion method | Score exposure |
|---|---|---|
| Custom (`MultiQueryRetriever`) | RRF across N result lists | RRF scores |
| LangChain (`MultiQueryRetriever`) | Union + dedup only (no cross-query boosting) | Placeholder 1.0 |
| LlamaIndex (`QueryFusionRetriever`) | RECIPROCAL_RANK (RRF) | Original cosine scores |

**Cost reality:** each query becomes 1 LLM call (rephrase) + N parallel retrieval calls. In production, used selectively — internal search tools, low-volume high-stakes queries, async workflows. Skipped for high-volume consumer products where cost multiplies directly with user count.

---

### 14.5 HyDE — The Representation Gap Made Visible

**The insight that makes HyDE click:**

When we index documents, we are indexing ground-truth answers. When a user queries, they send a question — which by definition does not know the answer, only the keywords or intent. Questions and answers are representationally asymmetric — they encode into different regions of embedding space even when semantically related.

HyDE's fix: generate an answer-shaped text, embed that instead of the query. Answer-to-answer similarity is a much tighter match than question-to-answer.

> The hypothetical answer doesn't need to be factually correct. It just needs the right shape and vocabulary to point the embedding vector toward the correct neighborhood in the corpus.

**The known failure mode (experimental):**

| Metric | Baseline | HyDE | Delta |
|---|---|---|---|
| faithfulness | 0.641 | 0.791 | +0.150 |
| answer_relevancy | 0.583 | 0.521 | -0.062 |
| answer_similarity | 0.538 | 0.659 | +0.121 |

HyDE improved document-space alignment (faithfulness and similarity up) but hurt question-answer alignment (relevancy down). The LLM generating the hypothetical answer subtly reframed the intent of the original question. The embedding landed in the right neighborhood of the corpus, but not the answer to what was actually asked.

**The fix:** average the hypothetical answer embedding with the original query embedding before searching. This preserves both alignments simultaneously.

---

### 14.6 Query Decomposition — When No Single Chunk Has the Full Answer

**Core use case:** questions with "and", comparative structure ("how does X differ from Y"), or multi-hop structure ("why does X cause Y, and how does that affect Z").

**Parallel vs sequential:**
- **Parallel** — sub-questions are independent, retrieved simultaneously. Covers most production cases.
- **Sequential** — answer to sub-Q1 informs sub-Q2. Required for multi-hop reasoning chains. Cannot be parallelized.

**On a simple, well-scoped test set the delta will be modest** — decomposition shows its value when the corpus and question set genuinely have multi-part structure. The mechanism is sound; the corpus needs to exercise it.

---

### 14.7 Parent Document Retrieval — The Precision-vs-Context Tradeoff

**The tradeoff all fixed-size chunking forces:**
- Small chunks → precise embedding (one focused idea) → thin context for generator
- Large chunks → rich context for generator → diluted embedding (too many ideas averaged)

Parent document retrieval solves this by decoupling the index unit from the generation unit:
- **Index small** (128 tokens) → precise retrieval
- **Return large** (512 tokens, the parent) → rich generation context

**Production storage patterns — the dict is only for the lab:**

| Pattern | Vector store | Parent store | When to use |
|---|---|---|---|
| Qdrant single-store | Qdrant (child vectors + parent text in payload) | None (payload is the store) | Smaller corpora, simpler infra |
| ChromaDB + Redis | ChromaDB (child vectors) | Redis (parent_id → parent text) | Production scale — stores scale independently |
| Vector DB + Postgres/Mongo | Any vector DB | Postgres/Mongo/DynamoDB | Enterprise — existing infra already present |

The two-store architecture (vector DB + doc store) is the production standard because the two stores scale independently: grow the Redis cluster for larger corpora without rebuilding the vector index.

---

### 14.8 GraphRAG — When the Answer Isn't in Any Single Chunk

**The class of questions chunk-based retrieval fundamentally cannot answer:**
- "What themes connect X and Y across these documents?"
- "How did this approach evolve over these reports?"
- "Who is connected to this entity and through what chain?"

The answer to these questions doesn't live in a chunk — it emerges from the **relationships** between entities across the corpus.

**The graph data model — physically two tables:**

```
Nodes table: node_id | label       | properties (JSON)
Edges table: edge_id | from_node   | to_node | relationship | properties (JSON)
```

Every node physically stores a pointer to its edges (index-free adjacency). Finding everything connected to a node is a pointer follow, not a table scan — this is why graph traversal stays fast.

**Classical NLP vs LLM-based entity extraction:**

| Approach | Speed | Cost | Flexibility |
|---|---|---|---|
| spaCy / NLTK / dependency parsing | Fast | Free | Fixed entity/relation types |
| BERT-based relation classifiers | Medium | Cheap | Narrow domains |
| LLM extraction (GraphRAG default) | Slow | Expensive | Open-ended, any domain |

LLMs became default for GraphRAG because real-world corpora have messy, ambiguous language that rule-based systems can't handle. Classical methods work on structured domains (financial filings, medical records) with consistent vocabulary.

**Two search modes in Microsoft's GraphRAG:**
- **Local search** — traverse from specific entities. Best for: "what is X?", "how does X relate to Y?"
- **Global search** — read LLM-generated community summaries (clusters of related entities). Best for: "what are the main themes?", "give me an overview of this corpus." **This is impossible with any chunk-based system.**

**The mental model:**
- Chunk-based RAG = library. Find the right book. Return the relevant passage.
- GraphRAG = encyclopedia with a cross-reference index. The relationships between concepts are pre-mapped. Questions about connections, themes, and cross-document synthesis are answerable because the structure is explicit.

**When NOT to use:** simple factual lookups, expository corpora without dense entity relationships, budget-constrained situations (index cost is real — 10-50x more LLM calls than standard RAG indexing), or latency-sensitive applications.

---

### 14.9 Cross-Cutting Rules of Thumb

These apply regardless of which strategy you're evaluating:

1. **answer_similarity is the most reliable eval signal** — it's deterministic (cosine sim vs ground truth), not subject to LLM judge variance. If this moves, the improvement is real.

2. **Corpus vocabulary consistency determines which retriever wins.** On technical corpora where questions reuse document terminology, BM25 and its variants outperform dense retrieval. On general/conversational corpora, dense retrieval has the edge.

3. **Every strategy adds an LLM call overhead at query time** (except BM25 and hybrid). Budget these before choosing a strategy for production.

4. **Index design decisions compound.** Better chunking improves both BM25 (cleaner token boundaries) and dense retrieval (coherent embeddings). Fix chunking before tuning retrieval strategy.

5. **A retrieval strategy is only as good as its weakest component.** Hybrid with a weak dense leg performs worse than BM25 alone. GraphRAG with poor entity extraction produces a noisy graph. Strengthen each component before composing them.

---

## References

1. [Building Production-Ready RAG Applications with LangChain v0.3](https://krishcnaik.substack.com/p/building-production-ready-rag-applications) - Introduction: Why RAG Matters in 2025. Retrieval-Augmented Generation (RAG) has emerged as the corne...

2. [MultiQueryRetriever — LangChain documentation](https://reference.langchain.com/v0.3/python/langchain/retrievers/langchain.retrievers.multi_query.MultiQueryRetriever.html)

3. [Improving Document Retrieval with Contextual Compression](https://blog.langchain.com/improving-document-retrieval-with-contextual-compression/) - A simple example of this is you may want to combine a TextSplitter and an EmbeddingsFilter to first ...

4. [Implement Contextual Compression And Filtering In RAG ...](https://medium.aiplanet.com/implement-contextual-compression-and-filtering-in-rag-pipeline-4e9d4a92aa8f) - The Contextual Compression Retriever passes queries to the base retriever,; It then takes the initia...

5. [Router Query Engine | LlamaIndex OSS Documentation](https://developers.llamaindex.ai/python/examples/workflow/router_query_engine/) - This notebook walks through implementation of Router Query Engine, using workflows. Specifically we ...

6. [Router Query Engine#](https://llamaindexxx.readthedocs.io/en/latest/examples/query_engine/RouterQueryEngine.html)

7. [Router Query Engine with Mistral AI and LlamaIndex](https://docs.mistral.ai/cookbooks/third_party-llamaindex-routerqueryengine) - Learn Router Query Engine with Mistral AI and LlamaIndex with practical examples and code snippets u...

8. [Create a knowledge base by connecting to a data source in Amazon ...](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-create.html) - To create a knowledge base, send a CreateKnowledgeBase request with an Agents for Amazon Bedrock bui...

9. [Configure and use the vertex...](https://docs.langchain.com/oss/python/integrations/retrievers/google_vertex_ai_search) - Integrate with the Google Vertex AI search retriever using LangChain Python.

10. [LangChain v1 migration guide](https://docs.langchain.com/oss/python/migrate/langchain-v1) - All LangChain packages now require Python 3.10 or higher. Python 3.9 reaches end of life in October ...

11. [langchain.retrievers.multi_query](https://reference.langchain.com/v0.3/python/_modules/langchain/retrievers/multi_query.html) - [docs] class MultiQueryRetriever(BaseRetriever): """Given a query, use an LLM to write a set of quer...

12. [ContextualCompressionRetriever](https://python.langchain.com/api_reference/langchain/retrievers/langchain.retrievers.contextual_compression.ContextualCompressionRetriever.html)

