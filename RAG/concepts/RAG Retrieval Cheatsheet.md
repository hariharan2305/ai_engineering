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

