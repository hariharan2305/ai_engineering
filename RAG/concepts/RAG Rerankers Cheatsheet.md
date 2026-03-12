# Re‑ranking Components – Code Cheat‑Sheet (2026)

This cheat‑sheet gives **component‑level Python snippets** for every reranker listed in your component stack guide:

- Cohere Rerank 3 / 3.5 (and v4 models) via Cohere API and LangChain
- BGE‑Reranker (base / large / v2 variants) via FlagEmbedding
- RankGPT / LLM‑based reranking via LlamaIndex node postprocessors
- Qwen3 Reranker models via Hugging Face + `transformers`
- Guidance on **when** to add reranking and how to plug these rerankers into your RAC/RAG pipeline

All APIs are based on the latest public docs as of early 2026.[^1][^2][^3][^4][^5][^6][^7][^8][^9]

> The typical pattern is: **retrieval → reranking → prompt**. You pass the query + a list of candidate documents/passages to the reranker, get relevance scores, sort by score, and keep the top‑N for the LLM.

***

## 1. Cohere Rerank 3 / 3.5 / v4 – Cohere API

Cohere’s Rerank models (e.g., `rerank-english-v3.0`, `rerank-multilingual-v3.0`, `rerank-v4.0-pro`) take a **query + list of documents** and return a sorted list with relevance scores.[^6][^1]

### 1.1 Install and initialize client

```bash
pip install -U cohere
```

```python
import cohere

co = cohere.Client("YOUR_COHERE_API_KEY")
```

### 1.2 Basic rerank call (top‑k passages)

```python
query = "What are the benefits of RAG for enterprise search?"

documents = [
    "RAG improves factuality by grounding the LLM in retrieved documents.",
    "This paragraph is about cooking recipes.",
    "Enterprises can use RAG to search across knowledge bases and documents.",
]

results = co.rerank(
    model="rerank-english-v3.0",   # or "rerank-multilingual-v3.0", "rerank-v4.0-pro"
    query=query,
    documents=documents,
    top_n=2,
)

for idx, r in enumerate(results.results):
    print(f"Rank: {idx+1}")
    print("Score:", r.relevance_score)
    print("Document:", documents[r.index])
    print("---")
```

This matches the current Cohere Rerank quickstart: `co.rerank(model=..., query=..., documents=..., top_n=...)`.[^1][^6]

### 1.3 Plugging into a RAG pipeline

1. Retrieve the top‑K docs from your vector DB (e.g., Qdrant, Weaviate).
2. Pass their **page_content** strings to `co.rerank` with the user query.
3. Sort the docs by `relevance_score` and keep the top N for your LLM prompt.

***

## 2. Cohere Rerank via LangChain (`CohereRerank`)

LangChain provides a `CohereRerank` document compressor that you can plug into a `ContextualCompressionRetriever`.[^8]

### 2.1 Install

```bash
pip install -U langchain-core langchain-cohere langchain-openai
```

### 2.2 Wrap a base retriever with CohereRerank

```python
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

os.environ["COHERE_API_KEY"] = "YOUR_COHERE_API_KEY"

# Build a basic vectorstore + retriever
docs = [
    Document(page_content="RAG helps improve factual accuracy by grounding answers in data."),
    Document(page_content="This text talks about sports and has nothing to do with RAG."),
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(docs, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Cohere reranker compressor
cohere_reranker = CohereRerank(
    model="rerank-english-v3.0",  # required per docs
    top_n=3,
)

compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=cohere_reranker,
)

query = "How does RAG improve factuality?"
reranked_docs = compression_retriever.invoke(query)

for d in reranked_docs:
    print("--- Reranked Doc ---")
    print(d.page_content)
```

This follows the current LangChain `CohereRerank` integration guide.[^8]

***

## 3. BGE‑Reranker (base / large / v2) – FlagEmbedding

BGE‑Reranker models are cross‑encoders tuned to pair with BGE embeddings and achieve SoTA reranking performance on MTEB tasks.[^10][^7][^9]

### 3.1 Install FlagEmbedding

```bash
pip install -U FlagEmbedding
```

### 3.2 Basic usage with `FlagReranker`

From the BGE Reranker and `bge-reranker-large` docs:[^7][^9]

```python
from FlagEmbedding import FlagReranker

# Use GPU with FP16 if available
reranker = FlagReranker(
    "BAAI/bge-reranker-large",   # or "BAAI/bge-reranker-base", "BAAI/bge-reranker-v2-m3", etc.
    use_fp16=True,
    devices=["cuda:0"],
)

query = "What is the capital of France?"
passages = [
    "Paris is the capital of France.",
    "China has a population of over 1.4 billion people.",
]

# Single score for one pair
score = reranker.compute_score([query, passages])
print("Score for passage 0:", score)

# Batch scores for multiple (query, passage) pairs
pairs = [
    [query, passages],
    [query, passages[^1]],
]

scores = reranker.compute_score(pairs)
print("Scores:", scores)
```

### 3.3 Integrate BGE‑Reranker in your RAG pipeline

Pseudo‑pattern:

```python
candidates = vector_retriever.invoke(user_query)  # list[Document]

pairs = [[user_query, doc.page_content] for doc in candidates]

scores = reranker.compute_score(pairs)

# Sort docs by score descending
ranked = sorted(zip(candidates, scores), key=lambda x: x[^1], reverse=True)

top_docs = [doc for doc, _ in ranked[:5]]
```

You can plug this into LangChain via a custom `DocumentCompressor` or into LlamaIndex via a custom node postprocessor.

***

## 4. RankGPT / LLM‑Based Reranking – LlamaIndex

LlamaIndex exposes multiple LLM‑based rerankers as **node postprocessors**, including generic LLM Rerank, RankGPT, ColBERT Reranker, and rankLLM.[^2][^4]

### 4.1 General node postprocessor pattern

From the Node Postprocessor docs:[^4]

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.postprocessor import SimilarityPostprocessor

# Load docs and build index
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Example: similarity cutoff postprocessor
similarity_processor = SimilarityPostprocessor(similarity_cutoff=0.75)

query_engine = index.as_query_engine(
    node_postprocessors=[similarity_processor],
)

response = query_engine.query("Your question here")
print(response)
```

### 4.2 Using RankGPT Reranker (LLM‑based)

The RankGPT postprocessor uses an LLM agent to score and rerank nodes.[^11][^2]

> The RankGPT API is documented as a beta node postprocessor in LlamaIndex; its usage is similar to other postprocessors.

A typical pattern (simplified to illustrate wiring):

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank

llm = LlamaOpenAI(model="gpt-4.1-mini", temperature=0)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Build index
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# RankGPT node postprocessor
rankgpt_reranker = RankGPTRerank(
    llm=llm,
    top_n=5,
)

query_engine = index.as_query_engine(
    similarity_top_k=20,                 # first stage retrieval
    node_postprocessors=[rankgpt_reranker],  # second stage rerank
)

response = query_engine.query("Explain our refund policy for EU customers")
print(response)
```

Check the exact import path and parameters (`RankGPTRerank` or similarly named) in the current LlamaIndex node postprocessor docs, as they evolve quickly.[^2][^4]

### 4.3 Generic LLM Rerank postprocessor

LlamaIndex’s generic LLM Rerank module (`LLMRerank`) uses an LLM to select and reorder nodes.[^2]

```python
from llama_index.postprocessor.llm_rerank import LLMRerank

llm_reranker = LLMRerank(
    llm=llm,
    top_n=5,
)

query_engine = index.as_query_engine(
    similarity_top_k=20,
    node_postprocessors=[llm_reranker],
)
```

Use RankGPT/LLM Rerank when you want **maximum quality** and can afford extra LLM calls.

***

## 5. Qwen3 Rerankers – Hugging Face + Transformers

The Qwen3 Embedding series exposes dedicated **reranking models** such as `Qwen3-Reranker-0.6B`, `Qwen3-Reranker-4B`, and others.[^3][^5][^12]

These are used like standard cross‑encoder rerankers via Hugging Face `transformers`.

### 5.1 Install

```bash
pip install -U transformers torch
```

### 5.2 Simple scoring loop with Qwen3 Reranker

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "Qwen/Qwen3-Reranker-0.6B"  # or 4B/8B depending on resources

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

query = "What are the use cases of retrieval-augmented generation?"
passages = [
    "RAG improves factuality and grounding by retrieving external documents.",
    "This passage is only about image generation and style transfer.",
]

scores = []
for p in passages:
    inputs = tokenizer(query, p, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Most rerankers put relevance in the single logit or the last dimension
        logits = outputs.logits
        score = float(logits.squeeze())
        scores.append(score)

# Sort passages by score descending
ranked = sorted(zip(passages, scores), key=lambda x: x[^1], reverse=True)

for text, score in ranked:
    print(f"Score: {score:.4f}\n{text}\n---")
```

Refer to the specific Hugging Face model card for any special output handling or normalization.[^12][^3]

You can wrap this in a LangChain or LlamaIndex compressor/postprocessor class to use in your RAG pipeline.

***

## 6. When to Add Reranking (and How)

From your component guide:

- **Add reranking** when you see **off‑topic context** in prompts even though retrieval appears reasonable; reranking helps surface the truly relevant passages to the top.
- For **very large corpora**, always use a reranker (Cohere, BGE‑Reranker, Qwen3 rerankers, or LLM‑based RankGPT) after the first‑stage ANN search to refine top‑K results before sending them to the LLM.[^9][^10][^7]

### 6.1 Practical wiring pattern

1. **First‑stage retrieval**: fast ANN retriever (Qdrant, Weaviate, Milvus, Pinecone, pgvector, OpenSearch, Bedrock KB, Vertex AI Search).
2. **Second‑stage rerank**: pick one of:
   - Cohere Rerank via API or LangChain `CohereRerank`.[^6][^8]
   - BGE‑Reranker / FlagReranker for open‑source cross‑encoder reranking.[^10][^7][^9]
   - Qwen3 Reranker for multilingual and high‑capacity use cases.[^5][^3]
   - RankGPT / LLMRerank in LlamaIndex when quality is more critical than latency.[^4][^2]
3. **Select top‑N** passages based on reranker scores.
4. **Compose the prompt** to the generation LLM with only these top‑N passages, often ordered by score.

Use this cheat‑sheet alongside your retrieval and embedding cheat‑sheets so you and your agents can quickly swap in different rerankers and evaluate their impact on answer quality.

---

## References

1. [Master Reranking with Cohere Models](https://docs.cohere.com/docs/reranking-with-cohere) - This page contains a tutorial on using Cohere's ReRank models.

2. [Node Postprocessor Modules | LlamaIndex OSS Documentation](https://developers.llamaindex.ai/python/framework/module_guides/querying/node_postprocessors/node_postprocessors/) - Uses RankGPT agent to rerank documents according to relevance. Returns the top N ranked nodes. from ...

3. [Qwen/Qwen3-Reranker-0.6B - Hugging Face](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) - We’re on a journey to advance and democratize artificial intelligence through open source and open s...

4. [Node Postprocessor#](https://llamaindexxx.readthedocs.io/en/latest/module_guides/querying/node_postprocessors/root.html)

5. [Qwen3 Embedding: Advancing Text Embedding and Reranking ...](https://qwenlm.github.io/blog/qwen3-embedding/) - These models are specifically designed for text embedding, retrieval, and reranking tasks, built on ...

6. [Reranking - quickstart - Cohere Documentation](https://docs.cohere.com/docs/reranking-quickstart) - A quickstart guide for performing reranking with Cohere's Reranking models (v2 API).

7. [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) - For examples, use bge embedding model to retrieve top 100 relevant documents, and then use bge reran...

8. [Cohere reranker integration - Docs by LangChain](https://docs.langchain.com/oss/python/integrations/retrievers/cohere-reranker) - Integrate with the Cohere reranker retriever using LangChain Python.

9. [BGE-Reranker — BGE documentation](https://bge-model.com/bge/bge_reranker.html) - Usage#. from FlagEmbedding import FlagReranker reranker = FlagReranker( 'BAAI/bge-reranker-base', qu...

10. [FlagEmbedding项目中的BGE Reranker技术详解与应用指南](https://blog.csdn.net/gitblog_00697/article/details/148417277) - 文章浏览阅读695次，点赞5次，收藏5次。在信息检索和自然语言处理领域，reranker（重排序器）扮演着至关重要的角色。FlagEmbedding项目中的BGE Reranker系列为开发者提供了一...

11. [Elasticsearch Reranker & LlamaIndex RankGPT: Usage & examples](https://www.elastic.co/search-labs/blog/elasticsearch-reranker-llamaindex-rankgpt) - In this article, we will explore the LlamaIndex RankGPT Reranker, which is a RankGPT reranker implem...

12. [Qwen/Qwen3-Reranker-4B - Hugging Face](https://huggingface.co/Qwen/Qwen3-Reranker-4B) - The Qwen3 Embedding model series is the latest proprietary model of the Qwen family, specifically de...

