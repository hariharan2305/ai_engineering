# Prompting, Context Compression & Grounding – Code Cheat‑Sheet (2026)

This cheat‑sheet covers the **prompt‑time tools** from your component guide and how to use them in code:

- LLMLingua (and LLMLingua‑2) – Python prompt compression
- LongLLMLingua – LlamaIndex node postprocessor integration for RAG
- xRAG – what it is and what its (research‑grade) integration would look like
- ACC‑RAG – how to approximate the paper’s adaptive compression pattern
- Practical prompt templates for grounded RAG with citations

Sources include the official LLMLingua project page, LlamaIndex blog & examples, and xRAG / ACC‑RAG papers.[^1][^2][^3][^4][^5][^6]

> These utilities sit **after retrieval, before generation**: input is (query, retrieved_docs), output is compressed/rewritten context or embeddings for the LLM.

***

## 1. LLMLingua / LLMLingua‑2 – Python Prompt Compression

LLMLingua is a small encoder‑style model (BERT‑scale) that **drops low‑importance tokens** from a prompt while preserving answer quality, often achieving 5–20× compression.[^5][^1]

### 1.1 Install

```bash
pip install -U llmlingua
```

(If the package name changes to `llmlingua2` in the future, follow the README on the GitHub repo.)[^5]

### 1.2 Minimal Python example – compress a RAG prompt

```python
from llmlingua import LLMLingua

# Initialize compressor
compressor = LLMLingua(
    model_name="microsoft/llmlingua-2-xsmall-llama",  # check README for latest recommended models
    device="cuda",  # or "cpu"
)

# Build a long RAG-style prompt
retrieved_chunks = [
    "Doc1: ... long text ...",
    "Doc2: ... long text ...",
]

context = "\n\n".join(retrieved_chunks)

prompt = (
    "You are a helpful assistant. Answer the question using ONLY the context.\n\n"
    f"Context:\n{context}\n\nQuestion: Explain our refund policy for EU customers."
)

# Compress prompt to ~25% of original length
compressed = compressor.compress_prompt(
    prompt,
    rate=0.25,               # target compression rate
    keep_head=True,
    keep_tail=True,
)

print("Original length:", len(prompt.split()))
print("Compressed length:", len(compressed.split()))
print("Compressed prompt:\n", compressed[:500])
```

The exact function name (`compress_prompt`, `compress`) and arguments may differ slightly; check the GitHub README for the latest.[^1][^5]

### 1.3 Using LLMLingua in a RAG call

The pattern is:

1. Build your **RAG prompt** from query + retrieved docs.
2. Run LLMLingua to compress to a target **token budget**.
3. Send the compressed prompt to your LLM (OpenAI, Claude, Gemini, etc.).

```python
# After computing `compressed`
from openai import OpenAI

client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": compressed}],
)

print(resp.choices.message.content)
```

This is exactly how LLMLingua is used in the RAG examples in the project repo and blog.[^6][^1][^5]

***

## 2. LongLLMLingua – LlamaIndex Integration

LongLLMLingua extends LLMLingua specifically for **long‑context + RAG** and is integrated into LlamaIndex as a **node postprocessor**.[^7][^2][^6]

### 2.1 Install LlamaIndex + LLMLingua

```bash
pip install -U llama-index-core llama-index-llms-openai llama-index-embeddings-openai
pip install -U llmlingua
```

### 2.2 LongLLMLingua as a LlamaIndex node postprocessor

Based on the LlamaIndex blog recipe and the LLMLingua RAGLlamaIndex example:[^2][^7][^6]

```python
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.longllmlingua import LongLLMLinguaPostprocessor

os.environ["OPENAI_API_KEY"] = "sk-..."

# 1) Load docs and build a VectorStoreIndex
documents = SimpleDirectoryReader("./data").load_data()

llm = LlamaOpenAI(model="gpt-4.1-mini", temperature=0)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# 2) Configure LongLLMLingua postprocessor

long_llmlingua = LongLLMLinguaPostprocessor(
    instruction_str=(
        "Given the context, please answer the final question as accurately as possible."
    ),
    # You can also set:
    # target_compression_ratio=0.25,
    # debug=True,
)

# 3) Build query engine with LongLLMLingua

query_engine = index.as_query_engine(
    similarity_top_k=20,              # retrieve generously
    node_postprocessors=[long_llmlingua],  # compress context before LLM
    llm=llm,
)

response = query_engine.query("Explain our refund policy for EU customers.")
print(response)
```

LlamaIndex will:

1. Retrieve `similarity_top_k` nodes.
2. Run LongLLMLingua on them to compress & re‑order for relevance.
3. Pass the compressed nodes to the LLM for final generation.[^7][^2]

### 2.3 Combining LongLLMLingua with reranking

A strong pattern is **CohereRerank → LongLLMLinguaPostprocessor** in the `node_postprocessors` list so you first pick the best passages, then compress them.[^7]

```python
from llama_index.postprocessor.cohere_rerank import CohereRerank

cohere_rerank = CohereRerank(top_n=10, api_key=os.environ["COHERE_API_KEY"])

query_engine = index.as_query_engine(
    similarity_top_k=40,
    node_postprocessors=[cohere_rerank, long_llmlingua],
    llm=llm,
)
```

***

## 3. xRAG – Extreme Context Compression (Research‑Grade)

**xRAG** is a research method that:

- Treats document embeddings as a **retrieval modality**.[^8][^3]
- Trains a **modality bridge** that maps retrieval‑space embeddings into the LM hidden space.
- At inference time, **omits the raw text** and feeds a **single learned token** representing each document into the LM, achieving >3.5× FLOP reduction while matching uncompressed RAG performance.[^9][^3]

There is not yet a stable off‑the‑shelf Python library for xRAG; integration is research‑grade. A high‑level skeleton of how you would integrate it once code is released:

```python
# PSEUDOCODE – not production-ready

# 1) Precompute & store document embeddings with your retriever (e.g., BGE, NV-Embed)
# docs: list[str]
# doc_embs: np.ndarray[num_docs, dim]

# 2) Train modality_bridge: R^dim_retrieval -> R^hidden_lm
# bridge: RetrievalEmbedding -> LMHiddenToken

# 3) At inference, retrieve top-K documents, get their embeddings
retrieved_doc_ids = retriever.search(query, top_k=20)
retrieved_embs = doc_embs[retrieved_doc_ids]

# 4) Map embeddings to LM token representations
lm_tokens = bridge(retrieved_embs)      # shape [k, hidden_dim]

# 5) Call LM with a sequence where these synthetic tokens replace full text
#    e.g., custom LM API that accepts extra key-value (KV) states or special tokens

response = lm.generate_with_extra_tokens(
    prompt_tokens=query_tokens,
    extra_retrieval_tokens=lm_tokens,
)
```

For now, xRAG should be treated as **future work / experimental**: study the paper and reference implementation once available, then wrap it as a custom retriever‑to‑LM adapter.[^3][^10][^8]

***

## 4. ACC‑RAG – Adaptive Context Compression (Research‑Grade Pattern)

ACC‑RAG (Adaptive Context Compression for RAG) is a framework that:

- Builds **hierarchical (multi‑granular) embeddings** of text (sentences, paragraphs, sections).[^11][^4]
- Uses a **context selector** to decide how much to compress for each query.
- Achieves up to **4× faster inference** than standard RAG while matching or improving accuracy.[^12][^4][^11]

There is no production library yet; but you can approximate the approach by combining **hierarchical chunking + dynamic truncation**.

### 4.1 Approximate ACC‑RAG in practice (pseudo‑code)

```python
from dataclasses import dataclass
from typing import List

@dataclass
class Chunk:
    text: str
    level: int        # 0 = coarse (section), 1 = paragraph, 2 = sentence
    score: float = 0  # query-specific score


def build_hierarchical_chunks(doc_text: str) -> List[Chunk]:
    """Offline step: create multi-granularity chunks for ACC-style RAG."""
    # Implement using your existing chunkers (section -> paragraph -> sentence)
    ...


def score_chunks(query: str, chunks: List[Chunk]) -> List[Chunk]:
    """Score chunks via embedding similarity or a lightweight reranker."""
    ...


def select_chunks(query: str, chunks: List[Chunk], token_budget: int) -> List[Chunk]:
    """Adaptive selection akin to ACC-RAG's context selector."""
    # 1) Estimate complexity of query (e.g., length, #entities, etc.)
    # 2) For simple queries, keep only fine-grained top-level chunks
    # 3) For complex queries, also include coarser context
    ...

# Runtime RAG step
all_chunks = build_hierarchical_chunks(document_text)
scored = score_chunks(user_query, all_chunks)
selected = select_chunks(user_query, scored, token_budget=2048)

context = "\n\n".join(ch.text for ch in selected)

prompt = f"Context:\n{context}\n\nQuestion: {user_query}"
```

To get closer to ACC‑RAG, follow the algorithmic details in the paper (hierarchical compressor, selector network) and implement them as a standalone Python module in your stack.[^4][^11]

***

## 5. Practical Prompting & Grounding Templates for RAG

Beyond compression tools, **prompt design** is crucial for grounded RAG.

### 5.1 Minimal grounded QA template with citations

```text
System:
You are a helpful assistant. Answer ONLY using the provided context. If the context
is insufficient, say "I don't know based on the provided documents." When you use a
piece of information, cite it as [Doc<ID>].

User:
Context:
[Doc1] <snippet or summary of doc 1>
[Doc2] <snippet or summary of doc 2>
...

Question: <user question>

Requirements:
- Provide a concise answer first.
- Then, optionally provide a short explanation with citations (e.g., [Doc1], [Doc2]).
- NEVER invent citations that are not in the context.
```

### 5.2 Combining reranking + compression

A production‑grade pattern for long‑context RAG:

1. **Retrieve generously** (top‑K = 40–100 passes) from your vector DB.
2. Apply **reranker** (e.g., Cohere Rerank, BGE‑Reranker, RankGPT) to keep only the top 10–20 highest‑relevance passages.
3. Run **LongLLMLingua / LLMLingua** on these passages to compress them to your **token budget**.
4. Feed compressed context into your LLM with a **grounded QA prompt** (as above).

```python
# Pseudocode skeleton
retrieved = retriever.invoke(user_query)          # Step 1
reranked = cohere_reranker.compress_documents(user_query, retrieved)  # Step 2

joined = "\n\n".join(doc.page_content for doc in reranked)

compressed_prompt = compressor.compress_prompt(
    base_prompt_template.format(context=joined, question=user_query),
    rate=0.25,
)

llm_response = llm_chat(compressed_prompt)
```

This pattern is exactly what LLMLingua + LongLLMLingua experiments show to **mitigate lost‑in‑the‑middle** while cutting cost by 4–20×.[^2][^1]

***

## 6. How to Use This Prompting & Compression Cheat‑Sheet

- Use **LLMLingua / LongLLMLingua** when token costs or context limits are binding, especially on GPT‑4‑class models or long RAG sessions.[^1][^2][^5]
- Treat **xRAG** and **ACC‑RAG** as **research directions**; read the papers and consider them when you are building very high‑scale or latency‑sensitive systems and can afford custom research engineering.[^11][^3][^4]
- Always pair compression with **good prompts and reranking** to avoid compressing irrelevant content.

Keep this Markdown file together with your other RAG component cheat‑sheets so you and your agents have a consistent reference for integrating prompt compression and grounding into your RAC/RAG systems.

---

## References

1. [LLMLingua Series | Effectively Deliver Information to LLMs via ...](https://llmlingua.com) - We use meeting transcripts from the MeetingBank dataset as an example to demonstrate the capabilitie...

2. [LongLLMLingua Prompt Compression Guide | LlamaIndex](https://www.llamaindex.ai/blog/longllmlingua-bye-bye-to-middle-loss-and-save-on-your-rag-costs-via-prompt-compression-54b559b9ddf7) - LongLLMLingua boosts RAG accuracy via prompt compression. Eliminate middle loss and slash enterprise...

3. [xRAG: Extreme Context Compression for Retrieval ...](https://arxiv.org/abs/2405.13792) - Abstract:This paper introduces xRAG, an innovative context compression method tailored for retrieval...

4. [Dynamic Context Compression for Efficient RAG](https://arxiv.org/abs/2507.22931v2) - Retrieval-augmented generation (RAG) enhances large language models (LLMs) with external knowledge b...

5. [microsoft/LLMLingua - GitHub](https://github.com/microsoft/LLMLingua) - LLMLingua-2, a small-size yet powerful prompt compression method trained via data distillation from ...

6. [LLMLingua/examples/RAGLlamaIndex.ipynb at main - GitHub](https://github.com/microsoft/LLMLingua/blob/main/examples/RAGLlamaIndex.ipynb) - Next, we will demonstrate the use of LongLLMLingua on the PG's essay dataset in LlamaIndex pipeline,...

7. [A Cheat Sheet and Some Recipes For Building Advanced ...](https://www.llamaindex.ai/blog/a-cheat-sheet-and-some-recipes-for-building-advanced-rag-803a9d94c41b) - Master advanced RAG techniques with this comprehensive cheat sheet. Explore retrieval optimization, ...

8. [NeurIPS Poster xRAG: Extreme Context Compression for Retrieval ...](https://neurips.cc/virtual/2024/poster/96497) - This paper introduces xRAG, an innovative context compression method tailored for retrieval-augmente...

9. [xRAG: Extreme Context Compression for Retrieval-augmented...](https://openreview.net/forum?id=6pTlXqrO0p) - This paper presents xRAG, a context compression method for Retrieval-Augmented Generation (RAG). The...

10. [[PDF] xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token | Semantic Scholar](https://www.semanticscholar.org/paper/xRAG:-Extreme-Context-Compression-for-Generation-Cheng-Wang/38fcc3667a907d6c94267c674aad114aae68441e) - This paper introduces xRAG, an innovative context compression method tailored for retrieval-augmente...

11. [Enhancing RAG Efficiency with Adaptive Context Compression - arXiv](https://arxiv.org/abs/2507.22931) - We propose Adaptive Context Compression for RAG (ACC-RAG), a framework that dynamically adjusts comp...

12. [Dynamic Context Compression for Efficient RAG](https://huggingface.co/papers/2507.22931) - Join the discussion on this paper page

