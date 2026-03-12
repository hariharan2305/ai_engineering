# RAG Component‑Wise Tech Stack Guide (2026)

This guide restructures the earlier RAG report into **components of a RAG system** and, for each component, lists:

- Typical **tools / models / services** ("tech stacks")
- **Providers** and where to get them
- **Why they’re used / when to pick them**
- **How to integrate** them into your project (API or library entry points)

Use it as a catalog: when you decide “I need X (e.g., embeddings, vector DB, reranker)”, jump to that section and pick from the table.

***

## 0. RAG Component Map

At a high level, a production RAG system decomposes into these components:

| Component | Purpose |
|----------|---------|
| Ingestion & Parsing | Turn raw files (PDF, Word, HTML, scans) into structured text + metadata.[^1][^2] |
| Chunking & Indexing | Split text into retrievable units and build indices (vector, sparse, graph).[^3][^4] |
| Embedding Models | Map text (and other modalities) to vectors for semantic search.[^5][^6] |
| Vector DB / Search Backend | Store embeddings and perform fast similarity/hybrid search.[^3][^7] |
| Retrieval Logic | Query translation, hybrid search, multi‑query, rank fusion.[^8][^9] |
| Re‑ranking | Reorder candidates for higher relevance/precision.[^8][^10] |
| Generation / LLMs | Generate grounded answers from retrieved context.[^11][^12] |
| Prompting & Compression | Format context, mitigate lost‑in‑the‑middle, reduce token usage.[^13][^14][^15] |
| Evaluation & Observability | Measure faithfulness, relevance, latency, cost; trace pipelines.[^16][^17][^18][^19] |
| Caching | Reuse previous results to cut cost & latency (semantic caching).[^20][^21][^22] |
| Orchestration & Agents | Wire components into workflows and agents (LangChain/LangGraph, LlamaIndex, DSPy, etc.).[^23][^24][^25] |

The rest of the guide dives into each.

***

## 1. Document Ingestion & Parsing

### 1.1 Tools and providers

| Tool / Service | Provider / Where | Why use it | How to integrate |
|----------------|------------------|------------|-------------------|
| **Unstructured** | Open‑source library by Unstructured.io; GitHub `Unstructured-IO/unstructured`, docs at docs.unstructured.io.[^26][^27] | Robust parsing of PDFs, HTML, Office docs, emails, images into structured elements (text, titles, tables, images). Great general‑purpose parser for RAG ingestion.[^27][^28] | `pip install unstructured`. Use `partition_pdf`, `partition_docx`, etc. Then convert returned elements to your own `Document` objects (LangChain, LlamaIndex) with metadata. |
| **LlamaParse** | LlamaIndex SaaS + Python SDK; product page at llamaindex.ai/llamaparse and PyPI `llama-parse`.[^29][^30][^31] | High‑quality parsing of complex PDFs (multi‑column, tables, figures) into Markdown/JSON; tightly integrated with LlamaIndex and LongLLMLingua.[^30][^31] | `pip install llama-parse`. Configure API key; in LlamaIndex, use `LlamaParseReader` to ingest files directly into indices. For generic Python, call LlamaParse client to get Markdown/JSON, then feed to your pipeline. |
| **Docling** | Open‑source project (`docling-project/docling`), docs at docling‑project.github.io.[^32][^1][^33] | Layout‑aware parsing for PDFs/Office docs into a structured document tree (blocks, tables, figures, equations). Strong for research papers, reports.[^1] | Install from PyPI, run `Document.from_file()` to parse; traverse the document tree and create chunks with rich metadata (e.g., table IDs, figure captions). |
| **Azure AI Document Intelligence** | Azure Cognitive Services; docs on Microsoft Learn.[^2][^34][^35] | Cloud OCR + layout parsing for forms, invoices, tables, long PDFs; integrates with Azure AI Search and Azure OpenAI for RAG pipelines.[^2][^36] | Use REST/SDK (`azure-ai-formrecognizer`) to call `begin_analyze_document` (layout/general). Store extracted text + structure in your index. Combine with Azure AI Search indexer or custom ingestion code. |
| **AWS Textract** | AWS AI service; docs & blogs on aws.amazon.com/textract.[^37][^38][^39] | OCR + key‑value and table extraction optimized for forms and financial/insurance docs; frequently used upstream of Bedrock Knowledge Bases and LangChain RAG flows.[^37][^38] | Use AWS SDK (`boto3.client('textract')`), call `analyze_document` or `analyze_expense`. Convert results into normalized text/table structures and send to OpenSearch / Aurora pgvector / other DB. |
| **Google Document AI** | Google Cloud Document AI; docs at cloud.google.com/document‑ai.[^40][^41] | Layout‑aware parsing with Gemini‑backed layout model; good for multi‑page, multi‑column documents and forms, integrated with Vertex AI Search/RAG.[^40] | Use the Document AI client library to call layout or form processors; convert returned structured JSON into chunks. Often wired via Vertex AI Search ingestion. |

### 1.2 When to pick what

- For **OSS, self‑hosted**: Unstructured or Docling are strong defaults.[^1][^26]
- For **tight cloud integration**: use Azure Document Intelligence (Azure), Textract (AWS), Document AI (GCP) so you can plug into their managed RAG/search services.[^2][^40][^38]
- For **maximum quality on PDFs** in Python: LlamaParse + LlamaIndex is hard to beat for quick, accurate ingestion.[^30][^31]

***

## 2. Chunking & Indexing Logic

Chunking is usually implemented via your **orchestration framework** plus parser metadata.

### 2.1 Common libraries

| Library / Feature | Provider | Why use it | How to integrate |
|-------------------|----------|-----------|-------------------|
| **LangChain text splitters** | LangChain OSS; docs at `docs.langchain.com`.[^42] | Wide variety of splitters (character, recursive, markdown/header‑aware, token‑based) with overlap. Easy defaults for quick RAG.[^7] | `from langchain_text_splitters import RecursiveCharacterTextSplitter` etc. Use on raw text or Unstructured/Docling output, then wrap into `Document` objects with metadata and write into a vector DB. |
| **LlamaIndex node parsers & index types** | LlamaIndex OSS + Cloud.[^24][^1][^31] | Node‑based abstraction that supports hierarchical (parent‑child) chunking, graph/tree indexes, and per‑section metadata.[^1][^43] | Use `SimpleNodeParser`, `HierarchicalNodeParser`, or `SentenceSplitter` when constructing indices. Choose index types (e.g., `VectorStoreIndex`, `TreeIndex`, `KGIndex`) based on retrieval pattern. |
| **Bedrock Knowledge Bases chunking options** | AWS Bedrock KB.[^44][^45][^4] | Managed ingestion with configurable semantic/hierarchical chunking and metadata extraction; removes need to custom‑build chunkers.[^4] | Configure via console/CloudFormation: choose advanced parsing, chunk size, overlap, semantic options. Then query KB via Bedrock RAG APIs. |
| **Azure AI Search indexers** | Azure AI Search.[^2][^35] | Built‑in indexers parse and chunk content from Blob, SharePoint, etc., into search documents with vector fields and text fields.[^35] | Define an index schema (text, vector, metadata fields), configure indexer data source and skillset; chunking is handled via cognitive skills. |

### 2.2 Practical guidance

- For **custom code**, start with a header‑aware or semantic splitter (LangChain Recursive + metadata, or LlamaIndex Sentence/Hierarchical parsers) with ~200–500 token chunks and 20–50 token overlap.[^7][^1]
- For **fully managed**, lean on Bedrock KB or Azure AI Search/Vertex AI Search which already embed chunking options.[^46][^35][^4]

***

## 3. Embedding Models

### 3.1 Popular choices and where to get them

| Model family | Provider / Access | Why it’s used | Integration entry points |
|--------------|------------------|---------------|--------------------------|
| **text‑embedding‑3‑small / large** | OpenAI API (`/v1/embeddings`); docs at platform.openai.com.[^5][^47] | Strong general‑purpose & multilingual performance with good price/performance; widely supported in frameworks.[^5] | Via OpenAI SDK (`openai.Embeddings.create`), or via LangChain/LlamaIndex `OpenAIEmbeddings`. Configure in Bedrock KB or Azure OpenAI equivalent when using managed RAG. |
| **NV‑Embed‑v1 (NV‑Embed series)** | Nvidia; HF hub `nvidia/NV-Embed-v1`, NIM services, Nvidia API.[^6][^48][^49] | Top‑tier MTEB performance and optimized for GPU inference; good for on‑prem high‑throughput retrieval.[^5][^6] | Use `transformers` or `sentence-transformers` in Python from Hugging Face, or deploy via Nvidia NIM/container. Plug into Qdrant/Weaviate/etc. as an embedding function. |
| **BGE (e.g., `bge-large-en-v1.5`, `bge-m3`)** | BAAI via Hugging Face `BAAI/bge-*` and FlagEmbedding repo.[^50][^51] | Very strong open‑source retrieval performance (top of MTEB), with multilingual and reranker variants (BGE‑Reranker).[^5][^50] | `pip install FlagEmbedding`; use `BGEM3FlagModel` or similar to encode queries/docs. Integrate via LangChain `HuggingFaceEmbeddings` or LlamaIndex `HuggingFaceEmbedding`. |
| **E5 / GTE families** | Various research groups, available on HF (`intfloat/e5-*`, `thenlper/gte-*`).[^5][^52] | Strong general‑purpose retrieval & classification; widely used in open‑source RAG.[^52] | As sentence‑transformers models; plug into any vector DB via their Python client. |
| **Gemini embeddings** | Google Gemini API / Vertex AI Embeddings.[^11][^46] | Tight integration with Vertex AI Search, Document AI, and Google Cloud infra; good multilingual support.[^11][^46] | Use Vertex AI SDK `TextEmbeddingModel`; or configure as embedding backend for Vertex AI Search. |
| **Bedrock Titan embeddings / Cohere Embed** | AWS Bedrock & Cohere APIs.[^44][^45] | Used in Bedrock Knowledge Bases; makes it trivial to stand up RAG on AWS without running your own models.[^44] | Configure embedding model in KB; for custom apps, call `bedrock-runtime` or Cohere SDK directly and store vectors in OpenSearch/Qdrant/etc. |

### 3.2 How to choose

- **Cloud‑first, low‑ops**: provider embeddings (OpenAI, Gemini, Bedrock/Cohere) via their APIs; let managed RAG services handle scaling.[^5][^45][^11]
- **Self‑hosted / data‑sovereign**: BGE/E5/GTE or NV‑Embed on your own GPUs; integrate through sentence‑transformers interfaces.[^6][^52]
- **Multilingual heavy**: BGE‑m3 or Gemini/Claude/CoHere embeddings explicitly optimized for multilingual performance.[^12][^5]

***

## 4. Vector Databases & Search Backends

### 4.1 Core options

| DB / Service | Provider / Where | Why it’s used | Integration pattern |
|--------------|------------------|---------------|----------------------|
| **Qdrant** | Open‑source (Rust) + Qdrant Cloud; docs at qdrant.tech.[^3][^53] | HNSW‑based vector DB with strong hybrid search demos (dense + sparse), metadata filtering, and JSON payloads.[^3][^53] | Run via Docker or cloud; use Python/TypeScript clients. In LangChain/LlamaIndex, configure `QdrantVectorStore`/`QdrantVectorStoreIndex`. Hybrid: use Qdrant’s sparse+dense API. |
| **Weaviate** | OSS + Weaviate Cloud.[^7][^54][^55] | Schema‑driven vector DB with hybrid search, late‑interaction support (ColBERT/ColPali integrations), and GraphQL‑like API.[^54][^55] | Deploy Weaviate; define classes with vector+text properties; use REST/gRPC or Python client; in LangChain, `WeaviateVectorStore`. Hybrid via `hybrid` search. |
| **Milvus / Zilliz** | OSS Milvus + Zilliz Cloud.[^56][^57] | High‑scale ANN (HNSW, IVF‑PQ) for billion‑scale collections; widely used in vector‑heavy workloads.[^57] | Use Milvus Python/Go/Java clients; or Zilliz Cloud; integrate via LangChain or custom code. |
| **pgvector** | PostgreSQL extension.[^58][^59] | Simple architecture: relational + vector in one DB; ideal for moderate scale or tighter transactional needs.[^58] | Install pgvector extension; create vector columns; use SQL + embedding function; integrate with any backend that can talk to Postgres. |
| **Pinecone** | Managed vector DB as a service.[^56][^57] | Offloads ops; strong SLA; good for startups that want “batteries‑included” vector search.[^56] | Use Pinecone Python/JS clients; in LangChain/LlamaIndex as `PineconeVectorStore`. |
| **OpenSearch / Elasticsearch kNN** | AWS OpenSearch, Elastic Cloud.[^54][^53] | Combine classic inverted indexing with vector search + aggregations; attractive if you already use Elastic/OpenSearch for logs/search.[^54] | Define vector fields + analyzers; index docs with both text and vectors; query with hybrid (BM25+vector) via kNN APIs. |
| **Cloud RAG search services** | Azure AI Search, Vertex AI Search, Bedrock KB vector stores.[^44][^45][^46] | Fully managed search + vector + RAG orchestration; minimal ops; enterprise IAM/governance.[^45][^35][^46] | Configure index, data source, and enrichment/embedding in provider console; query via REST/SDK; often you don’t see the vector DB directly. |

### 4.2 Practical choice heuristics

- **Greenfield, AWS**: Bedrock KB + OpenSearch Serverless or Aurora pgvector.[^44][^60][^45]
- **Greenfield, Azure**: Azure AI Search + Azure Blob/Graph + Azure OpenAI.[^35][^2]
- **Greenfield, GCP**: Vertex AI Search + Document AI + Gemini.[^40][^46]
- **Self‑hosted:** Qdrant or Weaviate for most RAG apps; Milvus for extreme scale; pgvector if you want minimal infra.[^3][^58][^7]

***

## 5. Retrieval Strategies & Libraries

Retrieval logic often lives in your framework (LangChain, LlamaIndex, DSPy) or cloud service (Vertex AI Search, Bedrock KB, Azure AI Search).

### 5.1 Retrieval‑layer tools

| Tool / Concept | Provider | Why use it | How to integrate |
|----------------|----------|-----------|-------------------|
| **LangChain retrievers (base, MultiQuery, Contextual)** | LangChain.[^42][^61] | Implement HyDE, multi‑query/RAG‑Fusion, contextual retrieval on top of any `VectorStore`.[^9][^61] | Use `MultiQueryRetriever`, `ContextualCompressionRetriever`, etc. Wrap your vector store, plug into chains/agents. |
| **LlamaIndex retrievers (VectorIndexRetriever, KGIndexRetriever, GraphRAG)** | LlamaIndex.[^24][^1][^62] | Provide pluggable retrievers over multiple index types (vector, list, tree, KG), including GraphRAG‑style hierarchical retrieval.[^62][^43] | Build appropriate index; call `.as_query_engine()` with custom retriever settings (top‑k, similarity, filters). |
| **Anthropic Contextual Retrieval recipe** | Anthropic blog + cookbook.[^8] | Hybrid contextual embeddings + BM25 + reranking; reduces retrieval failure ~67% over embeddings‑only baselines.[^8] | Use Anthropic’s examples (Python) to build contextualized chunks and BM25 index; integrate with your LLM via Claude API. |
| **Vertex AI Search** | Google Cloud.[^46] | End‑to‑end retrieval with vector + keyword search, ranking, facets and tight integration with Gemini.[^46][^11] | Configure data store/index; query via `SearchServiceClient`. For RAG, feed top results into Gemini prompt. |
| **Bedrock KB query APIs** | AWS.[^44][^45] | Abstract over embeddings + vector DB + hybrid search; returns ranked passages for RAG.[^44][^4] | Use Bedrock KB retrieve API; results are ready to stuff into the LLM prompt with metadata. |

### 5.2 Implementation notes

- Start with **hybrid retrieval** (BM25 + dense) whenever your queries involve IDs, codes, or long‑tail phrases.[^8][^54][^3]
- Add **multi‑query / query rewrite** for complex analytical queries via LangChain or LlamaIndex wrappers.[^63][^61]

***

## 6. Re‑ranking Components

### 6.1 Rerankers & where to get them

| Model / Service | Provider | Why use it | Integration pattern |
|-----------------|----------|-----------|----------------------|
| **Cohere Rerank 3 / 3.5** | Cohere via Cohere API; integrated in Pinecone and OpenSearch examples.[^64][^65][^66][^67] | Production‑grade cross‑encoder for reranking top‑k documents; strong zero‑shot ranking across domains.[^66] | Call Cohere `/rerank` with query + candidate texts; re‑order results. In OpenSearch, use Rerank plugin; in Pinecone, use model connector. |
| **BGE‑Reranker (base/large)** | BAAI via HF (`BAAI/bge-reranker-*`).[^50][^68][^51] | Open‑source rerankers tuned to pair with BGE embeddings; very strong on MTEB rerank tasks.[^50] | Use `FlagEmbedding` reranker or `transformers` to score query–doc pairs; integrate as a second stage in your retrieval pipeline (LangChain `CrossEncoderReranker`). |
| **RankGPT / LLM‑based rerankers** | Implemented in LlamaIndex and research frameworks.[^69][^70] | Use a powerful LLM itself to score and order passages; often best quality, higher cost.[^69] | In LlamaIndex, use `RankGPT` node post‑processor; more generally, build a prompt that asks the LLM to score each candidate and sort them. |
| **Qwen / Qwen3 rerankers** | Alibaba’s Qwen3 Embedding & reranking models.[^71] | Provide strong open‑source reranking with joint embedding/reranking objectives.[^71] | Use HF models or vendor APIs as with BGE reranker; drop into LangChain’s HF reranker wrappers. |

### 6.2 When to add reranking

- If you see **off‑topic context** in prompts despite good retrieval, add a reranker and reduce the number of final chunks.
- For **very large corpora**, always use reranking (Cohere, BGE‑Reranker, or LLM‑based) after an approximate ANN search.[^10][^8]

***

## 7. Generation Models (LLMs)

### 7.1 Providers and typical roles

| Provider | Example models | Why used in RAG | Integration |
|----------|----------------|-----------------|-------------|
| **OpenAI** | GPT‑4.1, o‑series, GPT‑4 Turbo | Strong general reasoning, tool use, wide ecosystem support.[^5] | OpenAI API; plus Azure OpenAI for enterprise. Plugs into all major RAG frameworks. |
| **Anthropic** | Claude 3.5 Sonnet/Opus | Very long context (200k+), strong safety, contextual retrieval guidance.[^12][^72][^8] | Claude API; integrated in many RAG recipes and agent frameworks. |
| **Google** | Gemini 1.5 Pro/Flash | Up to 1M tokens context in preview; tight integration with Vertex AI Search & Document AI.[^11][^46][^73] | Gemini API (direct) or Vertex AI; used in long‑context RAG or multi‑document QA. |
| **AWS Bedrock** | Claude, Llama, Mistral, Titan | Single API across multiple models; deep integration with KBs and Agents for Bedrock.[^44][^45] | Bedrock Runtime API; chain via LangChain’s Bedrock wrappers or native Agents for Bedrock. |
| **Azure OpenAI** | GPT‑4.x, GPT‑4o | Enterprise‑grade GPT with Azure identity, networking, and data governance.[^74][^75] | Azure OpenAI SDK; used together with Azure AI Search & Document Intelligence. |
| **Open‑source (Llama, Mistral, Qwen, Phi)** | Various HF models | Self‑hosted, customizable, cost‑controlled; often used behind vLLM or TGI.[^5][^6] | Deploy via Hugging Face TGI, vLLM, Nvidia NIM, or cloud marketplaces; integrate with LangChain/LlamaIndex as `ChatOpenAI`‑compatible wrappers or custom clients. |

### 7.2 Integration pattern

1. Pick model & provider.
2. Use provider SDK or framework wrapper (`ChatOpenAI`, `ChatAnthropic`, `ChatVertexAI`, `ChatBedrock`).
3. Ensure you pass **grounding context** (retrieved docs) and ask for **citations** explicitly in the prompt.

***

## 8. Prompting, Context Compression & Grounding

### 8.1 Prompt‑time tools

| Tool / Method | Provider | Why use it | Integration |
|---------------|----------|------------|-------------|
| **LLMLingua** | Microsoft research; open‑source.[^14][^76] | Coarse‑to‑fine token‑level prompt compression achieving up to 20× compression with small performance loss.[^14] | Use LLMLingua model to preprocess prompts before sending to the main LLM; LlamaIndex has built‑in integration for RAG. |
| **LongLLMLingua** | LlamaIndex integration.[^31][^15] | Specialization of LLMLingua for long‑context + RAG, mitigating “lost in the middle” and improving accuracy while cutting tokens ~4×.[^15] | Enable LongLLMLingua in LlamaIndex query engine configuration; use as a context compressor / rewriter on retrieved docs. |
| **xRAG** | Research prototype (NeurIPS poster).[^77] | Extreme context compression by fusing document embeddings into LM representation space via modality bridge; >3.5× FLOP reduction while matching uncompressed RAG.[^77] | Research‑grade; follow paper’s code (when released) to train a modality bridge; integrate by feeding embeddings instead of full text. |
| **ACC‑RAG (Adaptive Context Compression)** | Research paper (TACL / EMNLP‑style).[^78][^79] | Adaptive compression using hierarchical embeddings + selector to decide how much context to feed per query; ~4× faster inference with comparable or better accuracy.[^78] | Use as reference for implementing your own adaptive compression (hierarchical embeddings + dynamic truncation). |

### 8.2 Practical tips

- Use a **clear prompt template**: instructions, question, context (with document IDs/metadata), and explicit requirement for citing sources.
- Combine **reranking** + **prompt compression** to avoid lost‑in‑the‑middle: send only the most relevant, compressed snippets at the top of the prompt.[^13][^15]

***

## 9. Evaluation & Observability

### 9.1 RAG evaluation frameworks

| Tool | Provider | Focus | Integration |
|------|----------|-------|-------------|
| **RAGAS** | OSS; paper and GitHub.[^16][^80][^81] | Automated RAG evaluation (faithfulness, answer relevance, context precision/recall) using LLM‑as‑judge.[^16][^80] | `pip install ragas`; define dataset of (question, context, answer); run metrics. Integrates with LangChain/LlamaIndex examples. |
| **TruLens** | TruEra; OSS.[^17] | RAG triad: context relevance, groundedness, answer relevance; rich dashboards and tracing.[^17][^82] | Instrument your chains/agents; mark retriever and generator steps; compute triad metrics and visualize. |
| **DeepEval** | Confident AI / community project.[^83] | Emphasizes red‑teaming, safety, hallucinations plus RAG metrics.[^83] | Install library; configure metrics and LLM‑as‑judge; run against offline datasets and in CI. |
| **Arize Phoenix** | Arize AI; OSS library + UI.[^18][^84][^85][^86] | End‑to‑end LLM tracing + evaluation; supports RAG relevance, hallucination metrics, vector space visualization.[^84][^85] | Add Phoenix tracing to your app; send traces (inputs, retrieved docs, outputs); configure evals in UI. Works with LangChain/LlamaIndex/DSPy. |

### 9.2 LLM observability platforms

| Platform | Provider | Why use it | Integration |
|----------|----------|-----------|-------------|
| **LangSmith** | LangChain.[^19][^87][^88] | “Datadog for LLMs”: tracing, datasets, evals, prompt playground; native with LangChain/LangGraph but framework‑agnostic.[^19][^87] | Create LangSmith project; use LangChain’s `langsmith` client or SDK to log traces; inspect chains, agents, and RAG behavior in UI. |
| **Langfuse** | OSS project + cloud; acquired by ClickHouse.[^89][^90][^91][^92] | Open‑source LLM engineering/observability: traces, evals, prompt management, metrics; strong AWS/Bedrock integration.[^89][^90][^93] | `pip install langfuse`; instrument your app with SDK; plug into LangChain, LlamaIndex, DSPy; run self‑hosted or use Langfuse Cloud. |

### 9.3 How to add evals to your project

1. **Log everything**: user query, retrieved docs (IDs, scores), final answer, model, prompt, and latency.
2. Use RAGAS or TruLens offline on a labelled dataset to compare retrieval and generation variants.[^16][^17]
3. Add **online evals** via LangSmith, Phoenix, or Langfuse: sample live traffic, run LLM‑as‑judge metrics, and trend quality over time.[^19][^91][^84]

***

## 10. Caching & Performance

### 10.1 Semantic caching tools

| Tool | Provider | Why use it | Integration |
|------|----------|-----------|-------------|
| **GPTCache** | Zilliz; OSS library.[^22][^94][^95] | Semantic cache for LLM queries: caches responses keyed by embeddings, reducing API calls and latency.[^22][^96] | `pip install gptcache`; configure embedding backend and storage (SQLite, vector DB); wrap your LLM client so it checks cache before calling the model. Fully integrated with LangChain & LlamaIndex. |

### 10.2 Patterns

- Combine **exact‑match** and **semantic** caching (GPTCache supports both) to maximize hits.[^96][^22]
- Carefully manage TTL and cache invalidation when your underlying knowledge base changes.

***

## 11. Orchestration, Agents, and Workflows

### 11.1 Orchestration frameworks

| Framework | Provider | Why use it | Integration |
|-----------|----------|-----------|-------------|
| **LangChain** | OSS; langchain.com.[^42][^88] | Rich ecosystem of integrations (LLMs, vector DBs, tools); strong support for chains and agents.[^42][^97] | `pip install langchain`. Define `Document` loaders, `VectorStore`, `Retriever`, and `LLMChain` or `Runnable` graphs. |
| **LangGraph** | By LangChain; docs at docs.langchain.com/langgraph.[^23][^97][^98] | Graph‑based orchestration for complex, stateful, cyclic workflows (agentic RAG, multi‑agent systems).[^23][^99] | Build nodes (LLMs, tools, retrievers), connect in a directed graph; deploy as long‑running agent; integrate with LangSmith/Langfuse. |
| **LlamaIndex** | OSS + cloud.[^24][^43][^31] | Index‑first abstraction for RAG; advanced indices (vector, tree, graph, KG), LongLLMLingua & LlamaParse integration.[^24][^31] | Define documents → `ServiceContext` & `StorageContext` → build indices → use `query_engine` or `router_query_engine`. |
| **Haystack** | deepset; OSS.[^100][^101] | Production search pipelines with retrievers/readers/rerankers; strong Elastic/OpenSearch support.[^100] | Define pipelines in YAML or Python; use `DocumentStore` + `Retriever` + `Reader` nodes. |
| **DSPy** | Stanford; OSS.[^25][^102][^103] | Declarative, eval‑driven programming of RAG systems; automatically optimizes prompts/fine‑tuning.[^25][^103] | `pip install dspy-ai`; define `dspy.Module` graph with retrievers & generators; provide eval set; run DSPy optimizer to tune prompts. |

### 11.2 Cloud‑native orchestration

- **AWS Agents for Bedrock**: define agents that call Bedrock KBs, tools, and models through a configuration file; AWS manages orchestration and grounding.[^45][^44]
- **Azure AI Studio “Flows”**: low‑code orchestration of Document Intelligence, AI Search, and Azure OpenAI.[^74][^35]
- **Vertex AI Agents & Workflows**: combine Document AI, Vertex AI Search, and Gemini into multi‑step flows.[^46][^40]

***

## 12. Putting It Together: How to Use This Catalog

To implement or swap a **component** in your RAC/RAG system:

1. **Identify the component** (e.g., embeddings, parser, vector DB, reranker).
2. **Pick a tech stack** from the relevant table based on your constraints:
   - Cloud vs. self‑hosted
   - Scale and latency
   - Data governance and compliance
   - Language / modality needs
3. **Follow the “How to integrate” column** to:
   - Install the library or configure the cloud service.
   - Plug it into your orchestration framework (LangChain/LangGraph, LlamaIndex, DSPy) as the relevant abstraction (Loader → Splitter → VectorStore → Retriever → Reranker → LLM).
   - Add evaluation & observability (RAGAS/TruLens + LangSmith/Langfuse/Phoenix) so you can measure the impact.[^91][^17][^84][^19][^16]
4. **Iterate with A/B tests and evals** whenever you change a component, especially embeddings, retrieval strategies, and LLMs.

Used this way, the guide becomes a **living component directory**: when you say “I want to try a different reranker” or “I need cloud‑native parsing on Azure,” you can directly jump to the corresponding section, choose an option, and wire it into your RAC/RAG pipeline.

---

## References

1. [Docling Technical Report - arXiv.org](https://arxiv.org/html/2408.09869v3) - This technical report introduces Docling, an easy to use, self-contained, MIT-licensed open-source p...

2. [Azure AI Document Intelligence: Parsing PDF text & table data - Elastic](https://www.elastic.co/search-labs/blog/azure-ai-document-intelligence-parse-pdf-text-tables) - Azure AI Document Intelligence is a powerful tool for extracting structured data from PDFs. It can b...

3. [Semantic Chunking for RAG: Better Context, Better Results](https://www.multimodal.dev/post/semantic-chunking-for-rag) - Explore how semantic chunking enhances RAG systems by improving context, precision, and performance ...

4. [Amazon Bedrock Knowledge Bases now supports advanced ...](https://aws.amazon.com/blogs/machine-learning/amazon-bedrock-knowledge-bases-now-supports-advanced-parsing-chunking-and-query-reformulation-giving-greater-control-of-accuracy-in-rag-based-applications/) - Parsing documents is important for RAG applications because it enables the system to understand the ...

5. [NVIDIA Text Embedding Model Tops MTEB Leaderboard](https://developer.nvidia.com/blog/nvidia-text-embedding-model-tops-mteb-leaderboard/) - The NV-Embed model from NVIDIA achieved a score of 69.32 on the Massive Text Embedding Benchmark (MT...

6. [NV-Embed: Improved Techniques for Training LLMs as Generalist ...](https://arxiv.org/html/2405.17428v3) - In this work, we introduce NV-Embed, a generalist embedding model that significantly enhances the pe...

7. [Chunking Strategies to Improve LLM RAG Pipeline Performance](https://weaviate.io/blog/chunking-strategies-for-rag) - Learn how chunking strategies improve LLM RAG pipelines, retrieval quality, and agent memory perform...

8. [RAG Framework Comparison Guide 2025 - RAG Systems](https://artificial-intelligence-wiki.com/ai-development/rag-systems/rag-framework-comparison/) - Compare top RAG frameworks including LangChain, LlamaIndex, Haystack, and more. Performance benchmar...

9. [HyDE: Hypothetical Document Embeddings - Emergent Mind](https://www.emergentmind.com/topics/hypothetical-document-embeddings-hyde) - HyDE leverages LLM-generated synthetic answers to enhance semantic retrieval in both dense and spars...

10. [The top 6 Vector Databases to use for AI applications in 2025](https://appwrite.io/blog/post/top-6-vector-databases-2025) - 6 popular Vector Databases you should consider in 2025 · 1. MongoDB Atlas · 2. Chroma · 3. Pinecone ...

11. [Our next-generation model: Gemini 1.5 - Google Blog](https://blog.google/innovation-and-ai/products/google-gemini-next-generation-model-february-2024/) - Gemini 1.5 Pro comes with a standard 128,000 token context window. But starting today, a limited gro...

12. [How Claude Processes Long Documents (100K+ Tokens)](https://claude-ai.chat/guides/how-claude-processes-long-documents/) - Claude 3.5 (“Sonnet”) is Anthropic’s latest AI model known for its massive context window – up to 20...

13. [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) - We analyze the performance of language models on two tasks that require identifying relevant informa...

14. [LLMLingua: Compressing Prompts for Accelerated Inference of ...](https://arxiv.org/abs/2310.05736) - This paper presents LLMLingua, a coarse-to-fine prompt compression method that involves a budget con...

15. [LongLLMLingua Prompt Compression Guide | LlamaIndex](https://www.llamaindex.ai/blog/longllmlingua-bye-bye-to-middle-loss-and-save-on-your-rag-costs-via-prompt-compression-54b559b9ddf7) - LongLLMLingua boosts RAG accuracy via prompt compression. Eliminate middle loss and slash enterprise...

16. [Ragas: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) - We introduce RAGAs (Retrieval Augmented Generation Assessment), a framework for reference-free evalu...

17. [RAG Triad - TruLens](https://www.trulens.org/getting_started/core_concepts/rag_triad/) - The RAG triad is made up of 3 evaluations: context relevance, groundedness and answer relevance. Sat...

18. [Arize AI Debuts Phoenix, the First Open Source Library for ...](https://www.prnewswire.com/news-releases/arize-ai-debuts-phoenix-the-first-open-source-library-for-evaluating-large-language-models-301808045.html) - /PRNewswire/ -- Arize AI, a market leader in machine learning observability, debuted deeper support ...

19. [AI Agent & LLM Observability Platform - LangSmith - LangChain](https://www.langchain.com/langsmith/observability) - Complete AI agent and LLM observability platform with tracing and real-time monitoring. Debug agents...

20. [Improving RAG Applications with Semantic Caching and RAGAS](https://2024.allthingsopen.org/improving-rag-applications-with-semantic-caching-and-ragas) - Semantic caching is a way of boosting RAG performance by serving relevant, cached LLM responses, thu...

21. [Semantic Caching in RAG: Speed Without Sacrificing Relevance](https://www.linkedin.com/pulse/semantic-caching-rag-speed-without-sacrificing-joaquin-marques-6pkhc) - RAG systems face an expensive computational reality: every user query triggers vector database searc...

22. [GPTCache : A Library for Creating Semantic Cache for LLM Queries](https://gptcache.readthedocs.io)

23. [Build a custom RAG agent with LangGraph - Docs by LangChain](https://docs.langchain.com/oss/python/langgraph/agentic-rag) - Overview. In this tutorial we will build a retrieval agent using LangGraph. LangChain offers built-i...

24. [RAG Frameworks: LangChain vs LangGraph vs LlamaIndex](https://research.aimultiple.com/rag-frameworks/) - We benchmarked 5 RAG frameworks: LangChain, LangGraph, LlamaIndex, Haystack, and DSPy, by building t...

25. [LLMOps with DSPy: Build RAG Systems Using Declarative ...](https://pyimagesearch.com/2024/09/09/llmops-with-dspy-build-rag-systems-using-declarative-programming/) - Discover how to build Retrieval Augmented Generation (RAG) systems using declarative programming wit...

26. [GitHub - Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured) - The unstructured library provides open-source components for ingesting and pre-processing images and...

27. [Extract images and tables from documents - Unstructured](https://docs.unstructured.io/open-source/how-to/extract-image-block-types)

28. [Unstructured Leads in Document Parsing Quality: Benchmarks Tell ...](https://unstructured.io/blog/unstructured-leads-in-document-parsing-quality-benchmarks-tell-the-full-story) - The charts below show how different Unstructured pipelines each using different VLMs and enrichment ...

29. [llama-parse - PyPI](https://pypi.org/project/llama-parse/) - LlamaParse is a GenAI-native document parser that can parse complex document data for any downstream...

30. [AI Document Parsing Software: AI-Ready Data at Scale - LlamaIndex](https://www.llamaindex.ai/llamaparse) - AI-powered document processing for complex PDFs, spreadsheets, images, and more. Parse tables, chart...

31. [LongLLMLingua: Bye-bye to Middle Loss and Save on Your RAG Costs via Prompt Compression — LlamaIndex - Build Knowledge Assistants over your Enterprise Data](https://www.llamaindex.ai/blog/longllmlingua-bye-bye-to-middle-loss-and-save-on-your-rag-costs-via-prompt-compression-54b559b9ddf7?gi=fa3411984a90) - LongLLMLingua boosts RAG accuracy via prompt compression. Eliminate middle loss and slash enterprise...

32. [Documentation - Docling - GitHub Pages](https://docling-project.github.io/docling/)

33. [Docling - Open Source Document Processing for AI](https://www.docling.ai) - Docling converts messy documents into structured data and simplifies downstream document and AI proc...

34. [New And Updated Prebuilt...](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/announcing-the-general-availability-of-document-intelligence-v4-0-api/4357988) - The Document Intelligence v4.0 API is now generally available! This latest version of Document Intel...

35. [Complex Data Extraction using Document Intelligence and RAG](https://techcommunity.microsoft.com/blog/azurearchitectureblog/complex-data-extraction-using-document-intelligence-and-rag/4267718) - This guide will show an approach to building a solution for complex entity extraction using Document...

36. [Enhancing Document Extraction with Azure AI Document ...](https://chinnychukwudozie.com/2024/07/10/enhancing-document-extraction-with-azure-ai-document-intelligence-and-langchain-for-rag-workflows/) - Overview. The broadening of conventional data engineering pipelines and applications to include docu...

37. [Better RAG accuracy and consistency with Amazon Textract](https://community.aws/content/2njwVmseGl0sxomMvrq65PzHo9x/better-rag-accuracy-and-consistency-with-amazon-textract) - Crafting a Retrieval-Augmented Generation (RAG) pipeline may seem straightforward, but optimizing it...

38. [Intelligent document processing with Amazon Textract, Amazon ...](https://aws.amazon.com/blogs/machine-learning/intelligent-document-processing-with-amazon-textract-amazon-bedrock-and-langchain/) - This post takes you through the synergy of IDP and generative AI, unveiling how they represent the n...

39. [LLM ➕ OCR = 🔥 Intelligent Document Processing (IDP) with Amazon Textract, AWS Bedrock, & LangChain](https://www.youtube.com/watch?v=bKfWdW6BrfU) - In this video we are going to explore , how we can enhance an Intelligent Document Processing (IDP) ...

40. [Process documents with Gemini layout parser | Document AI](https://docs.cloud.google.com/document-ai/docs/layout-parse-chunk) - Document OCR: It can parse text and layout elements like heading, header, footer, table structure an...

41. [Layout parser Quickstart | Document AI - Google Cloud Documentation](https://docs.cloud.google.com/document-ai/docs/layout-parse-quickstart) - Use layout parser to extract elements from a document, such as text, tables, and lists. To follow st...

42. [LangChain, LlamaIndex, and Haystack are frameworks ... - Milvus](https://milvus.io/ai-quick-reference/what-are-the-differences-between-langchain-and-other-llm-frameworks-like-llamaindex-or-haystack) - LangChain, LlamaIndex, and Haystack are frameworks designed to help developers build applications wi...

43. [LlamaIndex 2024 Year In Review: Top Releases](https://www.llamaindex.ai/blog/the-year-in-llamaindex-2024) - March: LlamaParse, the world's best parser of complex document formats, is part of LlamaParse but la...

44. [Add flexibility to your RAG applications in Amazon Bedrock](https://community.aws/content/2gSzqTkFq25coY1upSDvpcVowV6/add-flexibility-to-your-rag-applications-in-amazon-bedrock?lang=en) - Use the right configuration options for your Knowledge Base

45. [Foundation Models for RAG - Amazon Bedrock Knowledge Bases](https://aws.amazon.com/bedrock/knowledge-bases/) - With Amazon Bedrock Knowledge Bases, you can give foundation models and agents contextual informatio...

46. [Long context | Generative AI on Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/long-context) - Gemini 1.5 Flash accepts up to 9.5 hours of audio in a single request and Gemini 1.5 Pro can accept ...

47. [text-embedding-3-large Model | OpenAI API](https://developers.openai.com/api/docs/models/text-embedding-3-large) - text-embedding-3-large is our most capable embedding model for both english and non-english tasks. E...

48. [nvidia/NV-Embed-v1 - Hugging Face](https://huggingface.co/nvidia/NV-Embed-v1) - We introduce NV-Embed, a generalist embedding model that ranks No. 1 on the Massive Text Embedding B...

49. [NV-Embed: NVIDIA's Groundbreaking Embedding Model Dominates ...](https://www.marktechpost.com/2024/05/28/nv-embed-nvidias-groundbreaking-embedding-model-dominates-mteb-benchmarks/) - NV-Embed: NVIDIA’s Groundbreaking Embedding Model Dominates MTEB Benchmarks

50. [BAAI/bge-reranker-base - Hugging Face](https://huggingface.co/BAAI/bge-reranker-base) - 3/18/2024: Release new rerankers, built upon powerful M3 and LLM (GEMMA and MiniCPM, not so large ac...

51. [FlagOpen/FlagEmbedding: Retrieval and Retrieval-augmented LLMs](https://github.com/FlagOpen/FlagEmbedding) - New reranker model: release cross-encoder models BAAI/bge-reranker-base and BAAI/bge-reranker-large ...

52. [Top embedding models on the MTEB leaderboard - Modal](https://modal.com/blog/mteb-leaderboard-article) - The Hugging Face MTEB leaderboard has become a standard way to compare embedding models. But the ran...

53. [Demo: Implementing a Hybrid Search System - Qdrant](https://qdrant.tech/course/essentials/day-3/hybrid-search-demo/) - Step-by-step demo on implementing hybrid search using Qdrant's Universal Query API. Explore dense vs...

54. [Hybrid Search Explained | Weaviate](https://weaviate.io/blog/hybrid-search-explained) - Hybrid search works by combining the results of sparse vector search (e.g., BM25) and dense vector s...

55. [An Overview of Late Interaction Retrieval Models: ColBERT, ColPali ...](https://weaviate.io/blog/late-interaction-overview) - Late interaction allow for semantically rich interactions that enable a precise retrieval process ac...

56. [Pinecone vs Weaviate vs Qdrant vs FAISS vs Milvus vs ...](https://liquidmetal.ai/casesAndBlogs/vector-comparison/) - Compare leading vector databases like Pinecone, Weaviate, and Qdrant to find the best solution for y...

57. [Best Vector Databases in 2026: A Complete Comparison Guide](https://www.firecrawl.dev/blog/best-vector-databases) - While Pinecone and Milvus focus on pure vector search, Weaviate does one thing better than any other...

58. [Chunking Strategies for RAG: Best Practices and Key Methods](https://unstructured.io/blog/chunking-for-rag-best-practices) - Chunking strategies for RAG directly affect retrieval precision and LLM response quality. Compare fi...

59. [Top 15 Vector Databases that You Must Try in 2025](https://www.geeksforgeeks.org/dbms/top-vector-databases/) - Top 15 Vector Databases that You Must Try in 2025 · 1. Chroma · 2. Pinecone · 3. Deep Lake · 4. Vesp...

60. [Dive deep into vector data stores using Amazon Bedrock ...](https://aws.amazon.com/blogs/machine-learning/dive-deep-into-vector-data-stores-using-amazon-bedrock-knowledge-bases/) - This post dives deep into Amazon Bedrock Knowledge Bases, which helps with the storage and retrieval...

61. [5 Proven Query Translation Techniques To Boost Your RAG ...](https://towardsdatascience.com/5-proven-query-translation-techniques-to-boost-your-rag-performance-47db12efe971/) - The LLM now has more granular information to solve a complex problem. # 4.1 Decomposition prompting ...

62. [Welcome - GraphRAG](https://microsoft.github.io/graphrag/) - The GraphRAG process involves extracting a knowledge graph out of raw text, building a community hie...

63. [Advanced RAG Optimization: Boosting Answer Quality on Complex ...](https://blog.epsilla.com/advanced-rag-optimization-boosting-answer-quality-on-complex-questions-through-query-decomposition-e9d836eaf0d5) - Query decomposition is a sophisticated technique in natural language processing and information retr...

64. [Cohere Rerank 3.5 - Oracle Help Center](https://docs.oracle.com/en-us/iaas/Content/generative-ai/cohere-rerank-3-5.htm) - The cohere.rerank.v3-5 model takes in a query and a list of texts and produces an ordered array with...

65. [cohere-rerank-3.5 - Pinecone Docs](https://docs.pinecone.io/models/cohere-rerank-3.5)

66. [Master Reranking with Cohere Models](https://docs.cohere.com/docs/reranking-with-cohere) - This page contains a tutorial on using Cohere's ReRank models.

67. [docs.opensearch.org › latest › tutorials › reranking › reranking-cohere](https://docs.opensearch.org/latest/tutorials/reranking/reranking-cohere/) - Reranking search results using Cohere Rerank

68. [BAAI/bge-reranker-large - Hugging Face](https://huggingface.co/BAAI/bge-reranker-large) - We’re on a journey to advance and democratize artificial intelligence through open source and open s...

69. [RankGPT Reranker Demonstration (Van Gogh Wiki)](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/rankGPT/)

70. [LLM4Ranking: An Easy-to-use Framework of Utilizing Large ... - arXiv](https://arxiv.org/html/2504.07439v1)

71. [Qwen3 Embedding: Advancing Text Embedding and Reranking ...](https://arxiv.org/html/2506.05176v1) - The Qwen3 Embedding series offers a spectrum of model sizes (0.6B, 4B, 8B) for both embedding and re...

72. [Anthropic's Transparency Hub](https://www.anthropic.com/transparency)

73. [How developers are using Gemini 1.5 Pro’s 1 million token context window](https://www.youtube.com/watch?v=cogrixfRvWw) - When Gemini 1.5 Pro was released, it immediately caught the attention of developers all over the wor...

74. [Data, Privacy, and Security for Microsoft 365 Copilot](https://learn.microsoft.com/en-us/copilot/microsoft-365/microsoft-365-copilot-privacy) - Microsoft 365 Copilot operates with multiple protections, which include, but aren't limited to, bloc...

75. [Build 2024: What's new for Microsoft Graph](https://devblogs.microsoft.com/microsoft365dev/build-2024-whats-new-for-microsoft-graph/) - In this blog, we'll highlight how you can expand the knowledge of Copilot for Microsoft 365 with Mic...

76. [LLMLingua: Innovating LLM efficiency with prompt compression](https://www.microsoft.com/en-us/research/blog/llmlingua-innovating-llm-efficiency-with-prompt-compression/) - LLMLingua identifies and removes unimportant tokens from prompts. This compression technique enables...

77. [NeurIPS Poster xRAG: Extreme Context Compression for Retrieval ...](https://neurips.cc/virtual/2024/poster/96497) - This paper introduces xRAG, an innovative context compression method tailored for retrieval-augmente...

78. [[PDF] Enhancing RAG Efficiency with Adaptive Context Compression](https://aclanthology.org/2025.findings-emnlp.1307.pdf)

79. [Enhancing RAG Efficiency with Adaptive Context ...](https://arxiv.org/html/2507.22931v1)

80. [Ragas: Automated Evaluation of Retrieval Augmented ...](https://arxiv.org/html/2309.15217v2)

81. [[PDF] RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://aclanthology.org/2024.eacl-demo.16.pdf)

82. [Evaluate Your Multimodal RAG Using Trulens - Zilliz blog](https://zilliz.com/blog/evaluating-multimodal-rags-in-practice-trulens) - Understand multimodal models and multimodal RAG as well as learn how to evaluate multimodal RAG syst...

83. [RAG evaluation metrics 2025: Ragas vs DeepEval ...](https://eonsr.com/en/rag-evaluation-2025-ragas-deepeval-trulens/) - RAG evaluation metrics 2025: compare Ragas, DeepEval, and TruLens with defaults, failure-mode mappin...

84. [LLM Tracing and Observability](https://arize.com/blog/llm-tracing-and-observability-with-arize-phoenix/) - Arize Phoenix, a popular open-source library for visualizing datasets and troubleshooting LLM-powere...

85. [GitHub - Arize-ai/phoenix: AI Observability & Evaluation](http://github.com/Arize-ai/phoenix) - AI Observability & Evaluation. Contribute to Arize-ai/phoenix development by creating an account on ...

86. [Arize AI Phoenix: Open-Source Tracing & Evaluation for AI (LLM/RAG/Agent)](https://www.youtube.com/watch?v=5PXRRXM8Iqo) - Welcome to my tutorial on using Phoenix by Arize AI, the open-source AI observability platform that'...

87. [What is LangSmith? Complete Guide to LLM Observability](https://www.articsledge.com/post/langsmith) - LangSmith is the Datadog for AI: it helps teams like Klarna & Uber debug, monitor & ship reliable LL...

88. [LangChain State of AI 2024 Report](https://blog.langchain.com/langchain-state-of-ai-2024/) - Dive into LangSmith product usage patterns that show how the AI ecosystem and the way people are bui...

89. [Transform Large Language Model Observability with Langfuse - AWS](https://aws.amazon.com/blogs/apn/transform-large-language-model-observability-with-langfuse/) - Learn how an AWS Advanced Technology Partner, Langfuse, offers an open-source LLM engineering platfo...

90. [Langfuse](https://langfuse.com) - Open Source LLM Engineering Platform. Traces, evals, prompt management and metrics to debug and impr...

91. [AI Agent Observability, Tracing & Evaluation with Langfuse](https://langfuse.com/blog/2024-07-ai-agent-observability-with-langfuse) - Langfuse is an open-source LLM engineering platform that provides deep insights into metrics such as...

92. [ClickHouse welcomes Langfuse: The future of open-source LLM ...](https://clickhouse.com/blog/clickhouse-acquires-langfuse-open-source-llm-observability) - We are thrilled to announce that ClickHouse has acquired Langfuse, the leading open-source platform ...

93. [LLM Observability & Application Tracing (Open Source) - Langfuse](https://langfuse.com/docs/observability/overview) - Open source application tracing and observability for LLM apps. Capture traces, monitor latency, tra...

94. [GitHub - zilliztech/GPTCache: Semantic cache for LLMs. Fully ...](https://github.com/zilliztech/gptcache) - Semantic cache for LLMs. Fully integrated with LangChain and llama_index. - GitHub - zilliztech/GPTC...

95. [GitHub - zilliztech/GPTCache: Semantic cache for LLMs. Fully integrated with LangChain and llama_index.](https://www.raghavgroups.com/zilliztech/GPTCache) - Semantic cache for LLMs. Fully integrated with LangChain and llama_index. - GitHub - zilliztech/GPTC...

96. [Optimizing Performance and Cost by Caching LLM Queries - Raga AI](https://raga.ai/resources/blogs/llm-cache-optimization) - An LLM cache stores previously generated responses, allowing your applications to access and reuse t...

97. [LangGraph: Agent Orchestration Framework for Reliable AI Agentswww.langchain.com › langgraph](https://www.langchain.com/langgraph) - Build controllable agents with LangGraph, our low-level agent orchestration framework

98. [Step B: The Nodes (the...](https://rahulkolekar.com/building-agentic-rag-systems-with-langgraph/) - Date: January 3, 2026 Category: Artificial Intelligence / Engineering Reading Time: 15 Minutes

99. [Building Agentic Workflows with LangGraph and Granite - IBM](https://www.ibm.com/think/tutorials/build-agentic-workflows-langgraph-granite) - In this tutorial, we'll explore how to build such AI agentic workflows by using two key tools: LangG...

100. [Reconstructing Context: Evaluating Advanced Chunking ...](https://www.alphaxiv.org/overview/2504.19754) - View recent discussion. Abstract: Retrieval-augmented generation (RAG) has become a transformative a...

101. [Haystack vs LlamaIndex vs LangChain – Which RAG Framework Should You Choose in 2025? | AgixTech](https://www.youtube.com/watch?v=cpKibCHLZD4) - Building enterprise-ready AI systems? This video compares Haystack, LangChain, and LlamaIndex to hel...

102. [DSPy: The framework for programming—not prompting—language ...](https://github.com/stanfordnlp/dspy) - DSPy is the framework for programming—rather than prompting—language models. It allows you to iterat...

103. [Optimize a RAG DSPy Application - Parea AI](https://docs.parea.ai/tutorials/dspy-rag-trace-evaluate/tutorial) - How many samples are necessary to achieve good performance with DSPy?

