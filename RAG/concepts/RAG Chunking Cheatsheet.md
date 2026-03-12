# Chunking & Indexing Logic – Code Cheat‑Sheet (2026)

This cheat‑sheet gives **component‑level Python snippets** for the chunking & indexing tools mentioned in the RAG Component Guide:

- LangChain text splitters (including RecursiveCharacterTextSplitter)
- LlamaIndex node parsers & indices (HierarchicalNodeParser, Markdown/HTML parsers, VectorStoreIndex)
- Amazon Bedrock Knowledge Bases (managed chunking & indexing)
- Azure AI Search (index + indexer for RAG ingestion)

All examples are aligned with **current docs as of early 2026**.[^1][^2][^3][^4][^5]

> These snippets focus on *how to call the APIs*; they assume you already have raw text or parsed documents from your ingestion layer.

***

## 1. LangChain Text Splitters

LangChain provides many splitters; the most used in RAG is **RecursiveCharacterTextSplitter**.[^6][^7]

### 1.1 Install LangChain

```bash
pip install langchain-core langchain-text-splitters
# For vector DBs and LLMs you’ll add extra packages (e.g., langchain-openai, langchain-qdrant)
```

### 1.2 Basic RecursiveCharacterTextSplitter

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

raw_text = """Your long document text goes here..."""

# Simple default splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
)

chunks = splitter.split_text(raw_text)

print("Num chunks:", len(chunks))
print(chunks[:300])
```

The default splitter uses a hierarchy of separators (e.g., double newline, newline, space, empty string) to keep larger structures together, then falls back to smaller ones only if needed.[^7][^8]

### 1.3 Sentence‑aware RecursiveCharacterTextSplitter (recommended tweak)

LangChain docs and maintainers recommend customizing `separators` if you want better sentence‑level behavior.[^8][^7]

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    chunk_size=1000,
    chunk_overlap=150,
)

chunks = splitter.split_text(raw_text)
```

### 1.4 Splitting LangChain `Document` objects

```python
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = [
    Document(page_content="Some long content...", metadata={"source": "file1"}),
    Document(page_content="Another doc...", metadata={"source": "file2"}),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

split_docs = splitter.split_documents(docs)

print("Split into", len(split_docs), "chunks")
print(split_docs.page_content[:200])
print(split_docs.metadata)
```

### 1.5 Indexing chunks into a vector store (example with Qdrant)

```bash
pip install qdrant-client langchain-qdrant
```

```python
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# 1) Split docs as above into `split_docs`

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

qdrant_client = QdrantClient(host="localhost", port=6333)

vector_store = QdrantVectorStore.from_documents(
    split_docs,
    embedding=embeddings,
    collection_name="my_rag_chunks",
    client=qdrant_client,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

results = retriever.invoke("What does the document say about X?")
for r in results:
    print(r.metadata["source"], "::", r.page_content[:200])
```

This pattern (split → embed → index) is reusable for other vector DB integrations (Pinecone, Weaviate, pgvector, etc.).

***

## 2. LlamaIndex Node Parsers & Indices

LlamaIndex organizes text into **Nodes** and various **Index** types (vector, tree, list, graph, KG). Node parsers control chunking; indices control indexing & retrieval.[^3]

### 2.1 Install LlamaIndex core

```bash
pip install llama-index-core
# plus the specific storage/vector DB/LLM integrations you need (e.g., llama-index-vector-stores-qdrant)
```

> If you used `llama-index` monolith previously, new projects should follow the **modular packages** in the official docs.[^3]

### 2.2 Basic Sentence / SimpleNodeParser (flat chunks)

```python
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser

raw_text = """Your long document text..."""

doc = Document(text=raw_text)
parser = SimpleNodeParser.from_defaults(chunk_size=800, chunk_overlap=200)

nodes = parser.get_nodes_from_documents([doc])

print("Num nodes:", len(nodes))
print(nodes.text[:300])
print(nodes.metadata)
```

### 2.3 HierarchicalNodeParser (parent‑child chunking)

The **HierarchicalNodeParser** creates a hierarchy of overlapping parent/child chunks (e.g., 2048 → 512 → 128 tokens).[^9][^10][^3]

```python
from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser

raw_text = """Long document text here..."""

doc = Document(text=raw_text)

hier_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128],  # parent, child, grandchild
)

nodes = hier_parser.get_nodes_from_documents([doc])

print("Num hierarchical nodes:", len(nodes))

# Each node has relationships to parents/children via metadata
node = nodes
print(node.text[:300])
print(node.metadata.get("node_relationships", {}))
```

### 2.4 MarkdownNodeParser / HTMLNodeParser (structure‑aware)

```python
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import Document

markdown_text = """# Title\n\n## Section A\nSome content...\n\n## Section B\nMore content..."""

doc = Document(text=markdown_text)
md_parser = MarkdownNodeParser.from_defaults()

nodes = md_parser.get_nodes_from_documents([doc])

for n in nodes:
    print("PATH:", n.metadata.get("section_path"))
    print(n.text[:150])
    print("---")
```

LlamaIndex’s hierarchical node parsers and format‑specific parsers (Markdown, HTML, code hierarchy, etc.) are documented in the **Node Parsers** section.[^11][^3]

### 2.5 Building a VectorStoreIndex from nodes

```bash
pip install llama-index-vector-stores-qdrant qdrant-client
pip install llama-index-embeddings-openai
```

```python
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import StorageContext, VectorStoreIndex

# Assume `nodes` from any parser

qdrant_client = QdrantClient(host="localhost", port=6333)
qdrant_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="llamaindex_rag_chunks",
)

storage_context = StorageContext.from_defaults(vector_store=qdrant_store)

embed_model = OpenAIEmbedding(model="text-embedding-3-small")

index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model,
)

query_engine = index.as_query_engine(similarity_top_k=5)

response = query_engine.query("What does the document say about X?")
print(response)
```

This is the canonical pattern for **chunk → nodes → index → query** in LlamaIndex’s current modular API.[^3]

***

## Strategy Quick Reference

A decision guide for choosing the right chunking strategy. Based on hands-on experimentation (see `RAG/rag_lab/`) and industry patterns.

### LangChain Text Splitters

| Splitter | Module | What it does | Industry verdict |
|---|---|---|---|
| `RecursiveCharacterTextSplitter` | `langchain_text_splitters` | Separator hierarchy: `\n\n` → `\n` → `.!?` → ` ` → chars | **Industry default. Use this first, always.** |
| `CharacterTextSplitter` | `langchain_text_splitters` | Splits on a single character only (default `\n\n`) | Redundant — recursive is strictly better |
| `TokenTextSplitter` | `langchain_text_splitters` | Splits by token count using tiktoken | Useful when you need to respect LLM context window limits precisely |
| `SentenceTransformersTokenTextSplitter` | `langchain_text_splitters` | Splits by the embedding model's own token count | Niche — useful when your embedding model has a hard token limit (e.g. 256 tokens) |
| `SemanticChunker` | `langchain_experimental.text_splitter` | Embeds sentences, splits on topic shifts via cosine similarity | Powerful but 2-3x ingestion cost. Use on heterogeneous, infrequently updated corpora |
| `MarkdownHeaderTextSplitter` | `langchain_text_splitters` | Splits on `#`, `##`, `###` headers, preserves hierarchy in metadata | Very useful for docs, wikis, README-style content |
| `HTMLHeaderTextSplitter` | `langchain_text_splitters` | Splits on HTML `<h1>`, `<h2>` etc. | Useful for web-scraped content |
| `NLTKTextSplitter` | `langchain_text_splitters` | Uses NLTK sentence tokenizer | Rarely used — NLTK is dated, recursive handles most cases |
| `SpacyTextSplitter` | `langchain_text_splitters` | Uses spaCy for sentence splitting | Occasionally useful for multilingual content |
| Language-aware (`from_language`) | `langchain_text_splitters` | Code-aware splitting for Python, JS, Go, etc. | Useful for code search / documentation RAG |

> **Note on SemanticChunker import:** it lives in `langchain_experimental.text_splitter`, NOT `langchain_text_splitters`. Always verify module paths with docs before importing.

### LlamaIndex Node Parsers

| Parser | Module | What it does | Industry verdict |
|---|---|---|---|
| `SentenceSplitter` | `llama_index.core.node_parser` | Splits on sentence boundaries, respects chunk_size | **LlamaIndex's default. Equivalent to RecursiveCharacterTextSplitter.** |
| `TokenTextSplitter` | `llama_index.core.node_parser` | Splits strictly by token count | Same use case as LangChain's TokenTextSplitter |
| `SemanticSplitterNodeParser` | `llama_index.core.node_parser` | Semantic chunking — same concept as LangChain's SemanticChunker | Same trade-off: better quality, high ingestion cost |
| `SentenceWindowNodeParser` | `llama_index.core.node_parser` | Indexes individual sentences; attaches surrounding context window at retrieval time | **Unique to LlamaIndex. Pairs with `MetadataReplacementPostProcessor` at retrieval — cannot be tested in chunking isolation.** |
| `HierarchicalNodeParser` | `llama_index.core.node_parser` | Creates parent/child chunks at multiple sizes (e.g. 2048→512→128) simultaneously | **Unique to LlamaIndex. Pairs with `AutoMergingRetriever` — cannot be tested in chunking isolation.** |
| `MarkdownNodeParser` | `llama_index.core.node_parser` | Splits Markdown respecting header hierarchy | Equivalent to LangChain's MarkdownHeaderTextSplitter |
| `JSONNodeParser` | `llama_index.core.node_parser` | Splits JSON documents by keys/structure | Useful for structured data RAG |
| `CodeSplitter` | `llama_index.core.node_parser` | Language-aware code splitting | Same niche as LangChain's language splitter |
| `LangchainNodeParser` | `llama_index.core.node_parser` | Wraps any LangChain splitter for use inside LlamaIndex pipelines | Useful when mixing both frameworks |

> **Note on SentenceWindowNodeParser and HierarchicalNodeParser:** these are chunking strategies but they couple tightly with specific retrieval components. Changing only the chunker without the matching retriever will produce worse results, not better. Test these as a chunking+retrieval pair.

### Head-to-Head: LangChain vs LlamaIndex

| Capability | LangChain | LlamaIndex |
|---|---|---|
| General-purpose chunking | `RecursiveCharacterTextSplitter` | `SentenceSplitter` |
| Semantic chunking | `SemanticChunker` | `SemanticSplitterNodeParser` |
| Structured docs (Markdown/HTML) | Stronger — dedicated splitters | Basic |
| Advanced retrieval-coupled patterns | Not built in | `SentenceWindowNodeParser`, `HierarchicalNodeParser` |
| Code splitting | Good | Good |
| Wrapping the other framework | — | `LangchainNodeParser` |

***

## 3. Amazon Bedrock Knowledge Bases (Managed Chunking & Indexing)

Amazon Bedrock Knowledge Bases (KBs) handle **embedding, chunking, vector store setup, and retrieval** for you. You create a KB, connect data sources, and Bedrock manages ingestion (including chunking configuration in the console).[^12][^1]

### 3.1 Install Boto3 and create a Bedrock Agent client

```bash
pip install boto3
```

```python
import boto3

bedrock_agent = boto3.client("bedrock-agent", region_name="us-east-1")
```

### 3.2 Create a vector‑based knowledge base

From the official example in the Bedrock user guide & Boto3 docs:[^13][^1]

```python
import uuid
from botocore.exceptions import ClientError


def create_knowledge_base(bedrock_agent_client, name, role_arn, description=None):
    """Create a new vector knowledge base backed by OpenSearch Serverless."""
    try:
        kwargs = {
            "name": name,
            "roleArn": role_arn,
            "knowledgeBaseConfiguration": {
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": (
                        "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v1"
                    )
                },
            },
            "storageConfiguration": {
                "type": "OPENSEARCH_SERVERLESS",
                "opensearchServerlessConfiguration": {
                    "collectionArn": "arn:aws:aoss:us-east-1::123456789012:collection/abcdefgh12345678defgh",
                    "fieldMapping": {
                        "metadataField": "metadata",
                        "textField": "text",
                        "vectorField": "vector",
                    },
                    "vectorIndexName": "rag-index-uuid",
                },
            },
            "clientToken": "client-token-" + str(uuid.uuid4()),
        }

        if description:
            kwargs["description"] = description

        response = bedrock_agent_client.create_knowledge_base(**kwargs)
        kb = response["knowledgeBase"]
        print("Created KB:", kb["knowledgeBaseId"], kb["knowledgeBaseArn"])
        return kb

    except ClientError as err:
        print("Error creating KB:", err)
        raise


kb = create_knowledge_base(
    bedrock_agent,
    name="my-rag-kb",
    role_arn="arn:aws:iam::123456789012:role/AmazonBedrockExecutionRoleForKnowledgeBase",
)
```

The **chunking and embedding settings** for documents are configured per data source in the console (or via additional API calls for data sources), not in this function.[^1][^12]

### 3.3 Query the KB (retrieval for RAG)

Once the KB and data sources are set up, use the `Retrieve` or `RetrieveAndGenerate` APIs to get chunks for RAG.

```python
response = bedrock_agent.retrieve(
    knowledgeBaseId=kb["knowledgeBaseId"],
    retrievalQuery={"text": "Explain policy X for customers in region Y"},
    retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 10}},
)

for result in response["retrievalResults"]:
    print(result["content"], "\n---\n")
```

Bedrock will have **chunked & indexed** content according to your KB’s configuration.[^12]

***

## 4. Azure AI Search – Index + Indexer for RAG

Azure AI Search can **chunk and vectorize content via skills‑based indexers** and store it in a **search index** that you can query for RAG.[^2][^4]

### 4.1 Install Azure Search SDK

```bash
pip install azure-search-documents azure-identity
```

### 4.2 Create a vector‑enabled index (Python)

From the Azure AI Search RAG pipeline tutorial (simplified):[^2]

```python
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)

endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]  # e.g. https://<service>.search.windows.net
api_key = os.environ["AZURE_SEARCH_API_KEY"]

index_client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))

index_name = "py-rag-tutorial-idx"

fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="content", type=SearchFieldDataType.String),
    SimpleField(name="parent_id", type=SearchFieldDataType.String, filterable=True),
    SearchField(
        name="contentVector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=1536,  # must match your embedding model
        vector_search_profile_name="myHnswProfile",
    ),
]

vector_search = VectorSearch(
    algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
    profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")],
)

index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)

result = index_client.create_or_update_index(index)
print("Created index:", result.name)
```

### 4.3 Create an indexer for skills‑based chunking & embedding

With Azure AI Search, an **indexer** connects a data source (e.g., Blob Storage) and a skillset that can chunk + embed documents (via Azure OpenAI vectorizer).[^4][^2]

```python
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndexer,
    SearchIndexerDataSourceConnection,
    SearchIndexerDataContainer,
)

indexer_client = SearchIndexerClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))

# Example: Blob data source

data_source = SearchIndexerDataSourceConnection(
    name="blob-datasource",
    type="azureblob",
    connection_string=os.environ["AZURE_BLOB_CONNECTION_STRING"],
    container=SearchIndexerDataContainer(name="docs"),
)

indexer_client.create_or_update_data_source_connection(data_source)

indexer = SearchIndexer(
    name="rag-indexer",
    data_source_name=data_source.name,
    target_index_name=index_name,
    # In a full RAG pipeline you would attach a skillset here that:
    #  - extracts text from PDFs/Office
    #  - chunks content
    #  - calls Azure OpenAI embeddings into `contentVector`
)

indexer_client.create_or_update_indexer(indexer)
indexer_client.run_indexer(indexer.name)

print("Indexer started: rag-indexer")
```

The skillset configuration (for chunking + embedding) is defined in separate JSON or SDK calls; see Azure’s *RAG solution pipeline* tutorial for full code.[^4][^2]

### 4.4 Querying the index for RAG

```python
from azure.search.documents import SearchClient

search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))

results = search_client.search(
    search_text="climate policy for region X",  # full‑text query
    vector={"value": your_query_embedding, "k": 5, "fields": "contentVector"},
)

for doc in results:
    print(doc["id"], doc["content"][:200])
```

Azure handles the **chunking & embedding via indexers + skillsets**, and you query via text + vector hybrid search.[^2][^4]

***

## How to Use This Chunking & Indexing Cheat‑Sheet in Your RAC/RAG

1. **Select your framework**:
   - LangChain → use `RecursiveCharacterTextSplitter` and a vector store.
   - LlamaIndex → use node parsers (Simple/Hierarchical/Markdown) + `VectorStoreIndex`.
   - Managed cloud → use Bedrock KB or Azure AI Search with indexers.
2. **Copy the relevant snippet** and adapt:
   - Replace file paths, environment variables, and model names.
   - Point to your vector DB (Qdrant, Pinecone, Weaviate, OpenSearch, Azure AI Search, Bedrock KB).
3. **Wire into your RAG pipeline**:
   - Parsing → Chunking (this sheet) → Embeddings → Retrieval.

Keep this Markdown file alongside your RAG repo so you and your agents can quickly wire up or swap chunking/indexing components.

---

## References

1. [create_knowledge_base - Boto3 1.42.59 documentation](https://docs.aws.amazon.com/boto3/latest/reference/services/bedrock-agent/client/create_knowledge_base.html) - To create a knowledge base, you must first set up your data sources and configure a supported vector...

2. [Run A Query To Check Results](https://docs.azure.cn/en-us/search/tutorial-rag-build-solution-pipeline) - Create an indexer-driven pipeline that loads, chunks, embeds, and ingests content for RAG solutions ...

3. [Hierarchical - LlamaIndex](https://developers.llamaindex.ai/python/framework-api-reference/node_parsers/hierarchical/) - Hierarchical node parser. Splits a document into a recursive hierarchy Nodes using a NodeParser. NOT...

4. [Create an indexer - Azure AI Search | Microsoft Learn](https://learn.microsoft.com/en-us/azure/search/search-how-to-create-indexers) - This article explains the basic steps for creating an indexer that automates data ingestion for supp...

5. [Partitioning - Unstructured](https://docs.unstructured.io/open-source/core-functionality/partitioning) - The easiest way to partition documents in unstructured is to use the partition function. If you call...

6. [RecursiveCharacterTextSplitter Explained (The Most Important Text ...](https://www.blog.qualitypointtech.com/2026/01/recursivecharactertextsplitter.html?m=1) - RecursiveCharacterTextSplitter Explained (The Most Important Text Splitter in LangChain) ; what it i...

7. [[langchain]: The default list of RecursiveCharacterTextSplitter should ...](https://github.com/langchain-ai/docs/issues/1175) - It seems that the default list of RecursiveCharacterTextSplitter should include sentence splitting c...

8. [RecursiveCharacterTextSplitter separator order - LangChain Forum](https://forum.langchain.com/t/recursivecharactertextsplitter-separator-order/176) - Hi! It's not very clear to me how the order of the items in the separator list, which is a parameter...

9. [HierarchicalNodeParser](https://docs.llamaindex.ai/en/stable/api/llama_index.node_parser.HierarchicalNodeParser.html)

10. [HierarchicalNodeParser - LlamaIndex v0.10.10](https://llamaindexxx.readthedocs.io/en/latest/api/llama_index.core.node_parser.HierarchicalNodeParser.html) - Splits a document into a recursive hierarchy Nodes using a NodeParser. NOTE: this will return a hier...

11. [Code Hierarchy Node Parser - llama-index-packs - GitHub](https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-code-hierarchy/examples/CodeHierarchyNodeParserUsage.ipynb) - Code Hierarchy Node Parser¶ · Open In Colab. The CodeHierarchyNodeParser is useful to split long cod...

12. [Create a knowledge base by connecting to a data source in Amazon ...](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-create.html) - To create a knowledge base, send a CreateKnowledgeBase request with an Agents for Amazon Bedrock bui...

13. [Use CreateKnowledgeBase with an AWS SDK - Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-agent_example_bedrock-agent_CreateKnowledgeBase_section.html) - The following code example shows how to use CreateKnowledgeBase . Python. SDK for Python (Boto3). No...

