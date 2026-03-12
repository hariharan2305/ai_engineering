# Vector Databases & Search Backends – Code Cheat‑Sheet (2026)

This cheat‑sheet shows **how to connect, upsert, and query** the main vector/search backends from the RAG Component Guide:

- Qdrant (self‑hosted / cloud)
- Weaviate
- Pinecone (gRPC client)
- pgvector (Postgres extension)
- OpenSearch / Amazon OpenSearch k‑NN
- Cloud RAG search: Vertex AI Search (via LangChain retriever)

All snippets follow **current docs as of early 2026**.[^1][^2][^3][^4][^5][^6][^7][^8][^9][^10][^11][^12][^13]

> These examples assume you already have **embeddings as Python lists/floats** (e.g., from the Embeddings cheat‑sheet) and focus on DB wiring.

***

## 1. Qdrant

Qdrant is an open‑source vector DB with HNSW indexing and hybrid (sparse+dense) support.[^14][^5]

### 1.1 Install client

```bash
pip install qdrant-client
```

### 1.2 Connect and create collection

```python
from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(host="localhost", port=6333)  # or cloud URL + api_key

collection_name = "documents"

client.recreate_collection(
    collection_name=collection_name,
    vectors=models.VectorParams(
        size=768,           # must match your embedding dim
        distance=models.Distance.COSINE,
    ),
)
print("Collection created:", collection_name)
```

This follows the Qdrant Python quickstart pattern.[^5][^7]

### 1.3 Upsert points (vectors + payload)

```python
texts = [
    "Qdrant is a vector database.",
    "Weaviate supports hybrid search.",
]

vectors = [
    [0.1] * 768,  # replace with real embeddings
    [0.2] * 768,
]

points = [
    models.PointStruct(
        id=i,
        vector=vectors[i],
        payload={"text": texts[i], "source": "demo"},
    )
    for i in range(len(texts))
]

client.upsert(
    collection_name=collection_name,
    points=points,
    wait=True,
)
print("Upserted", len(points), "points")
```

This mirrors the official `upsert` examples (PointStruct with `id`, `vector`, `payload`).[^1][^14]

### 1.4 Search with optional filter

```python
query_vector = [0.15] * 768

hits = client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=3,
    with_payload=True,
)

for hit in hits:
    print(hit.id, hit.score, hit.payload["text"])
```

For filtered search, use `query_filter` with `Filter` and `FieldCondition` as shown in the docs.[^15]

***

## 2. Weaviate

Weaviate is a schema‑driven vector DB with powerful **hybrid search** combining BM25 and dense vectors.[^2][^3]

### 2.1 Install client

```bash
pip install -U weaviate-client
```

### 2.2 Connect and create a collection

The new collections API is the recommended way in current Weaviate docs.[^3]

```python
import weaviate
from weaviate.classes.config import Property, DataType

client = weaviate.connect_to_local()  # or connect_to_weaviate_cloud / HTTP URL

client.collections.delete("JeopardyQuestion", ignore_missing=True)

jeopardy = client.collections.create(
    name="JeopardyQuestion",
    properties=[
        Property(name="question", data_type=DataType.TEXT),
        Property(name="answer", data_type=DataType.TEXT),
        Property(name="category", data_type=DataType.TEXT),
    ],
)

print("Collection created:", jeopardy.name)
```

### 2.3 Insert objects (Weaviate will embed if configured, or you can supply vectors)

```python
objects = [
    {"question": "This city is the capital of France", "answer": "Paris", "category": "Geography"},
    {"question": "This language is primarily spoken in Brazil", "answer": "Portuguese", "category": "Language"},
]

for obj in objects:
    jeopardy.data.insert(properties=obj)

print("Inserted", len(objects), "objects")
```

If you configure a custom vectorizer, you can send embeddings yourself via the `vectors` argument.

### 2.4 Hybrid search (dense + BM25)

From the official hybrid search examples:[^16][^2]

```python
from weaviate.classes.query import BM25Operator

jeopardy = client.collections.use("JeopardyQuestion")

response = jeopardy.query.hybrid(
    query="capital of France",
    query_properties=["question"],
    alpha=0.25,           # 0: pure BM25, 1: pure vector
    bm25_operator=BM25Operator.and_(),
    limit=3,
)

for obj in response.objects:
    print(obj.properties["question"], "->", obj.properties["answer"])
```

This is the canonical hybrid call from the latest Weaviate docs.[^2]

***

## 3. Pinecone (gRPC client)

Pinecone is a managed vector DB; the latest recommended Python client uses the **gRPC** interface (`PineconeGRPC`).[^4][^6]

### 3.1 Install

```bash
pip install "pinecone[grpc]"
```

### 3.2 Connect and target an index

```python
import os
from pinecone.grpc import PineconeGRPC as Pinecone

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# You must have already created the index in the Pinecone console or via SDK
index = pc.Index(host=os.environ["PINECONE_INDEX_HOST"])  # e.g. "docs-example-xxxx.svc.region.pinecone.io"
```

### 3.3 Upsert vectors

From the latest upsert docs:[^6]

```python
upsert_response = index.upsert(
    vectors=[
        {
            "id": "vec1",
            "values": [0.1, 0.2, 0.3, 0.4],
            "metadata": {"genre": "drama"},
        },
        {
            "id": "vec2",
            "values": [0.2, 0.3, 0.4, 0.5],
            "metadata": {"genre": "action"},
        },
    ],
    namespace="example-namespace",
)

print("Upserted count:", upsert_response["upsertedCount"])
```

### 3.4 Query vectors

```python
query_response = index.query(
    namespace="example-namespace",
    vector=[0.15, 0.25, 0.35, 0.45],
    top_k=3,
    include_values=False,
    include_metadata=True,
)

for match in query_response["matches"]:
    print(match["id"], match["score"], match["metadata"])
```

This is the current pattern recommended in the Pinecone Python client docs.[^4][^6]

***

## 4. pgvector (PostgreSQL extension)

pgvector adds a `vector` column type and `<->` distance operators. You can use **pure SQL** or integrate it with LangChain/LlamaIndex.[^10][^12]

### 4.1 SQL: create table and insert vectors

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE items (
    id bigserial PRIMARY KEY,
    embedding vector(3)
);

INSERT INTO items (embedding)
VALUES ('[1,2,3]'), ('[4,5,6]');
```

This mirrors the examples from the pgvector README.[^17][^12]

### 4.2 Python with psycopg2 – insert and similarity search

From recent pgvector tutorials:[^10]

```python
import psycopg2
import numpy as np

conn = psycopg2.connect("dbname=your_db user=your_user password=your_pass host=localhost")
cur = conn.cursor()

# Insert a vector
embedding = np.array([1.5, 2.5, 3.5])
cur.execute("INSERT INTO items (embedding) VALUES (%s)", (embedding.tolist(),))

# Perform similarity search
query_vector = np.array([2, 3, 4])
cur.execute(
    "SELECT id, embedding <-> %s AS distance FROM items ORDER BY distance LIMIT 5",
    (query_vector.tolist(),),
)

for row in cur.fetchall():
    print("id=", row, "distance=", row[^1])

conn.commit()
cur.close()
conn.close()
```

### 4.3 LangChain PGVector integration (optional)

From updated tutorials using `langchain_postgres`:[^10]

```python
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

connection_string = "postgresql://user:pass@localhost:5432/db_name"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = PGVector.from_documents(
    documents,  # list[langchain_core.documents.Document]
    embeddings,
    connection_string=connection_string,
    collection_name="docs_collection",
)

results = vector_store.similarity_search("Your query here", k=5)
for doc in results:
    print(doc.metadata, doc.page_content[:200])
```

***

## 5. OpenSearch / Amazon OpenSearch k‑NN Vector Search

OpenSearch and Amazon OpenSearch Service both support the `knn_vector` field type and `knn` queries.[^8][^18][^11][^13]

### 5.1 Create an index with k‑NN enabled (REST)

From the OpenSearch docs:[^11][^8]

```http
PUT my-knn-index-1
{
  "settings": {
    "index": {
      "knn": true,
      "knn.algo_param.ef_search": 100
    }
  },
  "mappings": {
    "properties": {
      "my_vector": {
        "type": "knn_vector",
        "dimension": 768,
        "space_type": "cosinesimil"
      },
      "text": { "type": "text" }
    }
  }
}
```

### 5.2 Index docs with vectors (Python `opensearch-py`)

```bash
pip install opensearch-py
```

```python
from opensearchpy import OpenSearch, helpers

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "admin"),  # or IAM-signed auth on Amazon OpenSearch
    use_ssl=False,
)

index_name = "my-knn-index-1"

# Example docs with embeddings
actions = []
for i in range(10):
    doc = {
        "_index": index_name,
        "_id": i,
        "my_vector": [0.01 * j for j in range(768)],  # replace with real embedding
        "text": f"Document {i}",
    }
    actions.append(doc)

helpers.bulk(client, actions)
client.indices.refresh(index=index_name)
```

### 5.3 k‑NN search query

```python
query_vec = [0.02] * 768

response = client.search(
    index=index_name,
    body={
        "size": 3,
        "query": {
            "knn": {
                "my_vector": {
                    "vector": query_vec,
                    "k": 3,
                }
            }
        },
    },
)

for hit in response["hits"]["hits"]:
    print(hit["_id"], hit["_score"], hit["_source"]["text"])
```

This is consistent with OpenSearch k‑NN examples in the latest docs and Python guide.[^18][^13][^8]

***

## 6. Vertex AI Search (Cloud RAG Search Backend)

Vertex AI Search is a managed search/RAG service; you typically configure the index in the console, then query via the **Discovery Engine API** or use LangChain’s `VertexAISearchRetriever`.[^9][^19][^20]

### 6.1 Using LangChain’s `VertexAISearchRetriever`

From the latest LangChain integration docs:[^9]

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
    max_documents=3,
)

query = "What does the PDF say about termination clauses?"
results = retriever.invoke(query)

for doc in results:
    print(doc.metadata.get("source"), "\n", doc.page_content[:300], "\n---\n")
```

Here, Vertex AI Search handles **indexing, chunking, and retrieval** over your content in GCS, websites, or other data sources; you just consume results in your RAG pipeline.[^19][^20][^9]

***

## 7. How to Use This Vector/Search Cheat‑Sheet

1. **Choose your backend** based on:
   - Cloud vs. self‑hosted, compliance, and scale.
   - Need for hybrid search (`Weaviate`, `OpenSearch` + BM25, `Qdrant` with sparse vectors, `Vertex AI Search`).[^14][^8][^2][^9]
2. **Copy the relevant connection + upsert + query snippet** into your repo.
3. Wire your **embedding step** so that every document and query gets encoded using the same model.
4. For managed backends like **Vertex AI Search** and **Bedrock KB**, treat them as retrievers in your RAG stack and skip manual index management where possible.[^20][^21][^9]

Keep this Markdown file alongside the Embeddings cheat‑sheet so you and your agents can quickly stand up or swap vector/search backends in your RAC/RAG systems.

---

## References

1. [Searching For Vectors​](https://docs.e2enetworks.com/docs/tir/VectorDatabase/Qdrant/PythonClient/) - Qdrant provides a Python client to interact with your Qdrant Database.

2. [Hybrid search](https://docs.weaviate.io/weaviate/search/hybrid) - Hybrid search combines the results of a vector search and a keyword (BM25F) search by fusing the two...

3. [Python | Weaviate Documentation](https://docs.weaviate.io/weaviate/client-libraries/python) - The Python client library is developed and tested using Python 3.8+. It is available on PyPI.org, an...

4. [pinecone - PyPI](https://pypi.org/project/pinecone/3.0.1/) - To see some more realistic examples of how this client can be used, explore some of our many Jupyter...

5. [Qdrant Quickstart](https://qdrant.tech/documentation/quickstart/) - In this short example, you will use the Python Client to create a Collection, load data into it and ...

6. [Upsert vectors - Pinecone Docs](https://docs.pinecone.io/reference/api/2024-07/data-plane/upsert) - The upsert operation writes vectors into a namespace. If a new value is upserted for an existing vec...

7. [Python client for Qdrant vector search engine - GitHub](https://github.com/qdrant/qdrant-client) - Client library and SDK for the Qdrant vector search engine. Library contains type definitions for al...

8. [Approximate k-NN search - OpenSearch Documentation](https://docs.opensearch.org/latest/vector-search/vector-search-techniques/approximate-knn/) - To use the approximate search functionality, you must first create a vector index with index.knn set...

9. [Configure and use the vertex...](https://docs.langchain.com/oss/python/integrations/retrievers/google_vertex_ai_search) - Integrate with the Google Vertex AI search retriever using LangChain Python.

10. [pgvector Tutorial: Integrate Vector Search into PostgreSQL](https://www.datacamp.com/tutorial/pgvector-tutorial) - Learn how to integrate vector search into PostgreSQL with pgvector. This tutorial covers installatio...

11. [k-NN vector - OpenSearch Documentation](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-vector/) - k-NN vector. Introduced 1.0. The knn_vector data type allows you to ingest vectors into an OpenSearc...

12. [pgvector/pgvector: Open-source vector similarity search for Postgres](https://github.com/pgvector/pgvector) - Storing. Create a new table with a vector column. CREATE TABLE items (id bigserial PRIMARY KEY, embe...

13. [opensearch-py/guides/plugins/knn.md at main · opensearch-project/opensearch-py](https://github.com/opensearch-project/opensearch-py/blob/main/guides/plugins/knn.md) - Python Client for OpenSearch. Contribute to opensearch-project/opensearch-py development by creating...

14. [Points - Qdrant](https://qdrant.tech/documentation/concepts/points/) - Python client optimizations. The Python client has additional features for loading points, which inc...

15. [Python Qdrant client library - PyPI](https://pypi.org/project/qdrant-client/0.11.8/) - Client library for the Qdrant vector search engine

16. [weaviate.collections.queries - Weaviate Python Client](https://weaviate-python-client.readthedocs.io/en/stable/weaviate.collections.queries.html) - weaviate.collections.queries.hybrid · weaviate.collections.queries.near_image ... Weaviate Python Cl...

17. [pgvector similarity search: Basics, tutorial and best practices](https://www.instaclustr.com/education/vector-database/pgvector-similarity-search-basics-tutorial-and-best-practices/) - You'll learn how to install pgvector, create tables with vector columns, insert and query high-dimen...

18. [k-Nearest Neighbor (k-NN) search in Amazon OpenSearch Service](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/knn.html) - k-NN for Amazon OpenSearch Service lets you search for points in a vector space and find the nearest...

19. [Use Vertex AI Search on PDFs (unstructured data) in Cloud Storage ...](https://codelabs.developers.google.com/codelabs/how-to-query-vertex-ai-search-cloud-run-service) - This codelab focuses on using Vertex AI Search, where you can build a Google-quality search app on y...

20. [Search from Vertex AI | Google quality search/RAG for enterprise](https://cloud.google.com/enterprise-search) - Vertex AI Search helps developers build secure, Google-quality search experiences for websites, intr...

21. [Create a knowledge base by connecting to a data source in Amazon ...](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-create.html) - To create a knowledge base, send a CreateKnowledgeBase request with an Agents for Amazon Bedrock bui...

