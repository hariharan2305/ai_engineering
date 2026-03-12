# Embedding Models – Code Cheat‑Sheet (2026)

This cheat‑sheet shows **how to call each embedding model** from the RAG Component Guide:

- OpenAI `text-embedding-3-*`
- BGE / BGE‑M3 (FlagEmbedding)
- NV‑Embed‑v1
- Generic E5 / GTE / other SentenceTransformers models
- Google: Vertex AI Text Embeddings (`text-embedding-005`) & Gemini API (`gemini-embedding-001`)
- AWS Bedrock: Titan Text Embeddings & Cohere Embed v3 (via Bedrock)
- Cohere Embed v3 (direct API)

All snippets are aligned with **current official docs/blogs as of early 2026**.[^1][^2][^3][^4][^5][^6][^7][^8][^9][^10][^11][^12]

> All examples are in Python. Set your API keys and cloud credentials via environment variables as appropriate.

***

## 1. OpenAI `text-embedding-3-small` / `text-embedding-3-large`

OpenAI’s current embeddings use the new `OpenAI` client and `client.embeddings.create()`.[^13][^5][^1]

### 1.1 Install and setup

```bash
pip install openai
```

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

### 1.2 Single text → embedding

```python
response = client.embeddings.create(
    model="text-embedding-3-small",  # or "text-embedding-3-large"
    input="Your text string goes here",
)

vector = response.data.embedding
print("Embedding dim:", len(vector))
print("First 5 dims:", vector[:5])
```

This matches the current OpenAI embeddings guide.[^1]

### 1.3 Batch encode documents

```python
texts = [
    "First document text",
    "Second document text",
    "Third document text",
]

resp = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts,
)

embeddings = [item.embedding for item in resp.data]
print("Num embeddings:", len(embeddings))
```

You can then insert `embeddings[i]` into your vector DB alongside document `texts[i]`.

***

## 2. BGE / BGE‑M3 via FlagEmbedding

BGE models (including **BGE‑M3**) are top open‑source retrieval models. The recommended interface is the **FlagEmbedding** library.[^4][^6]

### 2.1 Install

```bash
pip install -U FlagEmbedding
```

### 2.2 Dense + sparse embeddings with `BGEM3FlagModel`

From the official `BAAI/bge-m3` README on Hugging Face:[^6][^4]

```python
from FlagEmbedding import BGEM3FlagModel

# use_fp16=True for GPU, False for CPU
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

texts = [
    "What is BGE M3?",
    "Definition of BM25",
]

embeddings = model.encode(texts)

# Dense embeddings: embeddings["dense_vecs"]  -> shape (N, D)
# Sparse embeddings: embeddings["sparse_vecs"] -> dict format for BM25-like hybrid

print("Dense shape:", embeddings["dense_vecs"].shape)
print("First text dense (first 5 dims):", embeddings["dense_vecs"][:5])
```

Use `dense_vecs` for vector DB storage, and optionally `sparse_vecs` for hybrid retrieval where your backend supports it.

***

## 3. NV‑Embed‑v1 (NVIDIA)

NV‑Embed‑v1 is a high‑performing generalist embedding model on MTEB; it can be used via `sentence-transformers`.[^2]

### 3.1 Install

```bash
pip install -U sentence-transformers
```

### 3.2 Encode texts

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nvidia/NV-Embed-v1")

texts = [
    "This is the first document.",
    "This is another piece of content.",
]

embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

print("Embeddings shape:", embeddings.shape)
print("First vector (first 5 dims):", embeddings[:5])
```

Any other SentenceTransformers‑compatible embedding model (e.g., `thenlper/gte-large`, `intfloat/e5-base-v2`) uses the same interface.[^3]

***

## 4. Generic E5 / GTE / Mixedbread via SentenceTransformers

The **SentenceTransformers** docs show a standard pattern for all their pretrained models.[^14][^3]

### 4.1 Install (if not already)

```bash
pip install -U sentence-transformers
```

### 4.2 Example: E5 / GTE / Mixedbread

```python
from sentence_transformers import SentenceTransformer

# Pick the model you want:
# model_name = "intfloat/e5-base-v2"
# model_name = "thenlper/gte-large"
model_name = "mixedbread-ai/mxbai-embed-large-v1"

model = SentenceTransformer(model_name)

texts = [
    "The weather is lovely today.",
    "It is raining heavily.",
]

embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

print("Shape:", embeddings.shape)
```

Some models support **prompted encoding** (e.g., query vs document prompts) via `prompt_name` or `prompt`.[^15]

```python
query_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

q_emb = query_model.encode(
    "What are Pandas?",
    prompt_name="query",  # or explicit prompt string
)
```

***

## 5. Google – Vertex AI Text Embeddings (`text-embedding-005`)

Vertex AI’s text embedding API uses `TextEmbeddingModel`.[^7][^8][^16]

### 5.1 Install and init

```bash
pip install -U google-cloud-aiplatform
```

```python
import vertexai
from vertexai.language_models import TextEmbeddingModel

PROJECT_ID = "your-gcp-project-id"
REGION = "us-central1"  # choose the region where the model is available

vertexai.init(project=PROJECT_ID, location=REGION)

model = TextEmbeddingModel.from_pretrained("text-embedding-005")
```

### 5.2 Single text → embedding

```python
texts = ["What is cloud computing?"]

embeddings = model.get_embeddings(texts)

vector = embeddings.values
print("Embedding dim:", len(vector))
print("First 10 dims:", vector[:10])
```

This mirrors the official Vertex AI embedding examples.[^8][^7]

### 5.3 Batch embeddings

```python
texts = [
    "How to set up a load balancer on GCP",
    "Configuring autoscaling for GKE clusters",
]

embeddings = model.get_embeddings(texts)

for text, emb in zip(texts, embeddings):
    print("Text:", text[:40], "...")
    print("Dim:", len(emb.values))
```

***

## 6. Google – Gemini API Embeddings (`gemini-embedding-001`)

You can also use Gemini embeddings via the standalone Gemini API (`google-genai`) or LangChain’s integration.

### 6.1 Direct Gemini API (Python client)

From the Gemini API embeddings reference:[^17]

```bash
pip install -U google-genai
```

```python
from google import genai
from google.genai import types

client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

texts = [
    "What is the meaning of life?",
    "How does the brain work?",
]

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=texts,
    config=types.EmbedContentConfig(output_dimensionality=768),  # optional
)

embeddings = [e.values for e in result.embeddings]
print("Got", len(embeddings), "embeddings of dim", len(embeddings))
```

### 6.2 Via LangChain (GoogleGenerativeAIEmbeddings)

From the LangChain Gemini integration docs:[^9]

```bash
pip install -U langchain-google-genai
```

```python
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vec = embeddings.embed_query("hello, world!")
print(vec[:5])
```

This is convenient when you’re already using LangChain for RAG.[^9]

***

## 7. AWS Bedrock – Titan Text Embeddings & Cohere Embed v3

Amazon Bedrock exposes multiple embedding models behind a single API; **Titan Text Embeddings v2** and **Cohere Embed v3** are common choices.[^18][^10][^11]

### 7.1 Titan Text Embeddings v2 via `bedrock-runtime`

A Python pattern adapted from the Titan embeddings usage examples on HF and AWS docs:[^10][^11]

```bash
pip install boto3
```

```python
import json
import boto3

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

model_id = "amazon.titan-embed-text-v2:0"  # check AWS docs for latest version

text = "Please recommend books with a theme similar to the movie 'Inception'."

body = json.dumps({
    "inputText": text,
})

response = bedrock.invoke_model(
    modelId=model_id,
    body=body,
    contentType="application/json",
    accept="application/json",
)

payload = json.loads(response["body"].read())
vector = payload["embedding"]  # field name per Titan embed API

print("Embedding dim:", len(vector))
print("First 5 dims:", vector[:5])
```

Always confirm the response field name (`embedding` vs `embeddings`) in the Titan embed model parameters page.[^11]

### 7.2 Cohere Embed v3 via Cohere API (direct)

Cohere Embed v3 models are available directly via Cohere and via Bedrock; here is the **Cohere API usage**.[^12]

```bash
pip install -U cohere
```

```python
import cohere
import numpy as np

co = cohere.Client("YOUR_COHERE_API_KEY")

# Documents to index (search_document)
docs = [
    "The capital of France is Paris",
    "PyTorch is a machine learning framework based on the Torch library.",
    "The average cat lifespan is between 13-17 years",
]

doc_emb = co.embed(
    docs,
    input_type="search_document",
    model="embed-english-v3.0",
).embeddings

doc_emb = np.asarray(doc_emb)

# Query embedding (search_query)
query = "What is Pytorch"
query_emb = co.embed(
    [query],
    input_type="search_query",
    model="embed-english-v3.0",
).embeddings

query_emb = np.asarray(query_emb)

scores = np.dot(query_emb, doc_emb.T)

print("Query:", query)
for idx in np.argsort(-scores):
    print(f"Score: {scores[idx]:.2f}")
    print(docs[idx])
    print("--------")
```

This snippet is copied from the official Cohere embed‑english‑v3.0 repository.[^12]

> To use Cohere Embeddings via **Bedrock**, wrap the same JSON payload (`texts`, `input_type`, etc.) in `bedrock-runtime.invoke_model()` calls as shown above for Titan and documented in the Cohere Embed v3 Bedrock parameters page.[^18]

***

## 8. How to Plug These Embeddings into Your RAG Stack

1. **Choose an embedding backend** based on:
   - Cloud vs. self‑hosted (OpenAI / Vertex / Bedrock vs BGE / NV‑Embed / E5‑GTE).
   - Multilingual needs and licensing.
   - Infrastructure (GPU availability, on‑prem, etc.).
2. **Copy the relevant snippet** to:
   - Turn your documents and queries into vectors.
   - Store those vectors in your chosen vector DB (Qdrant, Weaviate, Milvus, pgvector, Pinecone, OpenSearch, Azure AI Search, Bedrock KB).
3. **Keep query and document embeddings consistent**:
   - Same model and config.
   - Correct `input_type` / prompts for models that differentiate query vs document (Cohere, BGE‑M3, some E5/GTE variants).[^15][^6][^18]

Keep this Markdown file next to your RAG repo so you and your agents can quickly wire or swap embedding backends while experimenting or deploying.

---

## References

1. [Vector embeddings | OpenAI API](https://developers.openai.com/api/docs/guides/embeddings/) - Learn how to turn text into numbers, unlocking use cases like search, clustering, and more with Open...

2. [NV-Embed-v1](https://paddlenlp.readthedocs.io/zh/latest/website/nvidia/NV-Embed-v1/index.html)

3. [Pretrained Models — Sentence Transformers documentation](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)

4. [BAAI/bge-m3 - Hugging Face](https://huggingface.co/BAAI/bge-m3) - pip install -U FlagEmbedding. Generate Embedding for text. Dense Embedding. from FlagEmbedding impor...

5. [New embedding models and API updates - OpenAI](https://openai.com/index/new-embedding-models-and-api-updates/) - We are introducing two new embedding models: a smaller and highly efficient text-embedding-3-small m...

6. [FlagEmbedding 1.2.7 - PyPI](https://pypi.org/project/FlagEmbedding/1.2.7/) - In this project, we introduce BGE-M3, the first embedding model which supports multiple retrieval mo...

7. [How to Create Text Embeddings Using the Vertex AI Embedding API](https://oneuptime.com/blog/post/2026-02-17-how-to-create-text-embeddings-using-the-vertex-ai-embedding-api/view) - Learn how to generate text embeddings using the Vertex AI Embedding API for semantic search, cluster...

8. [Text embeddings API | Generative AI on Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api)

9. [Google generative AI (AI studio & Gemini API) integration](https://docs.langchain.com/oss/python/integrations/text_embedding/google_generative_ai) - To access Google Generative AI embedding models you'll need to create a Google Cloud project, enable...

10. [Invoke Amazon Titan Text Embeddings on Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModelWithResponseStream_TitanTextEmbeddings_section.html) - Find the complete example and learn how to set up and run in the AWS Code Examples Repository . Crea...

11. [amazon/Titan-text-embeddings-v2 - Hugging Face](https://huggingface.co/amazon/Titan-text-embeddings-v2) - We’re on a journey to advance and democratize artificial intelligence through open source and open s...

12. [Cohere/Cohere-embed-english-v3.0 - Hugging Face](https://huggingface.co/Cohere/Cohere-embed-english-v3.0) - We’re on a journey to advance and democratize artificial intelligence through open source and open s...

13. [How to generate text embeddings using OpenAI in Python](https://mljar.com/notebooks/openai-embedding/) - See how to generate text embeddings using OpenAI models in Python. This notebook covers generating e...

14. [SentenceTransformers Documentation](https://sbert.net)

15. [Training with Prompts — Sentence Transformers documentation](https://sbert.net/examples/sentence_transformer/training/prompts/README.html) - These prompts are strings, prefixed to each text to be embedded, allowing the model to distinguish b...

16. [Get text embeddings | Generative AI on Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings) - To see an example of getting text embeddings ... You can get text embeddings for a snippet of text b...

17. [Embeddings | Gemini API | Google AI for Developers](https://ai.google.dev/api/embeddings)

18. [Cohere Embed v3 - Amazon Bedrock - AWS Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html) - The Cohere Embed models have the following inference parameters. texts – An array of strings for the...

