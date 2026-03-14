# Embedding Models – Code Cheat‑Sheet (2026)

This cheat‑sheet covers **all embedding model providers** used in the RAG lab, across **all modalities** — text, image, video, audio, code, and documents. Multimodal RAG is a first-class concern throughout.

Providers covered:
- OpenAI `text-embedding-3-*`
- BGE / BGE‑M3 (FlagEmbedding)
- NV‑Embed‑v1
- Generic E5 / GTE / SentenceTransformers models
- CLIP (image + text, via SentenceTransformers)
- Google: Vertex AI `text-embedding-005`, Gemini API `gemini-embedding-001`, `gemini-embedding-2-preview` (multimodal)
- AWS Bedrock: Titan Text Embeddings & Cohere Embed v3
- Cohere Embed v3 (direct API)
- Voyage AI (`voyage-3`, `voyage-3-lite`, `voyage-code-3`)
- Jina AI (`jina-embeddings-v3`)
- Perplexity AI (`pplx-embed-v1`, `pplx-embed-context-v1`)

All snippets are aligned with **current official docs as of March 2026**.

> All examples are in Python. Set your API keys and cloud credentials via environment variables as appropriate.

---

## Model Quick Reference

### Text Models

| Model | Provider | Dims | Context (tokens) | Symmetric? | Cost |
|---|---|---|---|---|---|
| `text-embedding-3-small` | OpenAI | 1536 (reducible) | 8191 | Yes | Low |
| `text-embedding-3-large` | OpenAI | 3072 (reducible) | 8191 | Yes | Medium |
| `all-MiniLM-L6-v2` | SentenceTransformers | 384 | 256 | Yes | Free (local) |
| `all-mpnet-base-v2` | SentenceTransformers | 768 | 514 | Yes | Free (local) |
| `BAAI/bge-small-en-v1.5` | SentenceTransformers | 384 | 512 | No (needs prefix) | Free (local) |
| `BAAI/bge-large-en-v1.5` | SentenceTransformers | 1024 | 512 | No (needs prefix) | Free (local) |
| `BAAI/bge-m3` | FlagEmbedding | 1024 | 8192 | No | Free (local) |
| `intfloat/e5-base-v2` | SentenceTransformers | 768 | 512 | No (needs prefix) | Free (local) |
| `nvidia/NV-Embed-v1` | SentenceTransformers | 4096 | 32768 | No | Free (local) |
| `pplx-embed-v1-0.6B` | Perplexity (HF) | 1024 | 32K | Yes | Free (local) |
| `pplx-embed-v1-4B` | Perplexity (HF) | 2560 | 32K | Yes | Free (local) |
| `pplx-embed-context-v1-0.6B` | Perplexity (HF) | 1024 | 32K | Yes | Free (local) |
| `pplx-embed-context-v1-4B` | Perplexity (HF) | 2560 | 32K | Yes | Free (local) |
| `voyage-3` | Voyage AI | 1024 | 32000 | No (input_type) | Low |
| `voyage-3-lite` | Voyage AI | 512 | 32000 | No (input_type) | Very low |
| `voyage-code-3` | Voyage AI | 1024 | 32000 | No (input_type) | Low |
| `jina-embeddings-v3` | Jina AI | 1024 | 8192 | No (task param) | Low |
| `text-embedding-005` | Google Vertex AI | 768 | 2048 | Yes | Low |
| `gemini-embedding-001` | Google Gemini API | 3072 (reducible) | 2048 | Yes | Low |
| `embed-english-v3.0` | Cohere | 1024 | 512 | No (input_type) | Low |
| `amazon.titan-embed-text-v2:0` | AWS Bedrock | 1024 | 8192 | Yes | Low |

### Multimodal Models

| Model | Provider | Modalities | Dims | Cost |
|---|---|---|---|---|
| `gemini-embedding-2-preview` | Google Gemini API | text, image, video, audio, documents | 3072 (reducible) | Low |
| `clip-ViT-B-32` | SentenceTransformers (HF) | text + image | 512 | Free (local) |
| `clip-ViT-L-14` | SentenceTransformers (HF) | text + image | 768 | Free (local) |
| `voyage-multimodal-3` | Voyage AI | text + image | 1024 | Low |

> **Symmetric vs asymmetric:** Symmetric = same encoding for queries and documents. Asymmetric = use different prefixes or `input_type` for queries vs documents — omitting this silently degrades retrieval quality.

> **Normalization:** OpenAI, Voyage, Cohere, Jina, and Perplexity API normalize automatically. Local SentenceTransformers models do not — always pass `normalize_embeddings=True`.

---

## Core Concepts: Dense vs Sparse Vectors

Understanding this distinction is essential before choosing a retrieval strategy. Most models produce one or the other. BGE-M3 produces both.

### Dense Embeddings

Every dimension has a value. Nothing is zero.

```
Text: "FastAPI is a Python web framework"

Dense vector (1024 dims):
[0.12, -0.34, 0.89, 0.05, -0.67, 0.23, -0.11, 0.78, ...]
```

Dimensions do not map to specific words — they are learned abstract features. You cannot read a dense vector and know what it means. But two semantically similar sentences will produce vectors that are close together in this space.

```
"FastAPI is a Python web framework"   →  [0.12, -0.34, 0.89, ...]
"Flask is used to build Python APIs"  →  [0.11, -0.31, 0.85, ...]  ← close
"My cat likes tuna"                   →  [0.67,  0.89, -0.23, ...]  ← far
```

**Dense = semantic meaning. Answers: "what is this text ABOUT?"**

### Sparse Embeddings

Most dimensions are zero. Only a handful have non-zero values, and those map directly to vocabulary terms. Stored as a dict because saving all the zeros is wasteful.

```
Text: "FastAPI is a Python web framework"

Sparse vector (stored as dict of non-zero terms):
{
  "fastapi":   0.87,
  "python":    0.65,
  "framework": 0.43,
  "web":       0.31,
}
# All other ~30,000 vocabulary terms are 0.0 — not stored
```

You CAN read a sparse vector. It tells you which words matter and by how much.

**Sparse = keyword signals. Answers: "what specific WORDS does this text contain?"**

### Why Each One Fails Alone

| Scenario | Dense | Sparse |
|---|---|---|
| Query: "car" → Doc: "automobile" | Finds it (semantic match) | Misses it (different word) |
| Query: "GPT-4o" → Doc: "GPT-4o release notes" | May confuse with similar model names | Finds it (exact term match) |
| Query: "how does attention work" → Doc explaining transformers | Finds it | Misses it (no word overlap) |
| Query: "errno 403" → Doc with exact error code | May drift to unrelated errors | Finds it (exact token) |

Dense misses exact keyword matches. Sparse misses paraphrases and synonyms. Hybrid search runs both and merges the scores — this is why hybrid retrieval substantially outperforms either alone on mixed corpora.

### BM25 vs SPLADE-Style Sparse

Not all sparse vectors are equal. There are two generations:

**BM25 — statistical counting, no language understanding**

Scores documents using Term Frequency (how often a word appears in the doc) and Inverse Document Frequency (how rare the word is across the corpus). Purely statistical — no model involved.

```
Query: "automobile safety"
Doc:   "car crash fatality statistics"

BM25 score: 0  ← not a single word overlaps. Complete miss.
```

BM25 cannot handle synonyms. It only matches words that literally appear in both query and document.

**SPLADE-style — learned sparse with term expansion**

Uses a transformer encoder to produce sparse vectors. The key innovation: it activates vocabulary terms that are **not in the input text**. The model learned during training that when you write "FastAPI", documents containing "python", "api", "framework" tend to be relevant — so it fires those terms too. This is called **term expansion**.

```
Input: "FastAPI tutorial"

BM25:
  {"fastapi": 1, "tutorial": 1}          ← only literal words

SPLADE-style:
  {"fastapi": 0.87, "tutorial": 0.65,    ← actual words
   "python":  0.43, "api": 0.38,         ← NOT in text, inferred
   "web":     0.31, "framework": 0.28}   ← NOT in text, inferred
```

Result: still sparse (efficient for inverted index lookup), but catches synonyms and related terms that BM25 would miss entirely.

```
Query: "automobile safety"

SPLADE query vector:
  {"automobile": 0.9, "safety": 0.8, "car": 0.7, "vehicle": 0.6,
   "accident": 0.5, "crash": 0.4, ...}

Doc vector:
  {"car": 0.9, "crash": 0.8, "fatality": 0.7, "safety": 0.5, ...}

Dot product: high score  ← match on "car", "crash", "safety", "accident"
```

> **BM25 asks "does this word appear?" SPLADE asks "does this concept appear?" — and answers in the same sparse format.**

### Models That Produce Learned Sparse Vectors

Two sub-types:
- **Weighting + expansion** — activates terms not in the text (full SPLADE capability)
- **Weighting only** — smarter weights for existing terms, no new terms added

| Model | Expansion? | Dense too? | Where it runs |
|---|---|---|---|
| `naver/splade-v3` | Yes | No | Local / HuggingFace |
| `naver/splade-cocondenser-ensemble-distil` | Yes | No | Local / HuggingFace |
| `BAAI/bge-m3` | Yes | Yes | Local / HuggingFace |
| Elastic ELSER (`.elser_model_2`) | Yes | No | Elasticsearch only |
| `amazon/neural-sparse-encoding-v1` | Yes | No | OpenSearch / HuggingFace |
| `castorini/unicoil-msmarco-passage` | No | No | Local / HuggingFace |
| BM25 (baseline) | No | No | Any search engine |

BGE-M3 is the only model that produces **both** dense and SPLADE-style sparse in a single forward pass — one inference call, two retrieval signals.

**Practical guide:**
- Building on Elasticsearch → use ELSER
- Building on OpenSearch / AWS → use `amazon/neural-sparse-encoding-v1`
- Want portable sparse, any vector DB → use `naver/splade-v3`
- Need dense + sparse from one model → `bge-m3` is the only option

---

## 1. OpenAI `text-embedding-3-small` / `text-embedding-3-large`

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

vector = response.data[0].embedding
print("Embedding dim:", len(vector))
```

### 1.3 Batch encode documents

```python
texts = ["First document", "Second document", "Third document"]

resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
embeddings = [item.embedding for item in resp.data]
```

### 1.4 Matryoshka dimension reduction

Both `text-embedding-3-*` models support requesting fewer dimensions with minimal quality loss:

```python
response = client.embeddings.create(
    model="text-embedding-3-large",
    input="Your text string goes here",
    dimensions=256,  # compress from 3072 → 256; valid range: 1 to model max
)

vector = response.data[0].embedding
print("Embedding dim:", len(vector))  # 256
```

---

## 2. BGE / BGE‑M3 via FlagEmbedding

### 2.1 Install

```bash
pip install -U FlagEmbedding
```

### 2.2 Dense + sparse embeddings with `BGEM3FlagModel`

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)  # use_fp16=True for GPU

texts = ["What is BGE M3?", "Definition of BM25"]
embeddings = model.encode(texts)

# Dense: embeddings["dense_vecs"]  -> shape (N, D)  — use for vector DB
# Sparse: embeddings["sparse_vecs"] -> dict — use for hybrid BM25-style retrieval
print("Dense shape:", embeddings["dense_vecs"].shape)
```

---

## 3. NV‑Embed‑v1 (NVIDIA)

```bash
pip install -U sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nvidia/NV-Embed-v1")
embeddings = model.encode(["doc one", "doc two"], normalize_embeddings=True)
print("Shape:", embeddings.shape)
```

---

## 4. Generic E5 / GTE / Mixedbread via SentenceTransformers

### 4.1 GTE / Mixedbread (symmetric — no prefix needed)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
embeddings = model.encode(["text one", "text two"], normalize_embeddings=True)
```

### 4.2 E5 models — MANDATORY query/document prefixes

E5 models are **asymmetric**: queries must be prefixed with `"query: "` and documents with `"passage: "`. Without this, retrieval quality degrades significantly.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("intfloat/e5-base-v2")

documents = [
    "passage: FastAPI is a modern web framework for building APIs with Python.",
    "passage: SQLAlchemy is a Python SQL toolkit and ORM.",
]
query = "query: What is FastAPI used for?"

doc_embeddings = model.encode(documents, normalize_embeddings=True)
query_embedding = model.encode(query, normalize_embeddings=True)

scores = np.dot(query_embedding, doc_embeddings.T)
print("Similarity scores:", scores)
```

### 4.3 BGE models via SentenceTransformers

BGE models benefit from a query instruction prefix:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

documents = ["RAG combines retrieval with generation.", "Vector DBs store embeddings."]
query = "Represent this sentence for searching relevant passages: How does RAG work?"

doc_embeddings = model.encode(documents, normalize_embeddings=True)
query_embedding = model.encode(query, normalize_embeddings=True)

scores = np.dot(query_embedding, doc_embeddings.T)
```

---

## 5. CLIP — Image + Text Embeddings (Multimodal)

CLIP maps both images and text into a **shared embedding space**, enabling cross-modal retrieval (search images with text queries, or vice versa).

```bash
pip install -U sentence-transformers pillow
```

```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image

model = SentenceTransformer("clip-ViT-B-32")  # or "clip-ViT-L-14" for higher quality

# Encode images
img1_emb = model.encode(Image.open("dog.jpg"))
img2_emb = model.encode(Image.open("cat.jpg"))

# Encode text queries
text_emb = model.encode([
    "a dog playing in the snow",
    "a cat sitting on a table",
    "a city skyline at night",
])

# Cross-modal similarity: text query → image retrieval
scores = util.cos_sim(text_emb, [img1_emb, img2_emb])
print("Text→Image similarity scores:", scores)
```

### Batch-encode an image corpus for indexing

```python
import glob
from PIL import Image

image_paths = glob.glob("images/*.jpg")
images = [Image.open(p) for p in image_paths]

# Encode all images for storage in a vector DB
img_embeddings = model.encode(
    images,
    batch_size=32,
    convert_to_tensor=True,
    show_progress_bar=True,
    normalize_embeddings=True,
)
print("Indexed", len(img_embeddings), "images")
```

> **Use case in RAG:** Index product photos, document screenshots, medical scans, or diagrams. At query time, encode the user's text query with the same CLIP model and retrieve the most similar images by cosine similarity.

---

## 6. Google – Vertex AI Text Embeddings (`text-embedding-005`)

```bash
pip install -U google-cloud-aiplatform
```

```python
import vertexai
from vertexai.language_models import TextEmbeddingModel

vertexai.init(project="your-gcp-project-id", location="us-central1")
model = TextEmbeddingModel.from_pretrained("text-embedding-005")

texts = ["What is cloud computing?", "How to configure autoscaling on GKE?"]
embeddings = model.get_embeddings(texts)

for text, emb in zip(texts, embeddings):
    print(text[:40], "| dim:", len(emb.values))
```

---

## 7. Google – Gemini API Embeddings

### 7.1 `gemini-embedding-001` (text only, GA)

```bash
pip install -U google-genai
```

```python
import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=["What is the meaning of life?", "How does the brain work?"],
    config=types.EmbedContentConfig(output_dimensionality=768),  # optional; max 3072
)

embeddings = [e.values for e in result.embeddings]
print("Dim:", len(embeddings[0]))
```

### 7.2 `gemini-embedding-2-preview` (multimodal — text, image, video, audio, documents)

Released March 2026. The **first natively multimodal embedding model from Google** — maps all modalities into a single shared vector space. Available via Vertex AI and Gemini API (public preview). Not available for local download.

```bash
pip install -U google-genai pillow
```

```python
import os
import base64
from pathlib import Path
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# --- Text embedding ---
text_result = client.models.embed_content(
    model="gemini-embedding-2-preview",
    contents=["Explain transformer architecture."],
)
text_vec = text_result.embeddings[0].values
print("Text embedding dim:", len(text_vec))

# --- Image embedding ---
image_bytes = Path("diagram.png").read_bytes()
image_b64 = base64.b64encode(image_bytes).decode()

image_result = client.models.embed_content(
    model="gemini-embedding-2-preview",
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    ],
)
image_vec = image_result.embeddings[0].values
print("Image embedding dim:", len(image_vec))

# --- Cross-modal similarity (text query → image retrieval) ---
import numpy as np

query_result = client.models.embed_content(
    model="gemini-embedding-2-preview",
    contents=["architecture diagram showing attention mechanism"],
)
query_vec = np.array(query_result.embeddings[0].values)
doc_vec = np.array(image_vec)

score = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
print("Cross-modal cosine similarity:", score)
```

> **Key insight:** Because text and images share the same vector space, you can index a mixed corpus of PDFs, images, and text documents, then retrieve all of them with a single text query — without separate pipelines per modality.

---

## 8. AWS Bedrock – Titan Text Embeddings & Cohere Embed v3

### 8.1 Titan Text Embeddings v2

```bash
pip install boto3
```

```python
import json, boto3, os

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

response = bedrock.invoke_model(
    modelId="amazon.titan-embed-text-v2:0",
    body=json.dumps({"inputText": "Recommend books similar to Inception."}),
    contentType="application/json",
    accept="application/json",
)

vector = json.loads(response["body"].read())["embedding"]
print("Dim:", len(vector))
```

### 8.2 Cohere Embed v3 (direct API)

```bash
pip install -U cohere
```

```python
import cohere, numpy as np, os

co = cohere.Client(os.environ.get("COHERE_API_KEY"))

docs = [
    "The capital of France is Paris",
    "PyTorch is a machine learning framework.",
]
doc_emb = np.asarray(co.embed(docs, input_type="search_document", model="embed-english-v3.0").embeddings)

query = "What is PyTorch"
query_emb = np.asarray(co.embed([query], input_type="search_query", model="embed-english-v3.0").embeddings)

scores = np.dot(query_emb, doc_emb.T)
for idx in np.argsort(-scores[0]):
    print(f"{scores[0][idx]:.3f} | {docs[idx]}")
```

---

## 9. Voyage AI

### 9.1 Install

```bash
pip install -U voyageai
```

### 9.2 Text retrieval

```python
import os, voyageai

vo = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))

doc_result = vo.embed(
    ["RAG stands for Retrieval Augmented Generation.", "ChromaDB is a vector database."],
    model="voyage-3",
    input_type="document",
)
query_result = vo.embed(["How does RAG work?"], model="voyage-3", input_type="query")

print("Doc dim:", len(doc_result.embeddings[0]))
```

### 9.3 Model variants

| Model | Best for | Dims |
|---|---|---|
| `voyage-3` | General-purpose RAG | 1024 |
| `voyage-3-lite` | High-throughput / cost-sensitive | 512 |
| `voyage-code-3` | Code retrieval | 1024 |
| `voyage-finance-2` | Financial documents | 1024 |
| `voyage-law-2` | Legal documents | 1024 |
| `voyage-multimodal-3` | Text + image (interleaved) | 1024 |

> Always pass `input_type="document"` for indexing and `input_type="query"` for search queries.

---

## 10. Jina AI (`jina-embeddings-v3`)

```bash
pip install -U requests
```

```python
import os, requests

url = "https://api.jina.ai/v1/embeddings"
headers = {
    "Authorization": f"Bearer {os.environ.get('JINA_API_KEY')}",
    "Content-Type": "application/json",
}

payload = {
    "model": "jina-embeddings-v3",
    "task": "retrieval.passage",   # use "retrieval.query" for search queries
    "input": ["RAG combines retrieval with generation.", "Vector DBs store embeddings."],
    "dimensions": 1024,
}

data = requests.post(url, headers=headers, json=payload).json()
embeddings = [item["embedding"] for item in data["data"]]
print("Dim:", len(embeddings[0]))
```

| `task` value | Use when |
|---|---|
| `retrieval.passage` | Encoding documents for indexing |
| `retrieval.query` | Encoding search queries |
| `text-matching` | Semantic similarity / paraphrase detection |
| `classification` | Text classification |
| `separation` | Clustering |

---

## 11. Perplexity AI — `pplx-embed` (Feb 2026)

Released February 26, 2026. Two model families, each at 0.6B and 4B parameter scales. MIT licensed, available on HuggingFace.

**`pplx-embed-v1`** — standard dense retrieval. Beats `gemini-embedding-001` and Qwen3-Embedding-4B on MTEB Multilingual v2.

**`pplx-embed-context-v1`** — contextual retrieval: each chunk embedding is informed by its surrounding document context (late chunking built into training). Sets SOTA on ConTEB benchmark (81.96% nDCG@10).

Key design decisions:
- **Bidirectional attention** via diffusion-based continued pretraining from Qwen3 — unlike most LLM-based embedders that use causal attention
- **Native INT8/binary quantization** — 4x/32x storage reduction vs FP32, trained with quantization-aware training (no post-hoc compression loss)
- **No instruction prefixes required** — no `input_type` mismatch risk at index vs query time

### 11.1 Install

```bash
pip install -U sentence-transformers
```

### 11.2 Standard dense retrieval with `pplx-embed-v1`

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 0.6B for low-latency; 4B for maximum quality
model = SentenceTransformer("perplexity-ai/pplx-embed-v1-0.6B")

documents = [
    "RAG combines retrieval with generation to ground LLM responses in facts.",
    "ChromaDB is an open-source vector database for storing dense embeddings.",
    "BM25 is a sparse retrieval algorithm based on term frequency.",
]
query = "How does RAG work?"

doc_embeddings = model.encode(documents, normalize_embeddings=True)
query_embedding = model.encode(query, normalize_embeddings=True)

scores = np.dot(query_embedding, doc_embeddings.T)
for i in np.argsort(-scores):
    print(f"{scores[i]:.3f} | {documents[i][:60]}")
```

### 11.3 Contextual retrieval with `pplx-embed-context-v1`

Use this when your documents are long and chunks lose meaning without their surrounding context (e.g., a table row that only makes sense with the table header, or a paragraph that references earlier definitions).

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("perplexity-ai/pplx-embed-context-v1-0.6B")

# The model encodes each chunk with awareness of the full document context
# Pass the full document alongside each chunk during indexing
documents = [
    "Section 3.2: The results show a 28% improvement over baseline.",
    "Appendix A: Model hyperparameters used in all experiments.",
]
query = "What was the improvement over baseline?"

doc_embeddings = model.encode(documents, normalize_embeddings=True)
query_embedding = model.encode(query, normalize_embeddings=True)

scores = np.dot(query_embedding, doc_embeddings.T)
print("Scores:", scores)
```

### 11.4 Model comparison

| Model | Params | Dims | Best for |
|---|---|---|---|
| `pplx-embed-v1-0.6B` | 0.6B | 1024 | Low-latency, high-throughput indexing |
| `pplx-embed-v1-4B` | 4B | 2560 | Maximum retrieval quality |
| `pplx-embed-context-v1-0.6B` | 0.6B | 1024 | Contextual chunks, fast |
| `pplx-embed-context-v1-4B` | 4B | 2560 | Contextual chunks, best quality |

> **INT8/binary quantization:** All models produce INT8 embeddings natively. The 4B binary variant reduces storage 32x with under 1.6% quality drop — viable for web-scale indexing.

---

## 12. How to Plug These Embeddings into Your RAG Stack

1. **Choose a model based on modality first:**
   - Text-only → any model in the text table above
   - Image + text → CLIP, `gemini-embedding-2-preview`, or `voyage-multimodal-3`
   - All modalities (text, image, video, audio, docs) → `gemini-embedding-2-preview`
   - Code → `voyage-code-3`
   - Contextual/long documents → `pplx-embed-context-v1`

2. **Then optimise for cost and quality:**
   - Best open-source text quality → `pplx-embed-v1-4B` or `bge-large-en-v1.5`
   - Best cloud text quality → `text-embedding-3-large` or `voyage-3`
   - Lowest cost local → `all-MiniLM-L6-v2` (baseline), `pplx-embed-v1-0.6B`

3. **Keep query and document encodings consistent:**
   - Same model, same config at both index time and query time
   - Correct `input_type` / `task` / prefix for asymmetric models (E5, BGE, Cohere, Voyage, Jina)

4. **Always normalize for cosine similarity:**
   - Cloud APIs (OpenAI, Voyage, Cohere, Jina, Perplexity API): normalize automatically
   - Local SentenceTransformers: always pass `normalize_embeddings=True`

---

## References

1. [Vector embeddings | OpenAI API](https://developers.openai.com/api/docs/guides/embeddings/)
2. [NV-Embed-v1](https://paddlenlp.readthedocs.io/zh/latest/website/nvidia/NV-Embed-v1/index.html)
3. [Pretrained Models — Sentence Transformers](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
4. [BAAI/bge-m3 - Hugging Face](https://huggingface.co/BAAI/bge-m3)
5. [New embedding models and API updates - OpenAI](https://openai.com/index/new-embedding-models-and-api-updates/)
6. [FlagEmbedding - PyPI](https://pypi.org/project/FlagEmbedding/1.2.7/)
7. [Text embeddings API | Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api)
8. [Get text embeddings | Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings)
9. [Google generative AI integration - LangChain](https://docs.langchain.com/oss/python/integrations/text_embedding/google_generative_ai)
10. [Invoke Amazon Titan Text Embeddings - AWS](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModelWithResponseStream_TitanTextEmbeddings_section.html)
11. [Cohere Embed v3 - AWS Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html)
12. [SentenceTransformers Documentation](https://sbert.net)
13. [Training with Prompts — Sentence Transformers](https://sbert.net/examples/sentence_transformer/training/prompts/README.html)
14. [Embeddings | Gemini API](https://ai.google.dev/api/embeddings)
15. [Voyage AI Embeddings Documentation](https://docs.voyageai.com/docs/embeddings)
16. [Jina Embeddings v3 - Hugging Face](https://huggingface.co/jinaai/jina-embeddings-v3)
17. [pplx-embed: State-of-the-Art Embedding Models - Perplexity Research](https://research.perplexity.ai/articles/pplx-embed-state-of-the-art-embedding-models-for-web-scale-retrieval)
18. [pplx-embed HuggingFace Collection](https://huggingface.co/collections/perplexity-ai/pplx-embed)
19. [Gemini Embedding 2 - Google Blog](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/)
20. [CLIP Image Search - Sentence Transformers](https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/image-search/README.md)
