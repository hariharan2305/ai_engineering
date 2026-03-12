# Document Ingestion & Parsing – Code Cheat‑Sheet (2026)

This cheat‑sheet gives **minimal, copy‑pasteable Python snippets** for each parser mentioned in the *Document Ingestion & Parsing* section:

- Unstructured (local partitioning + hosted API)
- LlamaParse (Llama Cloud)
- Docling
- Azure AI Document Intelligence
- AWS Textract
- Google Cloud Document AI (Layout Parser for RAG)

> All snippets are derived from the latest official docs as of early 2026.[^1][^2][^3][^4][^5][^6]

Set your API keys via environment variables (e.g. `UNSTRUCTURED_API_KEY`, `LLAMA_CLOUD_API_KEY`, `DOCUMENTINTELLIGENCE_API_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`).

***

## 1. Unstructured (Local Python Library)

### 1.1 Install

```bash
pip install "unstructured[local-inference]"  # see docs for extras based on OS
```

### 1.2 Parse a PDF with `partition_pdf`

```python
from unstructured.partition.pdf import partition_pdf

# Basic parse – returns a list of Element objects
elements = partition_pdf("example-docs/pdf/layout-parser-paper-fast.pdf")

for el in elements[:5]:
    print(type(el).__name__, "::", el.text[:80])
```

This is the canonical example from the Unstructured partitioning docs.[^1]

### 1.3 Parse a DOCX with `partition_docx`

```python
from unstructured.partition.docx import partition_docx

elements = partition_docx(filename="mydoc.docx")

for el in elements:
    print(el.category, "::", el.text)
```

The same pattern works for emails (`partition_email`), text files (`partition_text`), etc.[^1]

### 1.4 When to use

- You want **local parsing** (no external API) over many formats.
- You plan to **chunk and embed the resulting Elements** in your own RAG pipeline.

***

## 2. Unstructured Hosted API (Semantic Parsing as a Service)

If you don’t want to run heavy dependencies locally, you can call Unstructured’s hosted API.

### 2.1 Install SDK

```bash
pip install unstructured-client
```

### 2.2 Call `partition_via_api` (simple case)

```python
import os
from unstructured.partition.api import partition_via_api

filename = "example-docs/eml/fake-email.eml"

api_key = os.environ["UNSTRUCTURED_API_KEY"]

# Basic API call – Unstructured hosts the heavy models
elements = partition_via_api(
    filename=filename,
    api_key=api_key,
    content_type="message/rfc822",
)

for el in elements:
    print(el.category, "::", el.text[:80])
```

This mirrors the official `partition_via_api` examples in the Unstructured docs.[^7][^1]

### 2.3 Advanced: hi‑res PDF with table structure

```python
from unstructured.partition.api import partition_via_api

pdf_path = "example-docs/DA-1p.pdf"

elements = partition_via_api(
    filename=pdf_path,
    api_key=os.environ["UNSTRUCTURED_API_KEY"],
    strategy="hi_res",
    pdf_infer_table_structure="true",
)
```

Use this when you want better table extraction for RAG over financial / technical docs.[^7]

***

## 3. LlamaParse (Llama Cloud) via `llama-parse`

LlamaParse is a GenAI‑native parser that returns **Markdown or text** optimized for RAG; it is being migrated into the `llama-cloud` package but is supported until May 1, 2026.[^5]

### 3.1 Install

```bash
pip install -U llama-index  # core LlamaIndex
pip install llama-parse     # LlamaParse client (deprecated after May 1, 2026)
```

### 3.2 Basic Python usage

```python
import os
from llama_parse import LlamaParse

# or set LLAMA_CLOUD_API_KEY in your environment
parser = LlamaParse(
    api_key=os.environ["LLAMA_CLOUD_API_KEY"],
    result_type="markdown",   # or "text"
    num_workers=4,
    verbose=True,
    language="en",
)

# Parse a single PDF into a list of LlamaIndex Document objects
documents = parser.load_data("./my_file.pdf")

print(len(documents))
print(documents.text[:500])
```

This is adapted directly from the PyPI quickstart snippet.[^5]

### 3.3 Integrate with LlamaIndex `SimpleDirectoryReader`

```python
import os
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

parser = LlamaParse(
    api_key=os.environ["LLAMA_CLOUD_API_KEY"],
    result_type="markdown",
    verbose=True,
)

file_extractor = {".pdf": parser}

documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()

print("Loaded", len(documents), "docs from ./data")
```

This is the recommended integration path for RAG with LlamaIndex.[^5]

> **Migration note:** New projects can also use `llama-cloud` ≥ 1.0 as recommended on the same PyPI page; the parsing API is very similar.[^5]

***

## 4. Docling (IBM / LF AI)

Docling converts many formats (PDF, DOCX, PPTX, XLSX, HTML, LaTeX, audio, etc.) to a unified document representation and can export to Markdown, JSON, and more.[^4][^8]

### 4.1 Install

```bash
pip install docling
```

### 4.2 Quickstart: convert to Markdown

```python
from docling.document_converter import DocumentConverter

# "source" can be a local file path or a URL
source = "https://arxiv.org/pdf/2408.09869"  # example from official quickstart

converter = DocumentConverter()
result = converter.convert(source)

doc = result.document
markdown = doc.export_to_markdown()

print(markdown[:500])
```

This is directly from the Docling Quickstart.[^4]

### 4.3 How to use in RAG

- Use `export_to_markdown()` or structured JSON export.
- Chunk the resulting text/sections and embed them into your vector DB.

***

## 5. Azure AI Document Intelligence (Layout Model)

Azure’s **Document Intelligence** service (v4.0) provides the `prebuilt-layout` model that extracts text, tables, selection marks, and paragraphs—ideal as the **first stage** of a RAG pipeline.[^2][^9]

### 5.1 Install SDK

```bash
pip install azure-ai-documentintelligence
```

### 5.2 Minimal Python example (layout extraction)

```python
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult

endpoint = os.environ["DOCUMENTINTELLIGENCE_ENDPOINT"]  # e.g. https://<resource-name>.cognitiveservices.azure.com
key = os.environ["DOCUMENTINTELLIGENCE_API_KEY"]

client = DocumentIntelligenceClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key),
)

pdf_path = "./sample.pdf"

with open(pdf_path, "rb") as f:
    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",  # layout model for text + tables
        body=f,
    )

result: AnalyzeResult = poller.result()

for page in result.pages:
    print("Page", page.page_number, "has", len(page.lines), "lines")

print("First paragraph:")
if result.paragraphs:
    print(result.paragraphs.content)
```

This follows the official layout extraction snippet in the v4 Python client docs.[^2]

### 5.3 RAG usage

- Turn `result.paragraphs` and table cells into chunks.
- Store paragraph/table text, bounding boxes, and page numbers as metadata in your index.

***

## 6. AWS Textract (OCR + Layout for RAG)

Textract detects text, forms, and tables from PDFs/images and is often used upstream of RAG on AWS.[^10][^11][^12]

### 6.1 Install Boto3

```bash
pip install boto3
```

### 6.2 Minimal `detect_document_text` example (local file)

```python
import boto3

textract = boto3.client("textract", region_name="us-east-1")

with open("./sample.png", "rb") as f:
    image_bytes = f.read()

response = textract.detect_document_text(
    Document={"Bytes": image_bytes}
)

lines = [
    block["Text"]
    for block in response["Blocks"]
    if block["BlockType"] == "LINE"
]

print("First 5 lines:")
for line in lines[:5]:
    print("-", line)
```

The call signature is taken from the latest Boto3/Textract docs.[^13][^3]

### 6.3 RAG usage

- Concatenate lines into paragraphs or use bounding boxes to reconstruct layout.
- For forms/tables, use Textract’s form/table APIs from the official examples and store key–value pairs or table cells into your index.[^11]

***

## 7. Google Cloud Document AI – Layout Parser for RAG

Google’s **Document AI Layout Parser** is explicitly designed to produce RAG‑ready chunks, including tables and image annotations.[^6]

### 7.1 Install client

```bash
pip install --upgrade google-cloud-documentai
```

Authenticate with a service account JSON file:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
```

### 7.2 Python: process a GCS‑stored PDF with Layout Parser

```python
from google.cloud import documentai_v1 as documentai

# Set these from your environment or config
PROJECT_ID = "your-project-id"
LOCATION = "us"  # or "eu", etc.
PROCESSOR_ID = "your-processor-id"  # Layout parser processor you created
GCS_URI = "gs://your-bucket/path/to/file.pdf"
MIME_TYPE = "application/pdf"

# Instantiate client (regional endpoint is recommended)
client = documentai.DocumentProcessorServiceClient()

processor_version_id = "pretrained-layout-parser-v1.5-2025-08-25"
name = client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID, processor_version_id)

gcs_document = documentai.GcsDocument(
    gcs_uri=GCS_URI,
    mime_type=MIME_TYPE,
)

process_options = documentai.ProcessOptions(
    layout_config=documentai.ProcessOptions.LayoutConfig(
        enable_table_annotation=True,
        enable_image_annotation=True,
        chunking_config=documentai.ProcessOptions.LayoutConfig.ChunkingConfig(
            chunk_size=1024,
            include_ancestor_headings=True,
        ),
    ),
)

request = documentai.ProcessRequest(
    name=name,
    gcs_document=gcs_document,
    process_options=process_options,
)

result = client.process_document(request=request)
doc = result.document

print("Document processing complete. RAG‑ready chunks:")
for i, chunk in enumerate(doc.chunked_document.chunks[:5]):
    print(f"\n--- Chunk {i} ---")
    print(chunk.content[:300])
```

This closely follows the **Layout Parser Quickstart** Python sample from Document AI docs.[^6]

### 7.3 RAG usage

- Use `chunked_document.chunks` as your **pre‑chunked units**.
- Each chunk already carries context such as page headers, footers, and image annotations that can be embedded and stored for retrieval.

***

## How to Plug These Parsers into Your RAG Stack

1. **Choose a parser** based on your infra:
   - Local / OSS → Unstructured, Docling.
   - Cloud‑managed → Azure Document Intelligence, Textract, Document AI.
   - LlamaIndex‑centric → LlamaParse.
2. **Run the parser** to get structured text + metadata.
3. Convert outputs to your framework’s `Document`/`Node` objects.
4. Feed them into your **chunking + embedding** stage.

Use this file as a ready‑to‑paste reference when your agent or you need to wire up a new ingestion backend.

---

## References

1. [Partitioning - Unstructured](https://docs.unstructured.io/open-source/core-functionality/partitioning) - The easiest way to partition documents in unstructured is to use the partition function. If you call...

2. [Azure AI Document Intelligence client library for Python](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-documentintelligence-readme?view=azure-python) - Examples. The following section provides several code snippets covering some of the most common Docu...

3. [detect_document_text - Boto3 1.42.59 documentation](https://docs.aws.amazon.com/boto3/latest/reference/services/textract/client/detect_document_text.html) - If you use the AWS CLI to call Amazon Textract operations, you can't pass image bytes. The document ...

4. [Quickstart - Docling - GitHub Pages](https://docling-project.github.io/docling/getting_started/quickstart/) - In Docling, working with documents is as simple as: converting your source file to a Docling documen...

5. [llama-parse - PyPI](https://pypi.org/project/llama-parse/) - LlamaParse is a GenAI-native document parser that can parse complex document data for any downstream...

6. [Process documents with Gemini layout parser | Document AI](https://docs.cloud.google.com/document-ai/docs/layout-parse-chunk) - Document OCR: It can parse text and layout elements like heading, header, footer, table structure an...

7. [Accessing Unstructured API](https://unstructured.readthedocs.io/en/main/apis/usage_methods.html)

8. [Documentation - Docling - GitHub Pages](https://docling-project.github.io/docling/)

9. [Azure AI Document Intelligence: Parsing PDF text & table data - Elastic](https://www.elastic.co/search-labs/blog/azure-ai-document-intelligence-parse-pdf-text-tables) - Azure AI Document Intelligence is a powerful tool for extracting structured data from PDFs. It can b...

10. [Detecting Document Text with Amazon Textract](https://docs.aws.amazon.com/textract/latest/dg/detecting-document-text.html) - Describes how to detect document text with Amazon Textract.

11. [Amazon Textract examples using SDK for Python (Boto3)](https://docs.aws.amazon.com/code-library/latest/ug/python_3_textract_code_examples.html) - Shows how to use the AWS SDK for Python (Boto3) with Amazon Textract to detect text, form, and table...

12. [Better RAG accuracy and consistency with Amazon Textract](https://community.aws/content/2njwVmseGl0sxomMvrq65PzHo9x/better-rag-accuracy-and-consistency-with-amazon-textract) - Crafting a Retrieval-Augmented Generation (RAG) pipeline may seem straightforward, but optimizing it...

13. [Detect_document_text](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/textract/client/detect_document_text.html)

