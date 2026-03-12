# Generation Models (LLMs) – Code Cheat‑Sheet (2026)

This cheat‑sheet gives **implementation‑level examples** for all LLM providers and roles in your component guide, plus **OpenRouter** and **LiteLLM**:

- OpenAI (GPT‑4.1, o‑series, GPT‑4 Turbo) – direct + streaming
- Anthropic Claude 3.5 (Sonnet/Opus) – direct + streaming
- Google Gemini 1.5 / 2.x via **Gemini API** – direct + streaming
- AWS Bedrock (Claude, Llama, Mistral, Titan) – chat with `bedrock-runtime`
- Azure OpenAI (GPT‑4.x, GPT‑4o) – chat + streaming
- Open‑source models (Llama, Mistral, Qwen, Phi) via **TGI** / **vLLM** (OpenAI‑compatible)
- OpenRouter – unified multi‑provider API, including streaming
- LiteLLM – OpenAI‑compatible proxy over many providers

All APIs follow **current docs as of early 2026**.[^1][^2][^3][^4][^5][^6][^7][^8][^9][^10][^11][^12][^13][^14]

> Every example also shows **where to insert retrieved RAG context** into the prompt.

***

## 1. OpenAI – GPT‑4.1 / o‑series / GPT‑4 Turbo

OpenAI is widely used for **general reasoning, tool use, and RAG**.

### 1.1 Install and init

```bash
pip install openai
```

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

### 1.2 Basic chat completion (non‑streaming) with RAG context

```python
retrieved_chunks = [
    "Doc1: ...",
    "Doc2: ...",
]

context_block = "\n\n".join(retrieved_chunks)

messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant. Answer using ONLY the provided context. "
            "Cite sources as [Doc#] where possible."
        ),
    },
    {
        "role": "user",
        "content": f"Context:\n{context_block}\n\nQuestion: What is the refund policy?",
    },
]

response = client.chat.completions.create(
    model="gpt-4.1-mini",  # or gpt-4.1, gpt-4.1-preview, gpt-4o-mini, etc.
    messages=messages,
    temperature=0,
)

answer = response.choices.message.content
print(answer)
```

Pattern based on the Chat Completions examples in the latest OpenAI cookbook.[^1]

### 1.3 Streaming chat completion

```python
messages = [
    {"role": "user", "content": "What is 1+1? Answer in one word."},
]

stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=0,
    stream=True,
)

for chunk in stream:
    delta = chunk.choices.delta
    if delta.content:
        print(delta.content, end="", flush=True)
print()  # final newline
```

Streaming pattern matches the OpenAI "How to stream completions" cookbook.[^1]

***

## 2. Anthropic – Claude 3.5 Sonnet / Opus

Claude is used for **long‑context RAG, safety, and tool‑augmented reasoning**.[^3][^4]

### 2.1 Install and init

```bash
pip install anthropic
```

```python
import os
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
```

### 2.2 Basic messages.create with RAG context

```python
retrieved_chunks = ["Doc1: ...", "Doc2: ..."]
context = "\n\n".join(retrieved_chunks)

message = client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=1024,
    temperature=0,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a helpful assistant. Answer only from this context, "
                        "and say 'I don't know' if it is not covered.\n\n"
                        f"Context:\n{context}\n\nQuestion: What is the refund policy?"
                    ),
                }
            ],
        }
    ],
)

print(message.content.text)
```

Matches the Claude `messages.create` API.[^3]

### 2.3 Streaming responses (async text stream)

From the official Python streaming helpers:[^15][^4]

```python
import asyncio
from anthropic import AsyncAnthropic

client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

async def main() -> None:
    async with client.messages.stream(
        model="claude-3-5-sonnet-latest",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Say hello there!"},
        ],
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)
        print()

asyncio.run(main())
```

Use the same structure but replace the prompt with your **RAG‑formatted context + question**.

***

## 3. Google – Gemini API (Gemini 1.5 / 2.x)

Gemini API is used both **directly** and inside Vertex AI RAG stacks.

### 3.1 Install and init

```bash
pip install -U google-genai
```

```python
import os
from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
```

### 3.2 Basic `generate_content` with RAG context

From Gemini generate‑content docs:[^2]

```python
retrieved_chunks = ["Doc1: ...", "Doc2: ..."]
context = "\n\n".join(retrieved_chunks)

prompt = (
    "You are a helpful assistant. Answer based only on the given context. "
    "If the answer is not present, say you don't know.\n\n"
    f"Context:\n{context}\n\nQuestion: Explain our refund policy for EU customers."
)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt,
)

print(response.text)
```

### 3.3 Streaming with `generate_content_stream`

From `models.streamGenerateContent` examples:[^16][^2]

```python
stream = client.models.generate_content_stream(
    model="gemini-2.0-flash",
    contents="Write a story about a magic backpack.",
)

for chunk in stream:
    if chunk.text:
        print(chunk.text, end="", flush=True)
print()
```

Replace `contents` with your **context + question** when using Gemini for RAG.

***

## 4. AWS Bedrock – Claude, Llama, Mistral, Titan

Bedrock gives you **multiple foundation models** via one API plus **native RAG** via Knowledge Bases and Agents.[^6][^13]

### 4.1 Install Boto3 and init Bedrock Runtime client

```bash
pip install boto3
```

```python
import boto3

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
```

### 4.2 `converse` API (recommended chat interface)

From the Bedrock Runtime code examples (simplified):[^6]

```python
model_id = "anthropic.claude-3-5-sonnet-20241022-v1:0"  # example, check AWS docs for latest

retrieved_chunks = ["Doc1: ...", "Doc2: ..."]
context = "\n\n".join(retrieved_chunks)

conversation = [
    {
        "role": "user",
        "content": [
            {
                "text": (
                    "You are a helpful assistant. Answer using ONLY the context. "
                    "Say 'I don't know' if not covered.\n\n"
                    f"Context:\n{context}\n\nQuestion: What is the refund policy?"
                )
            }
        ],
    }
]

response = bedrock.converse(
    modelId=model_id,
    messages=conversation,
    inferenceConfig={
        "maxTokens": 512,
        "temperature": 0.2,
        "topP": 0.9,
    },
)

assistant_msg = response["output"]["message"]["content"]["text"]
print(assistant_msg)
```

Use different `modelId` values for Mistral, Llama, and Titan chat models as needed.[^13][^6]

> For **RAG with Knowledge Bases**, combine this with the `retrieve` step shown in your retrieval cheat‑sheet and feed retrieved text into the prompt.

***

## 5. Azure OpenAI – GPT‑4.x / GPT‑4o

Azure OpenAI wraps GPT‑4.x models with Azure identity, VNET, and regional controls.[^11][^14]

### 5.1 Install and init

```bash
pip install openai
```

```python
import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-01-preview"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),  # e.g. https://<resource>.openai.azure.com
)

deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
```

### 5.2 Chat completion with RAG context

```python
retrieved_chunks = ["Doc1: ...", "Doc2: ..."]
context = "\n\n".join(retrieved_chunks)

messages = [
    {"role": "system", "content": "You are a helpful assistant that only uses the provided context."},
    {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: What is the refund policy?",
    },
]

response = client.chat.completions.create(
    model=deployment,
    messages=messages,
    temperature=0,
)

print(response.choices.message.content)
```

### 5.3 Streaming example

Streaming is similar to OpenAI ChatCompletions, as shown in Azure chat examples and blogs.[^17][^11]

```python
stream = client.chat.completions.create(
    model=deployment,
    messages=[{"role": "user", "content": "Tell me a joke."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices.delta
    if delta.content:
        print(delta.content, end="", flush=True)
print()
```

Use this pattern whenever you want **token‑by‑token streaming** in an Azure‑hosted RAG app.

***

## 6. Open‑Source Models via TGI / vLLM (OpenAI‑Compatible)

Open‑source models like **Llama, Mistral, Qwen, Phi** are often exposed via **OpenAI‑compatible HTTP servers**, using:

- Hugging Face **Text Generation Inference (TGI)**
- **vLLM** HTTP server
- **Nvidia NIM** (often OpenAI‑compatible)

### 6.1 Example: calling a TGI/vLLM server via OpenAI client

If your TGI/vLLM server exposes `/v1/chat/completions`, you can treat it like OpenAI.[^12]

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy-key-or-token-if-needed",
    base_url="http://localhost:8000/v1",  # TGI/vLLM endpoint
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain retrieval-augmented generation in 3 bullet points."},
]

response = client.chat.completions.create(
    model="llama-3-70b-instruct",  # or whichever alias your server uses
    messages=messages,
)

print(response.choices.message.content)
```

Use the **same RAG context pattern** as OpenAI/Azure by concatenating retrieved chunks into the user or system message.

***

## 7. OpenRouter – Multi‑Provider Chat with One API

OpenRouter provides an **OpenAI‑compatible** `/chat/completions` API wrapping 400+ models (OpenAI, Anthropic, Google, etc.).[^7][^9][^18]

### 7.1 Install and configure client

```bash
pip install openai
```

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
```

### 7.2 Basic chat completion with RAG context

```python
retrieved_chunks = ["Doc1: ...", "Doc2: ..."]
context = "\n\n".join(retrieved_chunks)

messages = [
    {
        "role": "user",
        "content": (
            "You are a helpful assistant. Use only the context.\n\n"
            f"Context:\n{context}\n\nQuestion: Summarize our refund policy."
        ),
    },
]

response = client.chat.completions.create(
    model="anthropic/claude-3.5-sonnet",  # or openai/gpt-4.1, google/gemini-2.0-flash, etc.
    messages=messages,
)

print(response.choices.message.content)
```

### 7.3 Streaming with OpenRouter

From recent OpenRouter examples using the OpenAI client:[^9][^18][^7]

```python
stream = client.chat.completions.create(
    model="openai/gpt-4.1-mini",
    messages=[{"role": "user", "content": "Stream a short poem about RAG."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices.delta
    if delta.content:
        print(delta.content, end="", flush=True)
print()
```

OpenRouter will route to the chosen upstream model and stream tokens through the OpenAI‑compatible interface.

***

## 8. LiteLLM – OpenAI‑Compatible Proxy over Many Providers

LiteLLM exposes many providers (OpenAI, Anthropic, Vertex, Bedrock, etc.) behind an **OpenAI‑compatible proxy**.[^8][^10][^12]

Typical flow:

1. Run LiteLLM proxy with a config mapping model names → providers.
2. Point your OpenAI client at the LiteLLM proxy `base_url`.
3. Use `model="my-model"` and LiteLLM routes appropriately.

### 8.1 Example LiteLLM proxy config (YAML)

```yaml
model_list:
  - model_name: my-openai-model
    litellm_params:
      model: openai/gpt-4.1-mini
      api_base: https://api.openai.com/v1
      api_key: sk-openai-...

  - model_name: my-bedrock-claude
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet
      api_base: https://bedrock-runtime.us-east-1.amazonaws.com
      api_key: aws-bedrock-key-or-profile
```

(See official docs for the exact provider configuration syntax.)[^12]

### 8.2 Calling LiteLLM proxy with OpenAI client

From LiteLLM OpenAI‑compatible examples:[^10][^8]

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-litellm-proxy-key",  # LiteLLM virtual key or any placeholder
    base_url="http://localhost:4000",  # LiteLLM proxy URL
)

messages = [
    {"role": "user", "content": "this is a test request, write a short poem"},
]

response = client.chat.completions.create(
    model="my-openai-model",  # one of the names from config.yaml
    messages=messages,
)

print(response.choices.message.content)
```

LiteLLM also supports streaming with the same `stream=True` parameter, since it mirrors the OpenAI API.[^10]

```python
stream = client.chat.completions.create(
    model="my-openai-model",
    messages=[{"role": "user", "content": "Stream a haiku about RAG."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices.delta
    if delta.content:
        print(delta.content, end="", flush=True)
print()
```

You can point **LangChain, LlamaIndex, and other OpenAI‑compatible libraries** at LiteLLM simply by setting `base_url` and `api_key` appropriately.[^8][^10][^12]

***

## 9. Integration Pattern for RAG

Across these providers, the **RAG integration pattern is the same**:

1. **Pick model & provider** based on cost, latency, context length, and deployment constraints.
2. **Retrieve** relevant context from your vector DB / search backend.
3. **Format messages**:
   - System: instructions about grounding, style, and citation requirements.
   - User: include the **context block** + **end‑user question**.
4. **Call the chat API**:
   - For non‑streaming, get `message.content` and return.
   - For streaming, iterate over chunks and append `delta.content` / `chunk.text`.
5. **Log everything** (prompt, context, model, tokens, latency) into your observability stack.

Use this cheat‑sheet alongside your **Retrieval, Embeddings, and Reranker** cheat‑sheets so your agents and services can switch LLM providers with minimal code changes while keeping the same high‑level RAC/RAG architecture.

---

## References

1. [How to stream completions - OpenAI for developers](https://developers.openai.com/cookbook/examples/how_to_stream_completions/) - By default, when you request a completion from the OpenAI, the entire completion is generated before...

2. [Generating content | Gemini API | Google AI for Developers](https://ai.google.dev/api/generate-content) - The Gemini API supports content generation with images, audio, code, tools, and more. For details on...

3. [Python SDK - Claude API Docs](https://platform.claude.com/docs/en/api/sdks/python) - Install and configure the Anthropic Python SDK with sync and async client support

4. [Streaming Messages - Claude API Docs](https://platform.claude.com/docs/en/build-with-claude/streaming) - Streaming with SDKs. The Python and TypeScript SDKs offer multiple ways of streaming. The Python SDK...

5. [Gemini API quickstart - Google AI for Developers](https://ai.google.dev/gemini-api/docs/quickstart) - Here is an example that uses the generateContent method to send a request to the Gemini API using th...

6. [Amazon Bedrock Runtime examples using SDK for Python (Boto3)](https://docs.aws.amazon.com/code-library/latest/ug/python_3_bedrock-runtime_code_examples.html) - The following code examples show you how to perform actions and implement common scenarios by using ...

7. [OpenRouter: A Guide With Practical Examples](https://www.datacamp.com/tutorial/openrouter) - We'll be using the openai Python package to interact with OpenRouter's API, along with python-dotenv...

8. [Langchain, OpenAI SDK, LlamaIndex, Instructor, Curl 示例](https://www.aidoczh.com/litellm/docs/proxy/user_keys/) - LiteLLM Proxy 是 OpenAI 兼容 的，并支持以下功能：

9. [OpenRouter in Python: Use Any LLM with One API Key](https://snyk.io/articles/openrouter-in-python-use-any-llm-with-one-api-key/) - OpenRouter supports streaming responses, which is useful for chat applications: stream = client.chat...

10. [Langchain, OpenAI SDK, LlamaIndex, Instructor, Curl examples | liteLLM](https://litellm.vercel.app/docs/proxy/user_keys) - LiteLLM Proxy is OpenAI-Compatible, and supports:

11. [Azure Chat Completions example (preview)](https://cookbook.openai.com/examples/azure/chat) - This example will cover chat completions using the Azure OpenAI service. It also includes informatio...

12. [OpenAI-Compatible Endpoints - LiteLLM](https://docs.litellm.ai/docs/providers/openai_compatible) - Selecting openai as the provider routes your request to an OpenAI-compatible endpoint using the upst...

13. [Invoke a model with the OpenAI Chat Completions API](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html) - You can use the Create chat completion API with all OpenAI models supported in Amazon Bedrock and in...

14. [Azure Chat Completions example (preview) - OpenAI for developers](https://developers.openai.com/cookbook/examples/azure/chat/) - This example will cover chat completions using the Azure OpenAI service. It also includes informatio...

15. [anthropic-sdk-python/helpers.md at main · anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python/blob/main/helpers.md) - Contribute to anthropics/anthropic-sdk-python development by creating an account on GitHub.

16. [Streaming text - Gemini by Example](https://geminibyexample.com/002-streaming-text/) - Learn the Gemini API through annotated examples

17. [Streaming model responses when using Azure OpenAI](https://ravichaganti.com/blog/streaming-completions-azure-openai/) - Responses are streamed to the user interface as they are generated using ChatGPT and similar tools. ...

18. [Create a chat completion | OpenRouter | Documentation](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request) - Sends a request for a model response for the given chat conversation. Supports both streaming and no...

