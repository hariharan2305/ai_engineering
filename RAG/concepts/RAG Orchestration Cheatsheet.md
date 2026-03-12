# Orchestration, Agents, and Workflows – Full Code Cheat‑Sheet (2026)

This cheat‑sheet goes in **full detail** on orchestration and agents for RAC/RAG systems:

- LangChain (chains, Runnables, simple RAG chain)
- LangGraph (graph‑based, stateful agents)
- LlamaIndex (index‑first orchestration + RouterQueryEngine workflows)
- Haystack 2.x (Pipeline‑based RAG)
- DSPy (declarative RAG modules + optimization)
- Cloud‑native: AWS Agents for Bedrock, Azure AI Studio Flows, Vertex AI Agent Builder / Workflows

Each section shows:

- When and why to use that framework
- Installation
- Minimal-but-real code for a RAG‑style workflow or agent

Sources: current LangChain/LangGraph/LlamaIndex/Haystack/DSPy docs and tutorials, plus AWS/Azure/GCP agentic RAG guides.[^1][^2][^3][^4][^5][^6][^7][^8][^9][^10][^11][^12][^13]

***

## 1. LangChain – Chains, Runnables, and Simple RAG

LangChain is the **default orchestration framework** for many Python RAG apps thanks to its wide integration surface (LLMs, vector DBs, tools).[^14][^1]

### 1.1 Install (minimal core + OpenAI + vectorstores)

```bash
pip install "langchain-core>=0.3" "langchain-openai" "langchain-community" faiss-cpu
```

Add vector‑store specific packages as needed (e.g., `langchain-qdrant`, `langchain-pinecone`).

### 1.2 Simple chain with `ChatPromptTemplate` → LLM → output parser

This is the basic pattern LangChain promotes in its quickstart.[^1][^14]

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "Answer the following question: {question}"),
])

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"question": "What is retrieval-augmented generation?"})
print(result)
```

### 1.3 Simple RAG chain: Retriever + prompt + LLM

Assuming you have a LangChain `Retriever` (e.g., from Qdrant/FAISS/Pinecone) already set up, a typical small RAG chain is:[^1]

```python
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# retriever: BaseRetriever – e.g., vectorstore.as_retriever(k=4)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer ONLY from the context. "
        "If the answer is not in the context, say you don't know.",
    ),
    (
        "user",
        "Question: {question}\n\nContext:\n{context}",
    ),
])

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
parser = StrOutputParser()

# Runnable graph:
# {"question"} → {"question", "context"} → prompt → model → parser

def join_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {
        "question": itemgetter("question"),
        "context": itemgetter("question") | retriever | join_docs,
    }
    | prompt
    | model
    | parser
)

answer = rag_chain.invoke({"question": "Explain our refund policy for EU customers."})
print(answer)
```

Use this same pattern inside **LangGraph**, in FastAPI routes, or in background workers.

***

## 2. LangGraph – Stateful, Graph‑Based Agents

LangGraph builds on LangChain to support **stateful, cyclic, multi‑actor** LLM workflows and agents.[^2][^8]

### 2.1 Install

```bash
pip install -U langgraph "langchain[anthropic]"  # or [openai] etc.
```

### 2.2 Quickstart – prebuilt ReAct‑style agent

LangGraph has prebuilt agents; the quickstart uses `create_react_agent`.[^2]

```python
from langgraph.prebuilt import create_react_agent

# Tool the agent can call
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",  # via langchain[anthropic]
    tools=[get_weather],
    prompt="You are a helpful assistant.",
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "what is the weather in sf"},
    ]
})

print(result["messages"][-1]["content"])  # final assistant message
```

### 2.3 Build your own LangGraph state graph (conceptual pattern)

RealPython’s LangGraph tutorial outlines the pattern:[^8]

1. Define a **state** (Pydantic model / dict) for your agent.
2. Define **nodes** (functions) that operate on state.
3. Define **edges** (routing conditions) between nodes.
4. Compile into a `StateGraph` and run with `.invoke()` / `.stream()`.

Simplified skeleton:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

class AgentState(TypedDict):
    messages: list

llm = ChatOpenAI(model="gpt-4.1-mini")

# Node definitions
def call_model(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

# Build graph
graph = StateGraph(AgentState)

graph.add_node("model", call_model)

graph.set_entry_point("model")

graph.add_edge("model", END)

app = graph.compile()

# Run graph
result = app.invoke({
    "messages": [{"role": "user", "content": "Explain RAG in one paragraph."}],
})

print(result["messages"][-1].content)
```

You can extend this by adding nodes for **retrieval**, **tool calls**, **branching**, and **memory**, making LangGraph suitable for complex agentic RAG flows.[^8]

***

## 3. LlamaIndex – Index‑First Orchestration & RouterQueryEngine

LlamaIndex focuses on **indices** (vector, summary, tree, KG) and exposes **query engines** and **router query engines** for orchestration.[^9]

### 3.1 Install core + OpenAI integrations

```bash
pip install llama-index-core llama-index-llms-openai llama-index-embeddings-openai
```

### 3.2 Simple RAG: VectorStoreIndex query engine

```python
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

os.environ["OPENAI_API_KEY"] = "sk-..."

# 1) Load documents from a directory
documents = SimpleDirectoryReader("./data").load_data()

# 2) Build index
llm = LlamaOpenAI(model="gpt-4.1-mini", temperature=0)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# 3) Query engine
query_engine = index.as_query_engine(
    similarity_top_k=5,
    llm=llm,
)

response = query_engine.query("Explain our refund policy for EU customers.")
print(response)
```

### 3.3 RouterQueryEngine – multi‑index routing workflow

From LlamaIndex RouterQueryEngine example (implemented as a workflow):[^9]

```python
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors.llm_selectors import LLMSingleSelector

# Load docs
documents = SimpleDirectoryReader("./data").load_data()

# Indices
vec_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
sum_index = SummaryIndex.from_documents(documents)

# Query engines
vec_engine = vec_index.as_query_engine(similarity_top_k=5, llm=llm)
sum_engine = sum_index.as_query_engine(response_mode="tree_summarize", llm=llm)

# Wrap as tools
tools = [
    QueryEngineTool(
        query_engine=sum_engine,
        metadata=ToolMetadata(
            name="summarizer",
            description="Good for high-level summaries over the corpus.",
        ),
    ),
    QueryEngineTool(
        query_engine=vec_engine,
        metadata=ToolMetadata(
            name="retriever",
            description="Good for detailed, local questions.",
        ),
    ),
]

router_engine = RouterQueryEngine.from_defaults(
    query_engine_tools=tools,
    selector=LLMSingleSelector.from_defaults(llm=llm),
)

# Router chooses the best engine based on query
resp1 = router_engine.query("Give me a 2-sentence overview of this repo.")
print(resp1)

resp2 = router_engine.query("What does file X say about error handling?")
print(resp2)
```

`RouterQueryEngine` is LlamaIndex’s core primitive for **routing between retrieval strategies** in a single orchestration object.[^9]

***

## 4. Haystack 2.x – Pipeline‑Based RAG

Haystack 2.x (Haystack‑AI) uses a **Pipeline** abstraction: add components and connect them by input/output names.[^3][^10]

### 4.1 Install

```bash
pip install haystack-ai openai
```

### 4.2 Components and RAG pipeline

From the "First RAG pipeline" tutorial:[^3]

```python
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document

# 1) Setup document store
doc_store = InMemoryDocumentStore()

# 2) Write some documents

writer = DocumentWriter(document_store=doc_store)

docs = [
    Document(content="RAG improves factual grounding by retrieving docs."),
    Document(content="This text talks only about sports."),
]

writer.run(documents=docs)

# 3) Components

text_embedder = OpenAITextEmbedder(api_key="sk-...", model="text-embedding-3-small")
retriever = InMemoryBM25Retriever(document_store=doc_store)
prompt_builder = ChatPromptBuilder(
    template="""You are a helpful assistant. Answer based only on the context.\n\nContext:\n{{ documents }}\n\nQuestion: {{ question }}""",
)
chat_generator = OpenAIGenerator(api_key="sk-...", model="gpt-4.1-mini")

# 4) Build pipeline

basic_rag_pipeline = Pipeline()

basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", chat_generator)

# Connect components
basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder")
basic_rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

# 5) Run
question = "What does RAG help with?"

response = basic_rag_pipeline.run({
    "text_embedder": {"text": question},
    "prompt_builder": {"question": question},
})

print(response["llm"]["replies"].text)
```

Haystack pipelines are particularly strong when you need **observable, typed steps** with pluggable components for enterprise search & RAG.[^10][^3]

***

## 5. DSPy – Declarative RAG Modules & Optimization

DSPy is a declarative framework from Stanford’s Hazy Research that lets you specify **modules** (retriever + generator) and then **optimize** prompts or models to improve metrics.[^4][^11]

### 5.1 Install

```bash
pip install dspy-ai
```

### 5.2 Basic DSPy prediction module

```python
import dspy

# Configure LLM (OpenAI example)
dspy.settings.configure(
    lm=dspy.OpenAI(model="gpt-4.1-mini"),
)

# Signature: what you want to map from → to
qa = dspy.Predict("question: str -> response: str")

answer = qa(question="what are high memory and low memory on linux?")
print(answer.response)
```

### 5.3 Simple RAG module in DSPy

From the RAG tutorial:[^4]

```python
import dspy

# Suppose you have a `search` module that returns a list of passages

def search(question: str):
    # Your retrieval implementation (e.g., vector DB)
    return dspy.Passage("RAG improves grounding by retrieving documents.")

class RAG(dspy.Module):
    def __init__(self):
        # A chain-of-thought predictor taking context + question
        self.respond = dspy.ChainOfThought("context, question -> response")

    def forward(self, question: str):
        context = search(question).passages
        return self.respond(context=context, question=question)

rag = RAG()

result = rag("what is retrieval-augmented generation?")
print(result.response)
```

### 5.4 DSPy optimization (high‑level)

DSPy can automatically tune your RAG for a given **evaluation dataset** and metric:[^11]

1. Define a dataset of `(question, answer, docs)`.
2. Define a **metric** (e.g., exact match, BLEU, RAG‑specific metric).
3. Wrap your RAG module in a `dspy.Module`.
4. Run a DSPy optimizer (e.g., `dspy.MIPROV`) to improve prompts or weights.

This makes DSPy particularly suited for **LLMOps – eval‑driven RAG optimization**.[^11]

***

## 6. Cloud‑Native Orchestration

Cloud platforms ship increasingly full‑featured orchestration layers for RAG and agents.

### 6.1 AWS – Agents for Bedrock

Agents for Bedrock let you define **agents** that can:

- Call Bedrock Knowledge Bases (RAG)
- Invoke tools / action groups (Lambda, APIs)
- Use Bedrock models for reasoning

Configuration is usually done via console or IaC, but the Bedrock samples show Python setup and runtime calls.[^5][^12]

#### 6.1.1 High‑level steps

1. Define an **Agent** in Bedrock (name, description, foundation model, instructions).
2. Attach **Action Groups** (Lambda functions or native function definitions).
3. (Optional) Attach **Knowledge Bases** for RAG.
4. Call the Agent via the `bedrock-agent-runtime` client (`invoke_agent`).[^12][^5]

#### 6.1.2 Python runtime call (simplified)

```python
import boto3

agent_runtime = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

agent_id = "your-agent-id"
agent_alias = "your-agent-alias-id"  # e.g., "TSTALIASID"

user_input = "Book a parent-teacher meeting next Tuesday at 3pm."

response = agent_runtime.invoke_agent(
    agentId=agent_id,
    agentAliasId=agent_alias,
    sessionId="session-123",
    inputText=user_input,
)

# The response stream yields events (chunked text, tool calls, etc.) – see AWS docs
for event in response["completion"]:
    # Process event types: chunk, trace, failure, etc.
    ...
```

Bedrock Agents handle orchestration: choosing when to call tools, KBs, and how to respond.[^5][^12]

***

### 6.2 Azure AI Studio – Prompt Flow & Flows

Azure AI Studio (and Azure ML Prompt Flow) provides **low‑code pipelines** to orchestrate:

- Azure AI Search
- Azure OpenAI
- Document Intelligence
- Custom tools (Python components, web APIs)[^6][^13]

Typical pattern for a RAG flow in Azure AI Studio:

1. Create or use a **Prompt Flow** from templates (e.g., "Q&A on your data").[^13]
2. Add components: document chunking, embedding, Azure AI Search index querying, prompt builder, Azure OpenAI chat.
3. Wire them visually into a pipeline in the UI (similar to Haystack’s Pipeline code but GUI‑driven).[^6][^13]
4. Deploy as an endpoint that your app calls.

Code‑wise, Azure ML exposes these flows as **pipelines** that can be triggered via the Azure ML Python SDK, but in many RAG apps you just call the deployed endpoint and let Azure orchestrate retrieval + generation.[^6]

***

### 6.3 GCP – Vertex AI Agent Builder / Workflows

Vertex AI has multiple orchestration surfaces:

- **Agent Builder** (formerly Dialogflow‑like) for conversational agents + RAG + tools.
- **Vertex AI Search + Grounding** for RAG search backends.
- **Workflows** / custom code for multi‑step flows.

A typical RAG agent in **Agent Builder**:

1. Define an agent in the Vertex AI console.[^7]
2. Attach **data stores** (Vertex AI Search) for RAG over PDFs, websites, etc.[^15][^16]
3. Configure tools / APIs the agent can call.
4. Test in the console and expose as an endpoint.

From the Agent Builder walkthroughs, you don’t usually write Python for orchestration; you configure:

- Agent instructions and persona.
- RAG tools pointing to data stores.
- Tool invocation behavior and flows (e.g., multi‑tool queries).[^7]

Your app then calls the **Agent endpoint** (REST/gRPC) instead of directly orchestrating retrieval + LLM yourself.

***

## 7. How to Choose and Combine Orchestration Frameworks

- **LangChain**: best when you want **Python‑first**, code‑level control over RAG chains, retrievers, tools, and when you rely on the LangChain ecosystem (vector stores, tools, observability via LangSmith).[^17][^1]
- **LangGraph**: use when you need **stateful, multi‑step, or multi‑agent** workflows (e.g., complex agentic RAG, tool‑calling with loops, human‑in‑the‑loop approval). Graphs make control flow explicit and testable.[^2][^8]
- **LlamaIndex**: ideal when your mental model is **“build indices, then query them”** and you want advanced index types (tree, KG, graph) with built‑in prompt compression and router engines.[^9]
- **Haystack**: good fit if you need **production‑grade, search‑centric pipelines** with strong OpenSearch/Elastic integration and typed components.[^10][^3]
- **DSPy**: reach for it when you want **eval‑driven optimization** of RAG prompts and modules (LLMOps) rather than hand‑tuned prompts.[^4][^11]
- **Cloud‑native orchestration** (Bedrock Agents, Azure Flows, Vertex Agent Builder): choose these when you prefer **managed orchestration**, tight cloud integration, IAM/security, and less bespoke Python orchestration code.[^13][^5][^7][^6]

You can mix them: for example, **LangGraph + LangChain** inside your app, while using **Vertex AI Search** as the retrieval backend and **LangSmith/Langfuse** for observability.

Use this cheat‑sheet as the orchestration layer in your RAC/RAG component docs so you and your agents can quickly select and wire the right workflow framework for each project.

---

## References

1. [LangChain Quickstart - Create your Own LLM Chain with Exercises](https://www.youtube.com/watch?v=rqy2DUtY000) - Introduction to LangChain and LLM chains. We'll cover the Quickstart section of the LangChain docs. ...

2. [with a prebuilt agent - LangGraph quickstart - GitHub Pages](https://langchain-ai.github.io/langgraph/agents/agents/) - Build reliable, stateful AI systems, without giving up control

3. [Creating Your First QA Pipeline with Retrieval-Augmentation](https://haystack.deepset.ai/tutorials/27_first_rag_pipeline) - This tutorial shows you how to create a generative question-answering pipeline using the retrieval-a...

4. [Tutorial: Retrieval-Augmented Generation (RAG) - DSPy](https://dspy.ai/tutorials/rag/) - Let's walk through a quick example of basic question answering with and without retrieval-augmented ...

5. [Build an Agentic AI Application with Agents for Amazon Bedrock](https://dev.to/aws-builders/tutorial-build-an-agentic-ai-application-with-agents-for-amazon-bedrock-2cpk) - Here's a step-by-step process for building an application that uses Agents for Amazon Bedrock to...

6. [使用不带代码的 Azure 机器学习管道来构建 RAG 管道（预览版） - Azure Machine Learning](https://docs.azure.cn/zh-cn/machine-learning/how-to-use-pipelines-prompt-flow?view=azureml-api-2) - 设置 Azure 机器学习管道来运行 Prompt Flow 模型（预览版）

7. [Build Agent [walkthrough] (Chatbot, RAG, Tool Use) with Google's Vertex AI-Beginner's Guide](https://www.youtube.com/watch?v=R6bfcgdY-_M) - Learn (deep-dive style) to use 'Agent Builder' in Google Cloud Platform's Vertex AI to build LLM age...

8. [LangGraph: Build Stateful AI Agents in Python](https://realpython.com/langgraph-python/) - LangGraph is a versatile Python library designed for stateful, cyclic, and multi-actor Large Languag...

9. [Router Query Engine | LlamaIndex OSS Documentation](https://developers.llamaindex.ai/python/examples/workflow/router_query_engine/) - This notebook walks through implementation of Router Query Engine, using workflows. Specifically we ...

10. [Creating a Simple RAG Pipeline using Haystack 2.0](https://blog.gopenai.com/creating-a-simple-rag-pipeline-using-haystack-2-0-c84c7c660569?gi=0bb049338b81) - What is RAG?

11. [LLMOps with DSPy: Build RAG Systems Using Declarative ...](https://pyimagesearch.com/2024/09/09/llmops-with-dspy-build-rag-systems-using-declarative-programming/) - Discover how to build Retrieval Augmented Generation (RAG) systems using declarative programming wit...

12. [Create Agent with Return of Control](https://aws-samples.github.io/amazon-bedrock-samples/agents-and-function-calling/bedrock-agents/features-examples/03-create-agent-with-return-of-control/03-create-agent-with-return-of-control/) - Amazon Bedrock cookbook website

13. [Enhancing Language Models Using RAG Architecture in Azure AI ...](https://arinco.com.au/blog/enhancing-language-models-using-rag-architecture-in-azure-ai-studio/) - Implementing the RAG pattern in Azure AI Studio involves creating prompt flow that define the intera...

14. [LangChain Quickstart Guide | Part 1](https://www.youtube.com/watch?v=gVMp_vKwslI) - LangChain Quickstart Guide | Part 1 

LangChain is a framework for developing applications powered b...

15. [Use Vertex AI Search on PDFs (unstructured data) in Cloud Storage ...](https://codelabs.developers.google.com/codelabs/how-to-query-vertex-ai-search-cloud-run-service) - This codelab focuses on using Vertex AI Search, where you can build a Google-quality search app on y...

16. [Search from Vertex AI | Google quality search/RAG for enterprise](https://cloud.google.com/enterprise-search) - Vertex AI Search helps developers build secure, Google-quality search experiences for websites, intr...

17. [Trace LangChain applications (Python and JS/TS)](https://docs.langchain.com/langsmith/trace-with-langchain) - LangSmith supports distributed tracing with LangChain Python. This allows you to link runs (spans) a...

