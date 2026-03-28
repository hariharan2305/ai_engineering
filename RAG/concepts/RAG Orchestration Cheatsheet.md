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

RealPython's LangGraph tutorial outlines the pattern:[^8]

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

`RouterQueryEngine` is LlamaIndex's core primitive for **routing between retrieval strategies** in a single orchestration object.[^9]

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

DSPy is a declarative framework from Stanford's Hazy Research that lets you specify **modules** (retriever + generator) and then **optimize** prompts or models to improve metrics.[^4][^11]

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
3. Wire them visually into a pipeline in the UI (similar to Haystack's Pipeline code but GUI‑driven).[^6][^13]
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

From the Agent Builder walkthroughs, you don't usually write Python for orchestration; you configure:

- Agent instructions and persona.
- RAG tools pointing to data stores.
- Tool invocation behavior and flows (e.g., multi‑tool queries).[^7]

Your app then calls the **Agent endpoint** (REST/gRPC) instead of directly orchestrating retrieval + LLM yourself.

***

## 7. Corrective RAG (CRAG) – Self-Grading with Fallback

Corrective RAG adds a **relevance-grading step after retrieval**. If the retrieved chunks score below a threshold, the system triggers a fallback (re-query, web search, or query rewrite) instead of generating from weak context. The result is a retrieve → grade → decide loop before any generation happens.[^18]

### 7.1 When to use CRAG

- Your corpus has **coverage gaps** (some questions fall outside stored docs).
- You can tolerate a slightly higher latency in exchange for fewer hallucinations.
- You have a fallback source available (web search, a second index, a broader knowledge base).

### 7.2 The eval-then-decide loop

```
User query
    │
    ▼
[retrieve]  ──→  chunks
    │
    ▼
[grade chunks]  ──→  all relevant?
    ├── yes ──→ [generate answer]  ──→ response
    └── no  ──→ [fallback: rewrite query / web search]
                    │
                    ▼
                [retrieve again]  ──→ [generate answer]  ──→ response
```

### 7.3 Implementation with LangGraph

#### Install

```bash
pip install -U langgraph "langchain-openai" "langchain-core>=0.3"
```

#### Grade documents node (binary relevance check)

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain.chat_models import init_chat_model

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question.\n"
    "Retrieved document:\n\n{context}\n\n"
    "User question: {question}\n"
    "Give a binary score 'yes' or 'no': is this document relevant?"
)

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' if relevant, 'no' if not relevant")

grader_llm = init_chat_model("gpt-4o-mini", temperature=0)

def grade_documents(state) -> Literal["generate_answer", "rewrite_question"]:
    """Route based on whether retrieved docs are relevant."""
    question = state["messages"][0].content
    context  = state["messages"][-1].content
    prompt   = GRADE_PROMPT.format(question=question, context=context)
    result   = grader_llm.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    return "generate_answer" if result.binary_score == "yes" else "rewrite_question"
```

#### Wire the CRAG graph

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model

response_llm = init_chat_model("gpt-4o-mini", temperature=0)

# Node: decide whether to retrieve or answer directly
def generate_query_or_respond(state: MessagesState):
    response = response_llm.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}

# Node: generate final answer from (now-graded) context
def generate_answer(state: MessagesState):
    response = response_llm.invoke(state["messages"])
    return {"messages": [response]}

# Node: rewrite query before re-retrieval (the fallback trigger)
def rewrite_question(state: MessagesState):
    question = state["messages"][0].content
    rewritten = response_llm.invoke(
        [{"role": "user", "content": f"Rewrite this question for better retrieval: {question}"}]
    )
    return {"messages": state["messages"] + [rewritten]}

# Build graph
workflow = StateGraph(MessagesState)
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("rewrite_question", rewrite_question)

workflow.add_edge(START, "generate_query_or_respond")

# After LLM decides: call retriever tool or respond directly
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {"tools": "retrieve", END: END},
)

# After retrieval: grade docs, route to generate or fallback
workflow.add_conditional_edges("retrieve", grade_documents)

# Fallback loops back to let LLM retry with rewritten query
workflow.add_edge("rewrite_question", "generate_query_or_respond")
workflow.add_edge("generate_answer", END)

crag_graph = workflow.compile()
```

The key difference from a plain RAG chain: `grade_documents` is a **conditional edge**, not a node, so the routing decision happens in-graph without any external control flow.

### 7.4 Triggering fallback: decision criteria

| Signal | Recommended fallback |
|---|---|
| Grader scores all chunks as irrelevant | Rewrite query → re-retrieve from same index |
| Grader score is low AND query is time-sensitive | Web search (e.g., Tavily) |
| Grader score is low AND query is relational | Knowledge-graph lookup |
| Grader score is high but generation is uncertain | Ask clarifying question (human-in-the-loop) |

***

## 8. Agentic RAG – Planning-First Retrieval

Agentic RAG replaces a fixed retrieve → generate chain with an **autonomous planning loop**. The agent inspects the query, selects the right retrieval tool (vector search, KG traversal, web search, SQL, etc.), runs it, checks whether the result is sufficient, and either generates or plans another retrieval step.[^19]

### 8.1 When to use Agentic RAG

- Queries span **multiple retrieval sources** with different access patterns.
- The system needs to decide between retrieval strategies at runtime (not at design time).
- Queries may require **multi-hop reasoning** (retrieve → process → retrieve again).

### 8.2 The agent planning loop

```
User query
    │
    ▼
[planner / router LLM]  ──→  picks tool(s)
    │
    ├── vector_search(query)   ←── general factual questions
    ├── graph_lookup(entities) ←── relational / multi-hop questions
    └── web_search(query)      ←── current events / unknown topics
    │
    ▼
[retrieved context]
    │
    ▼
[LLM checks sufficiency]
    ├── sufficient ──→ [generate answer]
    └── not enough  ──→ back to planner with refined query
```

### 8.3 Implementation with LangGraph (multi-tool agent)

#### Define retrieval tools

```python
from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits, embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

@tool
def vector_search(query: str) -> str:
    """Search the internal knowledge base for factual questions."""
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)

@tool
def web_search(query: str) -> str:
    """Search the web for current events or topics not in the knowledge base."""
    # Replace with Tavily, SerpAPI, or any live-search client
    return f"[web results for: {query}]"

@tool
def graph_lookup(entities: str) -> str:
    """Query the knowledge graph for relational or multi-hop questions."""
    # Replace with your KG client (e.g., Neo4j, Amazon Neptune)
    return f"[KG results for entities: {entities}]"

tools = [vector_search, web_search, graph_lookup]
```

#### Option A: prebuilt ReAct agent (simplest for multi-tool routing)

```python
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

agent_llm = init_chat_model("gpt-4o-mini", temperature=0)

agentic_rag = create_react_agent(
    model=agent_llm,
    tools=tools,
    prompt=(
        "You are an intelligent RAG agent. For each question:\n"
        "- Use vector_search for factual questions over the knowledge base.\n"
        "- Use web_search for current events or topics not in the knowledge base.\n"
        "- Use graph_lookup for relational questions requiring multi-hop reasoning.\n"
        "Retrieve before answering. Only answer when you have sufficient context."
    ),
)

result = agentic_rag.invoke({
    "messages": [{"role": "user", "content": "What is the current EU AI Act status?"}]
})
print(result["messages"][-1].content)
```

#### Option B: custom graph for explicit planner control

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model

planner_llm = init_chat_model("gpt-4o-mini", temperature=0)

def plan_and_route(state: MessagesState):
    """Planner node: LLM decides which tool to call."""
    response = planner_llm.bind_tools(tools).invoke(state["messages"])
    return {"messages": [response]}

def generate_final_answer(state: MessagesState):
    """Generation node: produce answer from retrieved context."""
    response = planner_llm.invoke(state["messages"])
    return {"messages": [response]}

workflow = StateGraph(MessagesState)
workflow.add_node("planner", plan_and_route)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("generate", generate_final_answer)

workflow.add_edge(START, "planner")

# Planner either calls a tool or decides it has enough context
workflow.add_conditional_edges(
    "planner",
    tools_condition,
    {"tools": "tools", END: END},
)

# After tool execution, loop back to planner for sufficiency check
workflow.add_edge("tools", "planner")

agentic_graph = workflow.compile()
```

The `tools_condition` edge drives the planning loop: the planner keeps calling tools until it decides enough context exists, then routes to `END` with a final answer.

### 8.4 Tool selection heuristics

| Query characteristic | Tool to use |
|---|---|
| "What is…" / "Explain…" (static knowledge) | `vector_search` |
| "Who is related to X via Y" (multi-hop) | `graph_lookup` |
| "Latest / current / as of today" (temporal) | `web_search` |
| "How many rows in table T" (structured) | SQL tool |
| Mixed (e.g., compare KG entity to current news) | Multiple tools, planner sequences them |

### 8.5 CRAG vs Agentic RAG

| Dimension | CRAG | Agentic RAG |
|---|---|---|
| Control flow | Fixed: retrieve → grade → fallback | Dynamic: planner decides tool sequence |
| Tool selection | Single retriever + one fallback | Multiple tools, runtime choice |
| Latency | Low overhead (one extra grader call) | Higher (multi-step planning loop) |
| Best for | Improving a single-retriever pipeline | Multi-source, multi-hop queries |
| LangGraph primitive | Conditional edges on grader output | `create_react_agent` or `ToolNode` loop |

***

## 9. How to Choose and Combine Orchestration Frameworks

- **LangChain**: best when you want **Python‑first**, code‑level control over RAG chains, retrievers, tools, and when you rely on the LangChain ecosystem (vector stores, tools, observability via LangSmith).[^17][^1]
- **LangGraph**: use when you need **stateful, multi‑step, or multi‑agent** workflows (e.g., complex agentic RAG, tool‑calling with loops, human‑in‑the‑loop approval). Graphs make control flow explicit and testable.[^2][^8]
- **LlamaIndex**: ideal when your mental model is **"build indices, then query them"** and you want advanced index types (tree, KG, graph) with built‑in prompt compression and router engines.[^9]
- **Haystack**: good fit if you need **production‑grade, search‑centric pipelines** with strong OpenSearch/Elastic integration and typed components.[^10][^3]
- **DSPy**: reach for it when you want **eval‑driven optimization** of RAG prompts and modules (LLMOps) rather than hand‑tuned prompts.[^4][^11]
- **Cloud‑native orchestration** (Bedrock Agents, Azure Flows, Vertex Agent Builder): choose these when you prefer **managed orchestration**, tight cloud integration, IAM/security, and less bespoke Python orchestration code.[^13][^5][^7][^6]
- **CRAG**: layer on top of any single-retriever pipeline when you need self-correction without full agent complexity.[^18]
- **Agentic RAG**: use when queries are heterogeneous and require dynamic tool selection at runtime across multiple retrieval backends.[^19]

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

18. [Agentic RAG with LangGraph – LangChain OSS Python Docs](https://docs.langchain.com/oss/python/langgraph/agentic-rag) - Agentic RAG: conditional edges, document grading, and fallback routing with LangGraph StateGraph.

19. [LangGraph Agents – prebuilt ReAct agent and multi-tool routing](https://langchain-ai.github.io/langgraph/agents/agents/) - Build reliable, stateful AI systems with LangGraph's prebuilt agent primitives and ToolNode.

