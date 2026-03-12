# Evaluation & Observability – Code Cheat‑Sheet (2026)

This cheat‑sheet shows **how to use** each evaluation/observability component from your stack guide:

- RAGAS
- TruLens
- DeepEval
- Arize Phoenix
- LangSmith
- Langfuse

For each, you get: install, minimal wiring for a RAG pipeline, and how to interpret / use results.

Sources: current docs and tutorials as of early 2026.[^1][^2][^3][^4][^5][^6][^7][^8][^9][^10][^11][^12]

***

## 1. RAGAS – Automated RAG Evaluation

RAGAS is an open‑source framework focused on **RAG‑specific metrics**: faithfulness, answer relevance, context precision/recall, etc.[^11]

### 1.1 Install

```bash
pip install ragas
```

### 1.2 Quickstart pattern – evaluate a list of Q/A/context triples

The latest PyPI quickstart recommends building a Pandas/DataFrame‑style dataset and applying metrics.[^11]

```python
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)

# Example dataset
# questions: list[str]
# answers: list[str] (model outputs)
# contexts: list[list[str]] (each question has a list of context snippets)

questions = [
    "What is the refund policy for EU customers?",
]
answers = [
    "EU customers can request a refund within 30 days of purchase.",
]
contexts = [[
    "Our refund policy allows EU customers to request refunds within 30 days of purchase.",
    "APAC refund policy is 15 days.",
]]

dataset = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
}

result = evaluate(
    dataset,
    metrics=[answer_relevancy, faithfulness, context_precision, context_recall],
)

print(result)
# result.scores gives per-metric scores; you can log or compare across runs
```

### 1.3 RAGAS Quickstart CLI project

The new `ragas quickstart` command scaffolds a full RAG evaluation project.[^1][^11]

```bash
# List templates
ragas quickstart

# Create a RAG evaluation project
ragas quickstart rag_eval -o ./rag-eval-project

cd rag-eval-project
uv sync && python evals.py  # or your chosen runner
```

This gives you a ready‑made layout (`datasets/`, `experiments/`, `logs/`, `configs/`) aligned with best practices.[^1]

***

## 2. TruLens – RAG Triad & Tracing

TruLens defines the **RAG Triad** (context relevance, groundedness, answer relevance) and provides tools to compute these metrics on your RAG traces.[^2][^6][^13]

### 2.1 Install

```bash
pip install trulens_eval
```

### 2.2 Core concepts

- **Records**: serialized traces of your LLM/RAG pipeline.
- **Feedback functions**: LLM‑ or heuristic‑based metrics (e.g., context relevance, groundedness, answer relevance).[^6]

### 2.3 Minimal RAG Triad example (conceptual)

(Simplified from TruLens docs – actual wrappers differ by framework.)[^2][^6]

```python
from trulens_eval import Tru, Feedback
from trulens_eval.feedback import Groundedness, Relevance

tru = Tru()  # local or remote DB

# Suppose rag_app is your callable that returns {"answer": str, "context": list[str]}

def rag_app(question: str):
    ...

# Define feedback functions

groundedness = Feedback(Groundedness())
context_relevance = Feedback(Relevance())
answer_relevance = Feedback(Relevance())

# Wrap app with TruLens instrumentation

from trulens_eval import TruCustomApp

tru_app = TruCustomApp(
    app_id="my-rag-app",
    app=rag_app,
    feedbacks=[groundedness, context_relevance, answer_relevance],
)

# Run some evals

questions = ["What is the refund policy for EU customers?", "How do I reset my password?"]

for q in questions:
    tru_app.app(q)  # TruLens records traces & feedback

# Launch dashboard (optional)
tru.run_dashboard()  # open http://localhost:8501
```

In practice, you’ll use the official examples for LangChain/LlamaIndex/Haystack integrations, which wire your retriever and generator into **TruRecords** and compute RAG Triad metrics per interaction.[^13][^2]

***

## 3. DeepEval – RAG Evaluation & RAG Triad

DeepEval (by Confident AI) focuses on **RAG evaluation** with carefully designed metrics and RAG‑Triad‑style workflows.[^7][^12]

### 3.1 Install

```bash
pip install deepeval
```

### 3.2 Quickstart – metric & `@observe` for retriever

From the RAG Evaluation Quickstart docs:[^7]

```python
from deepeval.tracing import observe, update_current_span
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

contextual_relevancy = ContextualRelevancyMetric(threshold=0.6)

@observe(metrics=[contextual_relevancy])
def retriever(query: str):
    # Your retriever implementation
    retrieved_context = ["..."]

    # Attach the test case to the current span
    update_current_span(
        test_case=LLMTestCase(input=query, retrieval_context=retrieved_context),
    )

    return retrieved_context
```

### 3.3 Evaluating a full RAG pipeline with a dataset

From DeepEval’s RAG quickstart and RAG Triad tutorial:[^12][^7]

```python
from deepeval.dataset import EvaluationDataset, Golden

# Define dataset ("goldens") – expected behavior for queries
dataset = EvaluationDataset(
    goldens=[
        Golden(input="What is the refund policy?"),
        Golden(input="How do I cancel my subscription?"),
    ]
)

# Your RAG pipeline under test

def rag_pipeline(query: str) -> str:
    # 1) Retrieval
    retrieved = retriever(query)
    # 2) Generation using retrieved context
    answer = generate_answer(query, retrieved)
    return answer

# Run evaluation over dataset
for golden in dataset.evals_iterator():
    rag_pipeline(golden.input)

# Then run `deepeval test run` over your test file, as in docs
# deepeval test run test_rag_triad.py
```

DeepEval can separately evaluate **retriever** and **generator** components, or the full RAG Triad in one go.[^12]

***

## 4. Arize Phoenix – Tracing & RAG Evals

Phoenix provides **tracing, vector visualizations, and LLM‑based evaluations**, with built‑in RAG relevance and hallucination evaluators.[^3][^8]

### 4.1 Install

```bash
pip install phoenix-openai phoenix-traces
```

(See the latest install line in the Phoenix docs for your stack.)[^3]

### 4.2 Trace a RAG pipeline and evaluate retrieval relevance

Based on the Arize + Haystack/Phoenix tutorials:[^8][^3]

```python
import phoenix as px
from phoenix.session.evaluation import get_retrieved_documents
from phoenix.evals import (
    llm_classify,
    OpenAIModel,
    RAG_RELEVANCY_PROMPT_TEMPLATE,
)

# Connect to Phoenix server
client = px.Client()  # defaults to localhost:6006

# After running your RAG pipeline with Phoenix tracing enabled:
project_name = "my-rag-project"

# 1) Get retrieved documents as a DataFrame
retrieved_df = get_retrieved_documents(client, project_name=project_name)

# 2) Run relevance evals using an LLM

eval_model = OpenAIModel(model="gpt-4o")

relevancy_results = llm_classify(
    dataframe=retrieved_df,
    model=eval_model,
    template=RAG_RELEVANCY_PROMPT_TEMPLATE,
    rails=["relevant", "unrelated"],
    concurrency=10,
    provide_explanation=True,
)

# 3) Optionally derive numeric scores
relevancy_results["score"] = relevancy_results["explanation"].apply(
    lambda x: 1 if "relevant" in x.lower() else 0
)

print(relevancy_results.head())
```

Phoenix also has built‑in **Q&A correctness** and **hallucination** evaluators via `QAEvaluator` and `HallucinationEvaluator` using `run_evals`.[^8]

***

## 5. LangSmith – Tracing, Datasets, and Evals

LangSmith is an observability platform from LangChain for **tracing, dataset management, and evals** across LLM apps.[^4][^9]

### 5.1 Install and environment

```bash
pip install langsmith
```

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-langsmith-api-key>
export OPENAI_API_KEY=<your-openai-key>
```

### 5.2 Quick trace with `@traceable` decorator

From the tracing quickstart:[^9][^4]

```python
import openai
from langsmith import wrappers, traceable

# Wrap OpenAI client for auto-tracing
client = wrappers.wrap_openai(openai.Client())

@traceable
def pipeline(user_input: str) -> str:
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_input}],
    )
    return result.choices.message.content

pipeline("Hello, world!")
```

Traces appear in your LangSmith project (default `default`), where you can:

- Inspect **inputs, outputs, prompts, and latencies**.
- Attach datasets and run evals over different chain or retriever variants.[^9]

### 5.3 Tracing a LangChain RAG chain

From the RAG chain example:[^9]

```python
import uuid
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Please respond only based on the given context.",
    ),
    ("user", "Question: {question}\nContext: {context}"),
])

model = ChatOpenAI(model="gpt-4.1-mini")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

question = "Can you summarize this morning's meetings?"
context = "During this morning's meeting, we solved all world conflict."
run_id = uuid.uuid4()

result = chain.invoke({"question": question, "context": context}, config={"run_id": run_id})
print("Run id:", run_id)
```

You can then use LangSmith’s UI/evals to compare different retrievers, prompts, and models.

***

## 6. Langfuse – OSS Observability & Evals

Langfuse is an OSS + cloud observability platform focused on **traces, metrics, and evals** for LLM apps, with strong integration into LangChain/LlamaIndex/DSPy.[^5][^10]

### 6.1 Install and minimal SDK usage

```bash
pip install langfuse
```

Set environment variables as per docs:

```bash
export LANGFUSE_SECRET_KEY=<secret>
export LANGFUSE_PUBLIC_KEY=<public>
export LANGFUSE_HOST=https://cloud.langfuse.com  # or self-hosted URL
```

### 6.2 Create spans and generations (Python SDK)

From the Langfuse SDK overview:[^10]

```python
from langfuse import get_client

langfuse = get_client()

# Create a span (e.g., RAG pipeline step)
with langfuse.start_as_current_observation(as_type="span", name="rag-pipeline") as span:
    # Your retrieval step
    retrieved_docs = retriever.invoke("What is the refund policy?")

    span.update(input="What is the refund policy?", output="retrieved docs")

    # Log an LLM generation as a nested observation
    with langfuse.start_as_current_observation(as_type="generation", name="llm-answer") as gen:
        answer = llm_chat("...prompt with context...")
        gen.update(input="...prompt with context...", output=answer)

print("Traces sent to Langfuse")
```

Langfuse provides:

- A UI to inspect **spans**, **generations**, and metrics.
- Prompt management and versioning.[^5]
- Evals over logged traces (e.g., using an LLM‑as‑judge or custom metrics) configured in the UI or via SDK.

***

## 7. How to Add Evals & Observability to Your RAG Project

### 7.1 Logging

Always log at least:

- User query.
- Retrieved document IDs, scores, and content (or references to content).
- Final answer text.
- Model, prompt template/version, and parameters.
- Latency and token usage.

This enables **offline evaluation** (RAGAS, DeepEval, TruLens, Phoenix) and **online monitoring** (LangSmith, Langfuse).[^6][^10][^7][^8][^11][^9]

### 7.2 Offline eval flow

1. **Build a golden dataset** of `(question, expected_answer, context)` where possible.
2. Run **RAGAS or DeepEval** on this dataset to compare retrieval and generation variants.[^7][^11][^12]
3. Use **TruLens or Phoenix** to compute richer metrics (RAG Triad, hallucination) over stored traces.[^3][^6][^8]

### 7.3 Online eval & observability

1. Instrument your app with **LangSmith or Langfuse** for tracing.
2. Sample production traffic and periodically run **LLM‑as‑judge metrics** (via RAGAS, DeepEval, Phoenix, or custom LangSmith/Langfuse evals).
3. Monitor metric trends (faithfulness, relevance, latency, cost) and alert on regressions.

Use this cheat‑sheet with your other RAC/RAG component sheets to systematically **measure and improve** your system instead of relying on ad‑hoc manual checks.

---

## References

1. [Ragas Quickstart: The Easiest Leap Into LLM Evaluation - LinkedIn](https://www.linkedin.com/pulse/ragas-quickstart-easiest-leap-llm-evaluation-ragas-io-zboec) - One of the most common challenges we hear from developers, especially those building with LLMs for t...

2. [The RAG Triad¶](https://www.trulens.org/trulens/getting_started/core_concepts/rag_triad/) - Evaluate and track LLM applications. Explain Deep Neural Nets.

3. [Trace & Evaluate your Agent with Arize Phoenix - Hugging Face](https://huggingface.co/blog/smolagents-phoenix) - Arize Phoenix provides a centralized platform to trace, evaluate, and debug your agent's decisions i...

4. [Tracing Quick Start | 🦜️🛠️ LangSmith](https://docs.smith.lang.chat/old/tracing/quick_start) - You can get started with LangSmith tracing using either LangChain, the Python SDK, the TypeScript SD...

5. [LangFuse: LLM Engineering Platform For Monitoring And Evals](https://www.datacamp.com/tutorial/langfuse) - Discover how LangFuse brings visibility, organization, and quality control to LLM apps. Follow this ...

6. [RAG Triad - TruLens](https://www.trulens.org/getting_started/core_concepts/rag_triad/) - The RAG triad is made up of 3 evaluations: context relevance, groundedness and answer relevance. Sat...

7. [RAG Evaluation Quickstart | DeepEval by Confident AI](https://deepeval.com/docs/getting-started-rag) - RAG evaluation involves evaluating the retriever and generator as separately components. This is bec...

8. [Trace and Evaluate RAG with Arize Phoenix - Haystack](https://haystack.deepset.ai/cookbook/arize_phoenix_evaluate_haystack_rag) - In this tutorial, we will trace and evaluate a Haystack RAG pipeline. We'll evaluate using three dif...

9. [Trace LangChain applications (Python and JS/TS)](https://docs.langchain.com/langsmith/trace-with-langchain) - LangSmith supports distributed tracing with LangChain Python. This allows you to link runs (spans) a...

10. [Langfuse SDKs](https://langfuse.com/docs/observability/sdk/overview) - Fully typed SDKs for Python and JavaScript/TypeScript with unified setup, instrumentation, and advan...

11. [ragas](https://pypi.org/project/ragas/) - Evaluation framework for RAG and LLM applications

12. [Implement Rag Triad...](https://atamel.dev/posts/2025/01-14_rag_evaluation_deepeval/) - In my previous Evaluating RAG pipelines post, I introduced two approaches to evaluating RAG pipeline...

13. [Evaluate Your Multimodal RAG Using Trulens - Zilliz blog](https://zilliz.com/blog/evaluating-multimodal-rags-in-practice-trulens) - Understand multimodal models and multimodal RAG as well as learn how to evaluate multimodal RAG syst...

