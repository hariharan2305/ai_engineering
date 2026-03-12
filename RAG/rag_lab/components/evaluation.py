import numpy as np
from dataclasses import dataclass

from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


@dataclass
class EvalSample:
    """One QA pair for evaluation. ground_truth required for answer_similarity."""
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))


def build_ragas_metrics(model: str = "gpt-4o-mini") -> list:
    """
    Build RAGAS metrics using the 0.2.x class-based API.
    LLM and embeddings are injected at construction — no mutable global state.

    LLM-as-judge metrics (variance, but understands semantics):
      - Faithfulness:     is the answer supported by the retrieved context?
      - AnswerRelevancy:  does the answer address the question?
    """
    llm = LangchainLLMWrapper(ChatOpenAI(model=model))
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )
    return [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
    ]


def compute_answer_similarity(samples: list[EvalSample], embedder) -> float:
    """
    Deterministic metric: cosine similarity between answer and ground_truth embeddings.
    Uses our local SentenceTransformerEmbedder — zero LLM calls, fully reproducible.
    If this score moves when you swap a component, the improvement is real.
    """
    scores = []
    for s in samples:
        if not s.ground_truth:
            continue
        answer_emb = embedder.embed([s.answer])[0]
        truth_emb = embedder.embed([s.ground_truth])[0]
        scores.append(_cosine_similarity(answer_emb, truth_emb))
    return float(np.mean(scores)) if scores else 0.0


def evaluate_pipeline(
    samples: list[EvalSample],
    embedder=None,
    metrics: list | None = None,
) -> dict[str, float]:
    """
    Run full evaluation and return mean scores per metric.

    Metric interpretation:
      faithfulness        [LLM judge]   — is the answer grounded in context?
      answer_relevancy    [LLM judge]   — does the answer address the question?
      answer_similarity   [Deterministic] — cosine sim vs ground truth; trust this most
                                           when comparing component changes.
    """
    if metrics is None:
        metrics = build_ragas_metrics()

    data = {
        "question": [s.question for s in samples],
        "answer": [s.answer for s in samples],
        "contexts": [s.contexts for s in samples],
        "ground_truth": [s.ground_truth for s in samples],
    }
    dataset = Dataset.from_dict(data)
    result = ragas_evaluate(dataset, metrics=metrics)
    df = result.to_pandas()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    scores = {col: float(df[col].mean()) for col in numeric_cols}

    if embedder is not None:
        scores["answer_similarity"] = compute_answer_similarity(samples, embedder)

    return scores
