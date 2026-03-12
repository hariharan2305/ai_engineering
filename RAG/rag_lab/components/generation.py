from openai import OpenAI

from .base import RetrievedChunk

_SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using ONLY the information provided in the context below.
If the context does not contain enough information to answer the question, say "I don't have enough information in the provided context to answer this."
Do not use any prior knowledge beyond what is explicitly stated in the context."""


class OpenAIGenerator:
    """
    Generates answers using OpenAI models with retrieved chunks as context.

    Experiment knobs:
      - model: swap to gpt-4o for higher quality, gpt-4o-mini for speed/cost
      - max_tokens: observe truncation effects on answer quality
      - system_prompt: experiment with stricter/looser grounding instructions
    """

    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 512):
        self.client = OpenAI()
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        context = "\n\n---\n\n".join(
            f"[Context {i + 1} | source: {chunk.metadata.get('filename', 'unknown')}]\n{chunk.text}"
            for i, chunk in enumerate(chunks)
        )
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "developer", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
        )
        return response.choices[0].message.content
