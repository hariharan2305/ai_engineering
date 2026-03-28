# Chunking Strategy Matters More Than Embedding Model for Domain-Specific Corpora

**Date:** 2026-03-19
**Project:** ai_engineering
**Status:** raw
**Context:** Building a RAG pipeline and running RAGAS evaluations while swapping components one at a time.

---

## The Struggle

I spent two days swapping embedding models and barely moved the needle. I was convinced the embedding model was the bottleneck — better model, better retrieval, right? The RAGAS scores just didn't budge in any meaningful way no matter what I threw at it.

## The Drill-Down

I was methodically going through the RAG component stack trying to find the lever that moved quality. Two full days on embeddings and nothing. Then I went back to chunking — specifically fixing chunk boundaries to actually respect paragraph structure instead of cutting arbitrarily — and the scores jumped 28%. That's when I had to stop and rethink my mental model of where quality comes from in a RAG pipeline.

## The Aha Moment

If your chunks are cutting across paragraph boundaries, you're feeding the embedding model broken context — and no embedding model, no matter how good, can recover meaning from a fragment that lost its beginning or end. The embedding model is trying to encode something that was never coherent to begin with. Fixing chunk boundaries first gives the model something worth encoding. On domain-specific corpora especially, where the vocabulary and concepts are dense and precise, a bad chunk boundary wipes out the signal entirely.

## The One-Liner

> Two days on embeddings, barely moved the needle. Fixed my chunk boundaries, RAGAS jumped 28%. Chunking is the foundation — don't skip it.

---

*Raw insight captured from conversation. Run /polish-insight to generate LinkedIn post variations.*
