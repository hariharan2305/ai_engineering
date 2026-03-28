# Chunk Quality Beats Embedding Model for RAG Score Improvements

**Date:** 2026-03-19
**Project:** ai_engineering
**Status:** raw
**Context:** Debugging a RAG pipeline stuck at 0.55 RAGAS scores after trying multiple embedding model swaps.

---

## The Struggle

My RAGAS scores were stuck around 0.55. I swapped out all-MiniLM for a larger sentence-transformers model and barely saw any change. I spent two days swapping models and could not move the needle.

## The Drill-Down

First I assumed the embedding model was the bottleneck — bigger model should mean better representations. When that didn't work, I asked what my chunking looked like. I was using fixed-size, 512 tokens, no overlap. Then I pushed further: so the embedding model is not the lever here?

## The Aha Moment

Fixed-size chunking cuts at character boundaries regardless of sentence or concept breaks. A better embedding model can't fix broken chunks. For domain-specific corpora, the embedding model is almost never the lever — chunk quality determines whether the right information is retrievable in the first place. I switched to paragraph-based chunking and RAGAS jumped from 0.55 to 0.71. Two hours vs two days.

## The One-Liner

> I spent two days swapping embedding models and barely moved the needle — switched to paragraph chunking in two hours and got a 28% jump.

---

*Raw insight captured from conversation. Run /polish-insight to generate LinkedIn post variations.*
