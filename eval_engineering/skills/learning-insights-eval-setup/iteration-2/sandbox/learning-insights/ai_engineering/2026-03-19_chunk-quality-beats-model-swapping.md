# Why chunk quality matters more than the embedding model

**Date:** 2026-03-19
**Project:** ai_engineering
**Status:** raw
**Context:** Trying to improve RAG pipeline RAGAS scores that were stuck at 0.55 by experimenting with different embedding models.

---

## The Struggle

My RAGAS scores were stuck around 0.55 and I'd been trying to improve them. I swapped out all-MiniLM for a larger sentence-transformers model and barely saw any change. Two days of model-swapping and barely moving the needle.

## The Drill-Down

First I figured the embedding model was the lever — bigger model, better scores. When that didn't work I asked whether the embedding model was just not the right thing to change. That's when chunking came up. I was on fixed-size 512-token chunks with no overlap, which I'd assumed was pretty standard.

## The Aha Moment

Fixed-size chunking cuts at character count boundaries — it doesn't care if it's splitting mid-sentence or mid-concept. When the retriever fetches those chunks, it's pulling in fragments that lack enough context for the LLM to reason from. A better embedding model can't fix broken chunks. Changing the chunking to respect paragraph breaks jumped RAGAS scores from 0.55 to 0.71 — a 28% improvement — and it took two hours, not two days.

## The One-Liner

> Two days of swapping embedding models barely moved the needle. Two hours fixing chunk boundaries gave me a 28% jump.

---

*Raw insight captured from conversation. Run /polish-insight to generate LinkedIn post variations.*
