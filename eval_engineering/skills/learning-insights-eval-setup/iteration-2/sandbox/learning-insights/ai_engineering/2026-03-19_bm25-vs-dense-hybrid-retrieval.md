# BM25 vs Dense Retrieval and Why You'd Combine Them

**Date:** 2026-03-19
**Project:** ai_engineering
**Status:** raw
**Context:** Learning about retrieval strategies while working through the RAG lab curriculum.

---

## The Struggle

Trying to understand what BM25 actually is and how it differs from dense retrieval. The two approaches seem to solve the same problem — finding relevant documents — but in completely different ways.

## The Drill-Down

First I asked what the difference between BM25 and dense retrieval is. Then when that clicked, the natural follow-up was: so for a hybrid system you'd combine both?

## The Aha Moment

BM25 is keyword-based — it scores documents by term frequency and inverse document frequency. Dense retrieval uses embedding vectors to find semantically similar content regardless of exact word match. To get the best of both, you run both retrievers and merge results with RRF (Reciprocal Rank Fusion) or a weighted score. You get keyword precision plus semantic recall.

## The One-Liner

> BM25 gives you keyword precision, dense retrieval gives you semantic recall — hybrid with RRF gives you both.

---

*Raw insight captured from conversation. Run /polish-insight to generate LinkedIn post variations.*
