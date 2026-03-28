# Why Chunking Beats Embedding Swaps in a RAG Pipeline

**Date:** 2026-03-19
**Project:** ai_engineering
**Status:** raw
**Context:** Testing which RAG pipeline components actually move RAGAS scores — started with embedding model swaps, then tried fixing chunk boundaries.

---

## The Struggle

I spent two days swapping embedding models — going from MiniLM to a much larger model — expecting it to move the needle. Almost no change. That was frustrating. Then I changed how I set chunk boundaries to respect paragraph structure instead of using fixed-size splitting.

## The Drill-Down

First I tried the obvious thing: bigger model, better embeddings, better retrieval. That's the conventional wisdom. Two days, almost nothing moved. Then I asked what else could be different and tried fixing how chunks were cut — making them respect paragraph boundaries instead of slicing at a fixed character count.

## The Aha Moment

28% jump in RAGAS scores. Two hours of work. The chunks were the problem all along — when you split mid-sentence or mid-paragraph, the retriever pulls back broken context and the answer quality tanks, no matter how good the embedding model is. The embedding model can only work with what you give it. Garbage chunks in, garbage retrieval out.

## The One-Liner

> Your retrieval pipeline is only as good as your chunks — fix those before touching anything else.

---

*Raw insight captured from conversation. Run /polish-insight to generate LinkedIn post variations.*

---

## LinkedIn Post Variations

**Polished on:** 2026-03-19
**Voice samples used:** none

### Variation 1: Real experience / Impact story

Fixing chunk boundaries gave me a 28% RAGAS jump — more than two days of embedding model swaps ever did.

I've been stress-testing my RAG pipeline component by component, measuring what actually moves the scores.

First thing I tried: swap the embedding model. Went from MiniLM to a much larger model. Spent two days on it.

Almost no change.

Then I spent two hours changing how chunks are cut — respecting paragraph boundaries instead of slicing at a fixed character count.

28% improvement across RAGAS metrics.

The retriever was getting broken context all along. Mid-sentence cuts, mid-paragraph splits — no matter how good the embedding model, it can only work with what you give it. The embedding was fine. The input was broken.

If you're optimizing a RAG pipeline and you haven't looked at your chunk boundaries yet, look there first.

---

### Variation 2: Myth-buster

Better embedding models don't fix a broken RAG pipeline — better chunks do.

It's easy to believe otherwise. Embeddings are the mathematical heart of semantic search, so upgrading the model feels like the right lever to pull when retrieval quality is low.

I pulled that lever for two days. Swapped from a small model to a much larger one. The RAGAS scores barely moved.

Then I fixed my chunking strategy — stopped splitting at fixed character counts and started respecting paragraph boundaries. Two hours of work.

That gave me 28% across all RAGAS metrics.

The mental model shift: an embedding model maps text to a vector. If that text is a broken fragment — cut mid-sentence, missing its context — the vector it produces is noise. No model size fixes that.

Fix your chunks first. Then worry about the model.

---

### Variation 3: Assumption-flip

Two days of swapping embedding models, and my RAGAS scores barely moved — the problem was never the model.

The assumption made sense: bigger model, richer semantic representations, better similarity matching. That's how it's supposed to work.

What I hadn't looked at was how the chunks themselves were being cut. Fixed-size splitting doesn't care about sentence or paragraph boundaries — it just slices at a character count. So the retriever was fetching fragments. Broken context going into the LLM.

I switched to boundary-aware chunking — splitting at paragraph breaks instead of character counts. Two hours of work. 28% improvement across RAGAS metrics.

The thing I missed: the embedding model operates on whatever text you hand it. If that text is a mid-thought fragment, even a state-of-the-art model produces a noisy vector. The model isn't the bottleneck. The input is.

Now I check chunking first, before touching anything else in the pipeline.
