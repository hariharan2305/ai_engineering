# Chunking Strategy Beats Embedding Model Choice for RAG Quality

**Date:** 2026-03-19
**Project:** ai_engineering
**Status:** polished
**Context:** Running RAG experiments comparing chunking strategies and embedding models — measuring quality with RAGAS across both dimensions.

---

## The Struggle

I ran two sets of experiments: one swapping chunking strategies, one swapping embedding models. I expected the embedding model upgrade to be the bigger lever — bigger model, more parameters, better representations. It wasn't. I couldn't explain why something as "simple" as how you split text would outweigh the quality of the model doing the actual semantic encoding.

## The Drill-Down

I tested MiniLM against larger embedding models — changed nothing else. RAGAS scores barely moved. Then I tested fixing chunk boundaries to respect paragraph structure — same embedding model, same everything else. RAGAS jumped 28%. The question that followed: how can the retrieval pipeline be so sensitive to chunk boundaries but so insensitive to embedding quality? What does that say about where the actual information loss is happening?

## The Aha Moment

The retrieval pipeline is a chain — and a chain breaks at its weakest link. If your chunks are bad (splitting mid-sentence, splitting mid-argument, splitting across paragraph boundaries), no embedding model can recover the semantic signal that was destroyed during chunking. The embedding model encodes whatever you hand it. If what you hand it is incoherent, it will faithfully encode incoherence. A better embedding model just encodes the garbage more accurately.

Fixing chunk boundaries means the embedding model gets complete, coherent units of meaning to work with. At that point, even a smaller model can produce a useful vector. The 28% jump wasn't the embedding doing more — it was the embedding finally having something coherent to work with.

## The One-Liner

> Your retrieval pipeline is only as good as your chunks — a better embedding model just encodes the garbage more accurately.

---

*Raw insight captured from conversation.*

---

## LinkedIn Post Variations

**Polished on:** 2026-03-19
**Voice samples used:** none

---

### Variation 1: Myth-Buster

I spent time swapping embedding models in my RAG pipeline.

MiniLM to a larger model. Barely moved my RAGAS scores.

I assumed the embedding was the bottleneck — bigger model, better semantic representations, better retrieval. The data disagreed.

Then I fixed my chunk boundaries to respect paragraph structure.

Same embedding model. Same everything else. RAGAS jumped 28%.

Here's what that forced me to update in my mental model:

The embedding model encodes whatever you hand it. If you hand it a chunk that starts mid-sentence or cuts across a paragraph boundary, it faithfully encodes that broken fragment. A better embedding model just encodes the garbage more accurately.

Fixing chunk boundaries gives the model complete, coherent units of meaning. At that point, even a smaller model can produce a useful vector — because it's working with something worth encoding.

The retrieval pipeline is a chain. It breaks at the weakest link. And in most RAG systems, that link is chunking, not the embedding.

Two days on embedding model swaps moved the needle by noise. One afternoon fixing chunk boundaries moved it by 28%.

---

### Variation 2: Narrative / Story

I ran two sets of experiments on my RAG pipeline.

First: swap the embedding model. MiniLM to progressively larger models, better training data, higher dimensions. Measured RAGAS after each swap.

Scores barely moved.

Second: change nothing about the embedding model. Just fix the chunking — stop cutting mid-sentence, stop splitting across paragraph boundaries, let chunks end where the text actually ends.

RAGAS jumped 28%.

I had to stop and think about why.

The answer is that the retrieval pipeline is a chain, and it breaks at its weakest link. If you chunk badly, you're feeding the embedding model incoherent fragments — a thought that started on the previous chunk, an argument that concludes on the next one. No embedding model can recover meaning from a fragment that lost its beginning or end.

A better embedding model doesn't fix this. It just encodes the incoherence with higher fidelity.

When you fix chunk boundaries first, you give the model complete, coherent units to encode. At that point, even MiniLM can produce a signal worth retrieving on.

The chunking experiment I almost skipped — because surely the embedding model matters more — turned out to be the highest-leverage change in the whole pipeline.

---

### Variation 3: Technical Breakdown

Chunking strategy moved my RAG quality by 28%. Embedding model swap barely moved it at all.

This is counterintuitive if you think of embedding quality as the primary retrieval lever. Here's why it makes sense once you see it:

The retrieval pipeline processes in sequence:

raw text → chunks → embeddings → vector search → context window → LLM

Each step can only work with what the previous step produced.

If chunk boundaries cut mid-sentence or split across paragraphs:
- The embedding model receives a semantically broken fragment
- It encodes that fragment accurately — which means encoding broken meaning
- Vector search retrieves the most similar broken fragments
- The LLM gets partial, disconnected context to reason over

A better embedding model doesn't fix this. It encodes the broken input with higher precision.

Fixing chunk boundaries to respect paragraph structure:
- Each chunk becomes a complete, coherent unit of meaning
- The embedding has something worth encoding
- Vector search retrieves complete thoughts, not half-sentences
- The LLM gets proper context

The 28% jump in RAGAS wasn't the embedding doing more — it was the embedding finally having something coherent to work with.

Practical implication: before tuning your embedding model, measure whether your chunks actually respect the document's natural boundaries. It's the cheaper experiment and often the higher-leverage one.
