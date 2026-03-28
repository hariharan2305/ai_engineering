# Chunking Moves RAG Scores More Than Embedding Model Swaps

**Date:** 2026-03-19
**Project:** ai_engineering
**Status:** raw
**Context:** Testing RAG pipeline components one at a time to see what actually moves RAGAS scores.

---

## The Struggle

I spent two days swapping embedding models — went from MiniLM to a much larger model — and got almost nothing. No meaningful change in scores. Then I fixed chunk boundaries to respect paragraph structure instead of fixed-size splitting, and the numbers moved hard.

## The Drill-Down

I started with embedding models because that felt like the obvious lever. Bigger model, better representations, better retrieval — that was my assumption. Two days in and the RAGAS scores barely moved. Then I switched to chunking, fixed the boundaries to follow paragraphs instead of cutting at fixed character counts, and ran the eval again.

## The Aha Moment

28% jump in RAGAS scores. Two hours on chunking. Two days on embedding models gave me nothing. The chunks are what actually get retrieved — if they're cut badly, no embedding model can rescue broken context. Fix the chunks first, then worry about the model.

## The One-Liner

> Your retrieval pipeline is only as good as your chunks — fix those before touching anything else.

---

*Raw insight captured from conversation. Run /polish-insight to generate LinkedIn post variations.*

---

## LinkedIn Post Variations

**Polished on:** 2026-03-19
**Voice samples used:** none

---

### Variation 1: Real experience / Impact story

Two days on embedding models. Almost no change in RAGAS scores.

Two hours on chunking. 28% jump.

I was testing my RAG pipeline component by component — swapping in a much larger embedding model instead of MiniLM, assuming better representations would mean better retrieval. The eval barely moved.

Then I switched to the chunks. Fixed the boundaries so they followed paragraph structure instead of cutting at fixed character counts. Ran the eval again.

28%.

The lesson is uncomfortable if you've been chasing model upgrades: the retrieval pipeline can only work with what it gets. If chunks cut across paragraph boundaries, they deliver broken context to the retriever. No embedding model rescues that.

Fix the chunks before you touch anything else.

---

### Variation 2: Myth-buster

The common advice when RAG retrieval quality is low: upgrade your embedding model.

It's not wrong exactly. Better embeddings do help. But there's something that helps more, costs less, and gets skipped — because it's not as interesting to talk about.

Chunk boundaries.

I ran the experiment. Two days testing a much larger embedding model vs MiniLM. RAGAS scores barely moved.

Then I spent two hours fixing how chunks were cut — respecting paragraph structure instead of splitting at fixed character counts.

28% improvement.

The reason: RAGAS is measuring whether the retrieved context actually answers the question. A chunk that cuts mid-paragraph retrieves broken context. The embedding model represents that broken context faithfully — that's the problem. Garbage in, garbage out, regardless of how well you encode the garbage.

The corrected mental model: chunk quality determines retrieval ceiling. Embedding quality determines how close you get to that ceiling. You need both, but you need the ceiling first.

---

### Variation 3: Assumption-flip

I was confident the bottleneck was the embedding model.

Smaller model, smaller representation space, lower retrieval quality — that logic seemed airtight. So I spent two days swapping models, running evals, expecting to see the scores move.

Almost nothing.

Then I fixed the chunks. Not a new algorithm — just changed how boundaries were drawn, so they followed paragraph structure instead of cutting at fixed character counts.

28% jump in RAGAS scores. Two hours of work.

The original assumption was wrong in a specific way: I was assuming the embedding model was the quality gate. It's not. The chunk is. The embedding model compresses whatever you give it — if the chunk is broken, the embedding faithfully represents a broken chunk.

Better model doesn't fix that. Better boundaries do.

New mental model: in a RAG pipeline, quality flows from the corpus inward. Chunks first, then embeddings, then retrieval strategy. The order matters.
