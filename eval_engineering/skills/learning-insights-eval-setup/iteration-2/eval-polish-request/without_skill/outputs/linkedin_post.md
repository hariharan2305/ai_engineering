I spent two days swapping embedding models in my RAG pipeline. MiniLM to a larger model. Almost no change in RAGAS scores.

Then I spent two hours fixing chunk boundaries so they respect paragraph breaks instead of cutting mid-sentence.

28% jump.

The lesson is uncomfortable if you've been obsessing over model leaderboards: **your chunking strategy matters more than your embedding model — at least until the chunks are right.**

Garbage in, garbage out. If you're feeding a powerful embedding model broken, mid-thought text fragments, you're wasting its capacity. The model can't make meaning from half a sentence.

Fix the data representation first. Then optimize the model.

If your RAG pipeline is underperforming, before you swap models, ask: are my chunks actually coherent units of meaning?

That question is worth two hours of your time.

#RAG #LLM #GenAI #MLEngineering #VectorSearch
