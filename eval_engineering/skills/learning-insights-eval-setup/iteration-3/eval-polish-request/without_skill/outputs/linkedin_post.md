I spent two days swapping embedding models in my RAG pipeline. MiniLM to a larger model. Almost no change in RAGAS scores.

Then I spent two hours fixing chunk boundaries to respect paragraph breaks instead of cutting mid-sentence.

28% jump.

The lesson: **retrieval quality is upstream of everything**. A better embedding model cannot rescue a pipeline that's feeding it broken context. Garbage in, garbage out — no matter how sophisticated the model.

If your RAG pipeline isn't performing, don't reach for a bigger model first. Ask where the context is breaking down. Chunking strategy is unglamorous work, but it's where the real leverage is.

Fix the data before you upgrade the model.

#RAG #LLM #GenAI #MachineLearning #MLEngineering
