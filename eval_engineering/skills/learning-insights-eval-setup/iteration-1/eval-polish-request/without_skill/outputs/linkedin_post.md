Everyone talks about which embedding model to use for RAG.

I spent weeks running controlled experiments — swapping MiniLM for larger, more capable models — and my RAGAS scores barely moved.

Then I fixed my chunk boundaries to respect paragraph structure instead of splitting at arbitrary character limits.

28% jump across faithfulness, answer relevancy, and answer similarity.

Same embedding model. Same retrieval logic. Same LLM. Just better chunks.

The lesson that actually stuck: your retrieval pipeline is only as good as your chunks. An embedding model can't recover meaning that was shredded at the chunking stage. Garbage in, garbage out — but the garbage happens earlier in the pipeline than most people look.

If you're tuning RAG quality right now, audit your chunk boundaries before touching your embedding model. You'll likely find more signal there.

#RAG #LLM #GenAI #MachineLearning #NLP
