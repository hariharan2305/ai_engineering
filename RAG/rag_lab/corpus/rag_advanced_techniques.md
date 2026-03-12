# Advanced RAG Techniques

This document covers advanced techniques used to improve the quality and accuracy of Retrieval-Augmented Generation systems beyond basic dense retrieval.

## 1. Query Transformation

Query transformation techniques improve retrieval by rewriting or expanding the original user query before it hits the vector database.

### HyDE (Hypothetical Document Embeddings)

HyDE is a query transformation technique where the LLM generates a hypothetical answer to the query, and that hypothetical answer — rather than the original question — is used as the retrieval query. The core insight is that a hypothetical answer lives closer to the actual answer documents in embedding space than a short question does. Questions and answers often use different vocabulary and phrasing. By generating a document-like answer first, the query vector aligns better with stored chunk vectors.

HyDE adds one LLM call per query at retrieval time, which increases latency and cost. It performs best on factual, knowledge-dense corpora where the vocabulary gap between questions and documents is large.

### Multi-Query Retrieval

Multi-query retrieval generates multiple reformulations of the original query using an LLM, runs each reformulation as a separate retrieval, and then deduplicates and merges the results. This reduces sensitivity to the exact phrasing of the original query. A poorly phrased question might miss relevant chunks, but one of the reformulations is likely to retrieve them.

The trade-off is cost and latency: N reformulations means N retrieval calls. Typical values are 3 to 5 reformulations. Results from all retrievals are deduplicated before reranking or generation.

### Query Expansion

Query expansion adds related terms and synonyms to the original query to improve recall. Unlike multi-query, which runs separate retrievals, query expansion modifies the original query in place. It is less powerful than multi-query but also cheaper, as it still results in a single retrieval call.

## 2. Reranking

Reranking is a post-retrieval step that re-scores the initially retrieved chunks using a more precise but slower scoring model. The retriever fetches a candidate set (e.g., top-20), and the reranker reorders them before the top-K are passed to generation.

### Cross-Encoder Rerankers

Cross-encoders take the query and a candidate chunk together as a single input and produce a relevance score. Because they process query and document jointly, they capture fine-grained semantic relationships that bi-encoders miss. Cross-encoders cannot pre-compute document embeddings — every query-document pair must be scored at query time. This makes them too slow for first-stage retrieval over large corpora, but ideal for reranking a small candidate set of 20 to 50 documents.

Popular cross-encoder models include `cross-encoder/ms-marco-MiniLM-L-6-v2` from sentence-transformers.

### Bi-Encoder vs Cross-Encoder Trade-offs

Bi-encoders encode query and document independently, enabling pre-computation of document embeddings. This makes them fast for first-stage retrieval but less precise. Cross-encoders jointly encode query-document pairs, making them highly precise but too slow for large-scale retrieval. The standard production pattern combines both: bi-encoder for retrieval, cross-encoder for reranking.

## 3. Context Compression

Context compression reduces the amount of retrieved text passed to the LLM by removing irrelevant content from chunks before generation.

### Why Context Compression Matters

Retrieved chunks often contain only a few sentences relevant to the query, surrounded by unrelated content. Passing full chunks inflates the context window, increases cost, and introduces noise that can confuse the LLM. Context compression extracts only the relevant portions.

### LLMLingua

LLMLingua is a prompt compression technique that uses a small language model to score the importance of each token in the retrieved context. Low-importance tokens are dropped, reducing context length by up to 20x while preserving the key information. It is particularly useful when retrieved chunks are long and only partially relevant.

### Contextual Compression Retriever

LangChain provides a ContextualCompressionRetriever that wraps any base retriever and applies a compressor (LLM-based or extractive) to filter retrieved documents. The compressor can be an LLM that extracts relevant sentences, or a rule-based extractor.
