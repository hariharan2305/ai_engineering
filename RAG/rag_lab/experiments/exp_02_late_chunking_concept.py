"""
Late Chunking — Concept Reference

NOT a runnable experiment. Use this file as a reference before building exp_09_* or later.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT IS LATE CHUNKING?
━━━━━━━━━━━━━━━━━━━━━━

Every other experiment in this lab follows the same two-step order:
    1. SPLIT the document into chunks
    2. EMBED each chunk independently

Late chunking flips that order:
    1. EMBED the entire document (get token-level embeddings for ALL tokens)
    2. SPLIT those embeddings by chunk boundary into pooled chunk vectors

The key insight: in standard chunking, a chunk's embedding only "sees" the
text within that chunk. A sentence in the middle of a document has no idea
what came before or after it. Late chunking solves this by running the full
document through the encoder FIRST — every token's embedding carries
attention-based context from the entire document — then splitting afterward.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONCRETE EXAMPLE
━━━━━━━━━━━━━━━━

Document:
    "LLMs predict the next token given a context window.
     The context window is the key bottleneck.
     Transformers process the entire context in parallel."

With standard chunking, if "context window" is split across two chunks:
    Chunk 1 embedding: only sees "LLMs predict the next token given a context window."
    Chunk 2 embedding: only sees "The context window is the key bottleneck."

With late chunking:
    The encoder sees all three sentences. The embedding for "context window"
    in chunk 1 already carries signal from chunk 2 and chunk 3 via attention.
    After pooling per chunk boundary, each chunk vector is richer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW IT WORKS — STEP BY STEP
━━━━━━━━━━━━━━━━━━━━━━━━━━━

Standard pipeline:
    doc_text
        → text_splitter.split(doc_text)           → [chunk_1, chunk_2, ..., chunk_n]
        → [encoder(chunk_i) for chunk_i]           → [embed_1, embed_2, ..., embed_n]
        → store in vector DB

Late chunking pipeline:
    doc_text
        → text_splitter.get_boundaries(doc_text)  → [(0, 120), (120, 310), ...]  # char spans
        → encoder(doc_text, output_all_tokens=True) → token_embeddings (shape: [seq_len, dim])
        → map char spans → token spans             → [(tok_0, tok_18), (tok_18, tok_47), ...]
        → mean_pool(token_embeddings[tok_i:tok_j]) → [embed_1, embed_2, ..., embed_n]
        → store in vector DB

The stored text per chunk is still the original chunk text — only the VECTOR
representation is different. Retrieval and generation are unchanged.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PSEUDOCODE
━━━━━━━━━━

# Step 1: decide chunk boundaries (WITHOUT embedding yet)
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
chunk_texts = splitter.split_text(document_text)
# Note: overlap=0 for late chunking — boundaries must be non-overlapping
# so token span pooling covers the document exactly once

# Step 2: embed the FULL document, get all token embeddings
# Requires a model that supports long sequences AND returns token-level outputs
# sentence-transformers (all-MiniLM-L6-v2) has a 512-token max → NOT suitable
# Use Jina v3 (8192 tokens) or BGE-M3 (8192 tokens) instead
model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
token_embeddings = model.encode(
    document_text,
    output_value="token_embeddings",   # NOT the default CLS/mean pooled vector
    convert_to_numpy=True,
)
# token_embeddings.shape == (num_tokens_in_doc, embedding_dim)

# Step 3: map each chunk's character span to token indices
tokenizer = model.tokenizer
token_offsets = tokenizer(document_text, return_offsets_mapping=True)["offset_mapping"]
# token_offsets[i] = (char_start, char_end) for token i

# Step 4: for each chunk, find its token span and mean-pool
chunk_embeddings = []
char_pos = 0
for chunk_text in chunk_texts:
    chunk_start = char_pos
    chunk_end = char_pos + len(chunk_text)
    # find which token indices fall within [chunk_start, chunk_end)
    token_mask = [
        chunk_start <= tok_start < chunk_end
        for tok_start, tok_end in token_offsets
    ]
    chunk_token_embeddings = token_embeddings[token_mask]
    # mean pool → single vector for this chunk
    chunk_vector = chunk_token_embeddings.mean(axis=0)
    chunk_embeddings.append(chunk_vector)
    char_pos = chunk_end

# chunk_embeddings[i] is now a contextually-enriched vector for chunk_texts[i]
# Store (chunk_texts[i], chunk_embeddings[i]) in the vector DB as usual

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HARD CONSTRAINTS — READ BEFORE IMPLEMENTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. REQUIRES A LONG-CONTEXT EMBEDDING MODEL
   The full document must fit in the encoder's context window in a single pass.
   All-MiniLM-L6-v2 (used in this lab): 512 token max → CANNOT use for late chunking.
   You need a model with a large context window:
     - jinaai/jina-embeddings-v3 — 8192 tokens, native late chunking support
     - BAAI/bge-m3 — 8192 tokens
     - nomic-ai/nomic-embed-text-v1.5 — 8192 tokens
   Documents longer than the model's context window must be handled separately
   (split into model-window-sized segments, late chunk each segment independently).

2. NO OVERLAP IN CHUNK BOUNDARIES
   Standard chunking uses overlap (e.g. overlap=50) to avoid losing context at
   boundaries. Late chunking doesn't need this — context is preserved via attention
   across the full document, not by duplicating text. Use overlap=0.
   Overlapping boundaries would mean the same tokens contribute to two chunk
   vectors, double-counting their signal.

3. INGESTION IS SLOWER
   Running the full document through the encoder (not individual chunks) means
   one large forward pass per document instead of N small passes. For a 4000-token
   document with 512-token chunks: standard = 8 small passes, late = 1 large pass.
   In practice, large models on long sequences can be slower due to the O(n^2)
   attention cost.

4. CHROMADB STORES PRE-COMPUTED VECTORS — COMPATIBLE
   You generate chunk vectors yourself (step 4 above) and pass them directly to
   ChromaDB. This lab's vectordb.add_chunks() accepts pre-computed embeddings,
   so the storage layer requires no changes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHEN TO USE LATE CHUNKING
━━━━━━━━━━━━━━━━━━━━━━━━━

Good fit:
  - Documents where pronouns, abbreviations, or concepts refer back/forward
    (e.g. legal docs: "the aforementioned clause", "pursuant to section 3")
  - Technical docs where a term is defined once and used throughout
  - Narrative text where context builds across paragraphs
  - Documents shorter than the embedding model's context window

Poor fit:
  - Very long documents that exceed the model's context window
    (you lose the main benefit — full-document attention)
  - Corpora of short, self-contained chunks (tweets, product descriptions)
    (no cross-chunk context to preserve — late chunking adds cost with no benefit)
  - High-throughput ingestion pipelines where latency is critical
    (one large forward pass can be slower than N small parallel ones on a GPU)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LATE CHUNKING vs CONTEXTUAL ENRICHMENT (exp_02_contextual_enrichment.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Both techniques solve the same problem: chunks embedded in isolation lose
document-level context. But they solve it differently:

  Contextual Enrichment:
    - Uses an LLM to generate a context description, prepended as TEXT
    - Cost: 1 LLM call per chunk at ingestion time
    - Works with ANY embedding model (context is just prepended text)
    - Result: richer text → richer embedding

  Late Chunking:
    - Uses the embedding model's own attention to capture cross-chunk context
    - Cost: one large encoder forward pass per document (no LLM call)
    - Requires a long-context embedding model
    - Result: same text → richer embedding via attention

Both improve retrieval. Contextual enrichment is easier to implement and
works with your existing embedder. Late chunking is more principled and
cheaper at inference time (no LLM call) but requires model changes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REFERENCES
━━━━━━━━━━

[1] Günther et al. (2024) — "Late Chunking: Contextual Chunk Embeddings Using
    Long-Context Embedding Models"
    https://arxiv.org/abs/2409.04701
    The original paper. Introduces the term, the method, and ablation studies.

[2] Jina AI Blog (2024) — "Late Chunking in Long-Context Embedding Models"
    https://jina.ai/news/late-chunking-in-long-context-embedding-models/
    Practical walkthrough with code. Shows the token-offset mapping approach.
    Their jina-embeddings-v3 model has native late chunking support built in.

[3] Anthropic (2024) — "Contextual Retrieval"
    https://www.anthropic.com/news/contextual-retrieval
    The LLM-based alternative (implemented in exp_02_contextual_enrichment.py).
    Good comparison point for when to prefer each approach.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TO IMPLEMENT THIS AS A REAL EXPERIMENT (exp_09_late_chunking.py):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Add to pyproject.toml:
       "llama-index-embeddings-jinaai>=0.2.0"
   Or use sentence-transformers directly with jinaai/jina-embeddings-v3.

2. Replace SentenceTransformerEmbedder with a long-context embedder.

3. In ingest():
   - Get chunk boundaries ONLY (no embedding yet)
   - Encode full document with output_value="token_embeddings"
   - Map char spans to token spans
   - Mean-pool token embeddings per chunk
   - Call vectordb.add_chunks() with pre-computed vectors

4. Keep retrieval and generation identical to baseline.

5. Compare answer_similarity against:
   - exp_01_baseline (standard MiniLM embeddings)
   - exp_02_contextual_enrichment (same problem, different approach)
"""

# This file is intentionally not executable.
# Remove this line and implement main() when ready to run the experiment.
raise NotImplementedError(
    "This is a concept reference file. "
    "Implement the experiment following the pseudocode above. "
    "See exp_02_contextual_enrichment.py for a working enrichment alternative."
)
