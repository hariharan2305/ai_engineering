import uuid

from .base import Document, Chunk


class FixedSizeChunker:
    """
    Splits documents into fixed-size character chunks with overlap.
    Baseline strategy — the simplest possible chunker.

    Experiment knobs:
      - chunk_size: smaller = more precise retrieval, more chunks to store
      - overlap: higher = less information loss at boundaries, more redundancy
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    doc_id=doc.id,
                    chunk_index=idx,
                    metadata={**doc.metadata, "start_char": start, "end_char": end},
                ))
                idx += 1

            if end == len(text):
                break

            start = end - self.overlap

        return chunks


class RecursiveChunker:
    """
    Splits documents using LangChain's RecursiveCharacterTextSplitter.

    Unlike FixedSizeChunker (which blindly cuts every N characters),
    this tries a hierarchy of separators in priority order:
        \n\n (paragraphs) → \n (lines) → . ! ? (sentences) → space (words) → characters

    It only falls back to a finer separator when the current piece still exceeds
    chunk_size. Result: chunks that respect natural text boundaries and rarely
    cut mid-sentence.

    Same chunk_size and overlap as the baseline — the only variable is the
    splitting strategy itself.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            # Hierarchy of separators — tries each in order, recurses on the next
            # only if the resulting piece is still too large
            separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        )

    def chunk(self, doc: Document) -> list[Chunk]:
        pieces = self.splitter.split_text(doc.text)
        chunks = []
        for idx, text in enumerate(pieces):
            text = text.strip()
            if text:
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    text=text,
                    doc_id=doc.id,
                    chunk_index=idx,
                    metadata={**doc.metadata, "chunker": "recursive"},
                ))
        return chunks


class SemanticChunker:
    """
    Splits documents using LangChain's SemanticChunker.

    Unlike character-based strategies, this embeds each sentence and finds points
    where cosine similarity between adjacent sentences drops sharply — a topic shift.
    It splits at those breakpoints, so each chunk contains one coherent idea.

    Under the hood (what SemanticChunker does):
      1. Split text into sentences
      2. Embed each sentence using the provided embedding model
      3. Compute cosine distance between every pair of adjacent sentence embeddings
      4. Find breakpoints where distance exceeds a threshold (percentile-based by default)
      5. Group sentences between breakpoints into a single chunk

    Uses our local all-MiniLM-L6-v2 model via a thin LangChain adapter — no API cost.
    No chunk_size parameter: chunk boundaries are driven by meaning, not character count.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", breakpoint_threshold_type: str = "percentile"):
        from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker
        from langchain_huggingface import HuggingFaceEmbeddings

        # ── PREFERRED: Use LangChain's native HuggingFace integration ────────────
        # langchain-huggingface provides HuggingFaceEmbeddings which wraps
        # sentence-transformers and already implements the LangChain Embeddings
        # interface (embed_documents + embed_query). No custom adapter needed.
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # ── REFERENCE: Manual adapter pattern (kept for learning purposes) ───────
        # Use this pattern when a third-party library requires an interface (e.g.
        # LangChain's Embeddings ABC) but you only have a library that does the same
        # thing under a different method signature.
        #
        # WHEN TO USE THE ADAPTER PATTERN:
        #   - No official integration exists between library A and library B
        #   - You want to plug in a custom/internal embedder into a framework
        #   - You're prototyping and don't want to add a new dependency just yet
        #
        # HOW TO BUILD ONE:
        #   1. Find the interface the framework requires (read its source or docs)
        #   2. Subclass that interface in your adapter
        #   3. In each required method, delegate to your actual implementation
        #   4. Keep it private (_ClassName) and co-located — it's a wiring detail, not a feature
        #
        # from langchain_core.embeddings import Embeddings
        # from sentence_transformers import SentenceTransformer
        #
        # class _STAdapter(Embeddings):
        #     def __init__(self, model_name: str):
        #         self.model = SentenceTransformer(model_name)
        #
        #     def embed_documents(self, texts: list[str]) -> list[list[float]]:
        #         return self.model.encode(texts, show_progress_bar=False).tolist()
        #
        #     def embed_query(self, text: str) -> list[float]:
        #         return self.model.encode([text], show_progress_bar=False)[0].tolist()
        #
        # embeddings = _STAdapter(model_name)
        # ─────────────────────────────────────────────────────────────────────────

        # breakpoint_threshold_type options: "percentile" (default), "standard_deviation", "interquartile"
        # "percentile" splits at the top-N% most dissimilar adjacent sentence pairs
        self.splitter = LCSemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
        )

    def chunk(self, doc: Document) -> list[Chunk]:
        pieces = self.splitter.split_text(doc.text)
        chunks = []
        for idx, text in enumerate(pieces):
            text = text.strip()
            if text:
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    text=text,
                    doc_id=doc.id,
                    chunk_index=idx,
                    metadata={**doc.metadata, "chunker": "semantic"},
                ))
        return chunks


class MarkdownChunker:
    """
    Splits Markdown documents using LangChain's MarkdownHeaderTextSplitter.

    Instead of splitting on character count, this splits on Markdown headers (#, ##, ###).
    Each chunk contains the content under one header section, with the header hierarchy
    stored in chunk metadata. This preserves document structure — a section about
    "## Reranking" stays together as one chunk rather than being cut mid-paragraph.

    Under the hood: scans for header tokens, groups all lines between headers into one
    chunk, attaches the header path as metadata (e.g. {"h1": "Advanced RAG", "h2": "Reranking"}).
    strip_headers=False keeps the header text inside the chunk for better retrieval context.
    """

    def __init__(self):
        from langchain_text_splitters import MarkdownHeaderTextSplitter
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            strip_headers=False,  # keep header text in chunk — helps embedding capture section topic
        )

    def chunk(self, doc: Document) -> list[Chunk]:
        # Returns LangChain Document objects with header info in .metadata
        pieces = self.splitter.split_text(doc.text)
        chunks = []
        for idx, piece in enumerate(pieces):
            text = piece.page_content.strip()
            if text:
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    text=text,
                    doc_id=doc.id,
                    chunk_index=idx,
                    # header hierarchy stored alongside doc metadata for traceability
                    metadata={**doc.metadata, "chunker": "markdown", **piece.metadata},
                ))
        return chunks


class HTMLChunker:
    """
    Splits HTML documents using LangChain's HTMLHeaderTextSplitter.

    Splits on <h1>, <h2>, <h3> tags — the semantic structure of the HTML document.
    Each chunk contains the content under one header section with the header path
    stored in chunk metadata. Requires the `lxml` package for HTML parsing.

    Under the hood: parses the HTML DOM, finds header elements, groups content
    between headers into chunks, attaches header text as metadata. Works on raw
    HTML strings — no need to strip tags before chunking.
    """

    def __init__(self):
        from langchain_text_splitters import HTMLHeaderTextSplitter
        self.splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=[
                ("h1", "h1"),
                ("h2", "h2"),
                ("h3", "h3"),
            ],
        )

    def chunk(self, doc: Document) -> list[Chunk]:
        # Returns LangChain Document objects with header info in .metadata
        pieces = self.splitter.split_text(doc.text)
        chunks = []
        for idx, piece in enumerate(pieces):
            text = piece.page_content.strip()
            if text:
                chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    text=text,
                    doc_id=doc.id,
                    chunk_index=idx,
                    metadata={**doc.metadata, "chunker": "html", **piece.metadata},
                ))
        return chunks


class LISentenceSplitter:
    """
    LlamaIndex's SentenceSplitter — the direct counterpart to LangChain's RecursiveChunker.

    Both try to respect natural text boundaries instead of blindly cutting on character count.
    SentenceSplitter's priority order:
        sentence boundaries → paragraph breaks → word boundaries → characters
    (vs RecursiveCharacterTextSplitter's: \\n\\n → \\n → .!? → space → chars)

    Same chunk_size and overlap as the baseline — only the splitting strategy changes.
    This makes it a clean one-variable comparison against exp_02_recursive_chunking.py.

    LlamaIndex concepts introduced here:
      - Document (llama_index.core): LI's document type — takes text + metadata dict
      - from_defaults(): LI's standard constructor pattern — used consistently across
        all LI components so you can always find the knobs in one place
      - get_nodes_from_documents(): LI's processing method, equivalent to our .chunk()
      - TextNode: LI's chunk type — has .text, .metadata, .node_id (UUID auto-generated)

    Knobs:
      - chunk_size: target chunk size in tokens (LI counts tokens, not chars by default)
      - chunk_overlap: overlap between adjacent chunks
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        # llama_index.core.node_parser — all LI chunking strategies live here
        from llama_index.core.node_parser import SentenceSplitter

        # from_defaults() is LI's standard constructor — equivalent to __init__ with defaults
        # but used consistently across all LI components for uniform API surface
        self.splitter = SentenceSplitter.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, doc: Document) -> list[Chunk]:
        # Convert our Document dataclass to LlamaIndex's Document type
        # LI's Document wraps text + metadata; it's consumed by all LI node parsers
        from llama_index.core import Document as LIDocument

        li_doc = LIDocument(text=doc.text, metadata=doc.metadata)

        # get_nodes_from_documents() — LI's processing entry point
        # Takes a list of Documents, returns a list of TextNode objects
        nodes = self.splitter.get_nodes_from_documents([li_doc])

        chunks = []
        for idx, node in enumerate(nodes):
            text = node.text.strip()
            if text:
                chunks.append(Chunk(
                    id=node.node_id,  # LI auto-generates a UUID per node — reuse it as our id
                    text=text,
                    doc_id=doc.id,
                    chunk_index=idx,
                    metadata={**doc.metadata, "chunker": "li_sentence_splitter"},
                ))
        return chunks


class LISemanticChunker:
    """
    LlamaIndex's SemanticSplitterNodeParser.

    Same concept as LangChain's SemanticChunker (exp_02_semantic_chunking.py):
    embed sentences, split where cosine similarity between adjacent groups drops.
    This lets you compare the two implementations directly on the same corpus.

    LlamaIndex-specific differences vs LangChain's SemanticChunker:
      - Uses HuggingFaceEmbedding (singular) vs LangChain's HuggingFaceEmbeddings (plural)
      - buffer_size: number of sentences to group before comparing (LI-specific knob)
        buffer_size=1 → compare individual sentences (most granular)
        buffer_size=2 → group 2 sentences, then compare groups
      - breakpoint_percentile_threshold: equivalent to LangChain's percentile threshold
        95 → split at top 5% most dissimilar adjacent groups (fewer, larger chunks)
        80 → more splits (more, smaller chunks)
      - Takes embed_model= instead of embeddings= (LI naming convention)

    LlamaIndex concepts introduced here:
      - HuggingFaceEmbedding from llama_index.embeddings.huggingface
        (note: singular "Embedding", vs LangChain's plural "Embeddings")
      - SemanticSplitterNodeParser from llama_index.core.node_parser
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        breakpoint_percentile_threshold: int = 95,
    ):
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        # llama_index.embeddings.huggingface — separate install: llama-index-embeddings-huggingface
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        # LI's HuggingFaceEmbedding wraps sentence-transformers under the hood,
        # same as LangChain's HuggingFaceEmbeddings — just different API surface.
        embed_model = HuggingFaceEmbedding(model_name=model_name)

        self.splitter = SemanticSplitterNodeParser(
            buffer_size=1,                                          # group size before similarity comparison
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            embed_model=embed_model,                                # LI uses embed_model=, not embeddings=
        )

    def chunk(self, doc: Document) -> list[Chunk]:
        from llama_index.core import Document as LIDocument

        li_doc = LIDocument(text=doc.text, metadata=doc.metadata)
        nodes = self.splitter.get_nodes_from_documents([li_doc])

        chunks = []
        for idx, node in enumerate(nodes):
            text = node.text.strip()
            if text:
                chunks.append(Chunk(
                    id=node.node_id,
                    text=text,
                    doc_id=doc.id,
                    chunk_index=idx,
                    metadata={**doc.metadata, "chunker": "li_semantic"},
                ))
        return chunks


class LITokenSplitter:
    """
    LlamaIndex's TokenTextSplitter — chunks by token count, not character count.

    This is the key production distinction vs all other chunkers in the lab:

        FixedSizeChunker, RecursiveChunker, SentenceSplitter
            → chunk_size measured in CHARACTERS

        TokenTextSplitter
            → chunk_size measured in TOKENS (via tiktoken, OpenAI's tokenizer)

    Why it matters in production:
      LLMs have context windows measured in tokens, not characters. English
      prose averages ~4 chars/token, but code can be 1 char = 1 token, and
      non-English text can be 1 char = multiple tokens. If you chunk by
      character count, you cannot predict how many tokens each chunk occupies.
      Token-based chunking gives you precise control over the LLM context budget.

    Practical effect in this experiment:
      chunk_size=512 characters ≈ ~128 tokens for English prose.
      chunk_size=512 TOKENS ≈ ~2048 characters for English prose.
      Same number, very different result — chunks will be ~4x larger,
      more coherent, potentially improving answer quality at the cost of
      fewer, broader chunks retrieved per query.

    LlamaIndex uses tiktoken (OpenAI's tokenizer) by default, which means
    token boundaries are computed with GPT-style BPE encoding. This is fine
    even when using a non-OpenAI embedder — token counting is only for size
    control, not for generating embeddings.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        # llama_index.core.node_parser — same module as SentenceSplitter
        from llama_index.core.node_parser import TokenTextSplitter

        # chunk_size here = tokens (via tiktoken), not characters
        # chunk_overlap here = tokens of overlap between adjacent chunks
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, doc: Document) -> list[Chunk]:
        from llama_index.core import Document as LIDocument

        li_doc = LIDocument(text=doc.text, metadata=doc.metadata)
        nodes = self.splitter.get_nodes_from_documents([li_doc])

        chunks = []
        for idx, node in enumerate(nodes):
            text = node.text.strip()
            if text:
                chunks.append(Chunk(
                    id=node.node_id,
                    text=text,
                    doc_id=doc.id,
                    chunk_index=idx,
                    metadata={**doc.metadata, "chunker": "li_token_splitter"},
                ))
        return chunks


class LICodeSplitter:
    """
    LlamaIndex's CodeSplitter — AST-aware chunking for source code.

    Uses tree-sitter to parse the source code into an Abstract Syntax Tree
    (AST), then splits on syntactic boundaries (function definitions, class
    definitions, top-level blocks) rather than line counts or character counts.

    Why AST-aware splitting matters for code RAG:
      A naive character or line splitter will routinely cut through the middle
      of a function body, creating chunks that are syntactically invalid and
      semantically meaningless. A retriever matching "how does retry work?"
      against half a function body gets poor results. CodeSplitter respects
      Python's actual structure: a function definition stays together, a class
      definition stays together, and docstrings remain attached to the code
      they document.

    Knobs — note: different from all other chunkers:
      chunk_lines:         Target lines per chunk (not chars, not tokens).
      chunk_lines_overlap: Lines of overlap between adjacent chunks.
      max_chars:           Hard character ceiling per chunk — safety net for
                           very large functions that exceed chunk_lines.
      language:            tree-sitter language name (e.g. "python", "javascript",
                           "typescript", "go", "rust", "java").

    Dependencies:
      Requires tree-sitter and tree-sitter-language-pack (bundled grammars).
      Both are in pyproject.toml. Without them, the import will fail.
    """

    def __init__(
        self,
        language: str = "python",
        chunk_lines: int = 40,
        chunk_lines_overlap: int = 10,
        max_chars: int = 1500,
    ):
        # llama_index.core.node_parser — same import namespace as all LI parsers
        from llama_index.core.node_parser import CodeSplitter

        self.splitter = CodeSplitter(
            language=language,               # tree-sitter grammar to use
            chunk_lines=chunk_lines,         # target lines per chunk
            chunk_lines_overlap=chunk_lines_overlap,  # overlap in lines
            max_chars=max_chars,             # hard char ceiling per chunk
        )
        self.language = language

    def chunk(self, doc: Document) -> list[Chunk]:
        from llama_index.core import Document as LIDocument

        li_doc = LIDocument(text=doc.text, metadata=doc.metadata)
        nodes = self.splitter.get_nodes_from_documents([li_doc])

        chunks = []
        for idx, node in enumerate(nodes):
            text = node.text.strip()
            if text:
                chunks.append(Chunk(
                    id=node.node_id,
                    text=text,
                    doc_id=doc.id,
                    chunk_index=idx,
                    metadata={
                        **doc.metadata,
                        "chunker": "li_code_splitter",
                        "language": self.language,
                    },
                ))
        return chunks


def chunk_documents(docs: list[Document], chunker) -> list[Chunk]:
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunker.chunk(doc))
    return all_chunks
