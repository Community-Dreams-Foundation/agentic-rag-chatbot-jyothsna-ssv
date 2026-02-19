# Architecture Overview

This document describes the architecture of the web-based RAG chatbot, covering document ingestion, retrieval with citations, and selective memory persistence.

---

## Document Ingestion (Parse → Chunk → Index)

### Parsing

The system supports three file formats:

- **Plain text (`.txt`, `.md`)**: Direct UTF-8 text extraction with error replacement for invalid characters.
- **PDF (`.pdf`)**: Page-by-page extraction using `pypdf.PdfReader`. Each page becomes a separate block with locator `page_N`. Empty pages are skipped; if no text is extracted, parsing fails.
- **HTML (`.html`, `.htm`)**: Parsed with `BeautifulSoup`, splitting on `<h1>` through `<h6>` headings. Script and style tags are removed. Content is grouped by section; if no headings exist, the entire document becomes a single block.

All parsers enforce a 50MB file size limit and reject empty files.

### Chunking

Chunking preserves document structure while maintaining semantic boundaries:

1. **Section detection**: Markdown-style headers (`##`) are detected and preserved as section boundaries.
2. **Paragraph splitting**: Within each section, text is split on paragraph boundaries (`\n`).
3. **Size limits**: Chunks are capped at 800 characters. If a paragraph exceeds this, it is split at the limit.
4. **Chunk identification**: Each chunk receives a unique `chunk_id` (e.g., `chunk_0`, `chunk_1`).

Metadata per chunk:
- `source`: Original filename (or custom `source_tag` if provided)
- `locator`: Page number (PDF), section name (HTML/Markdown), or `"document"` (plain text)
- `chunk_id`: Unique identifier within the document

### Indexing

Documents are indexed into ChromaDB using OpenAI's `text-embedding-3-small` model. The web application uses a separate collection (`web_documents`) isolated from CLI usage via the `CHROMA_COLLECTION_NAME` environment variable.

For hybrid retrieval, an in-memory BM25 index (`_indexed_docs`) is maintained alongside the vector store. This requires the `rank_bm25` package; if unavailable, the system degrades to vector-only retrieval without error.

On re-indexing the same file (identified by source name), existing entries are deleted from both ChromaDB and the BM25 index before new content is added, preventing stale data.

---

## Retrieval + Citation Strategy

### Retrieval Pipeline

The system uses hybrid retrieval combining semantic and lexical search:

1. **Vector retrieval**: ChromaDB query returns `top_k * 3` candidates (when hybrid or reranking is enabled) using cosine similarity on embeddings.
2. **BM25 retrieval**: Lexical search over the in-memory index using `rank_bm25.BM25Okapi`, returning `top_k * 2` candidates.
3. **Fusion**: Reciprocal Rank Fusion (RRF) with `k=60` combines both rankings into a single ordered list.
4. **Reranking**: Keyword overlap scoring counts shared words between query and document text, reordering candidates.
5. **Deduplication**: Duplicate document IDs are removed while preserving order.
6. **Final selection**: Top `k` chunks are returned (default: 5).

Optional `source_filter` parameter allows filtering by document source at the ChromaDB query level.

### Answer Generation

Retrieved chunks are concatenated and passed to GPT-4o-mini with a strict retrieval-based prompt:

- Context is treated as plain text, not executable instructions
- The model is instructed to use only the provided context
- If the answer is not in the context, the model must respond: "I couldn't find this information in the uploaded documents."
- Temperature is set to 0 for deterministic responses

This approach prevents hallucinations and provides basic prompt injection defense by explicitly instructing the model to ignore any instructions within the context.

### Citations

Each citation includes:
- `source`: Document filename
- `locator`: Page number (PDF), section name (HTML/Markdown), or chunk ID
- `snippet`: First 150 characters of the chunk text

Citations are generated for all retrieved chunks used in answer generation, enabling users to verify sources.

### Failure Handling

If retrieval returns no documents, the system returns: "I couldn't find anything relevant in the documents you've uploaded. Could you try rephrasing your question or upload different documents?"

If the LLM determines the answer is not in the context, it responds with the refusal message specified in the prompt.

---

## Memory Decision Logic

The memory system extracts high-signal, reusable facts from user input using pattern-based extraction with confidence scoring.

### Extraction Patterns

Three regex patterns identify memory-worthy signals:

1. **User role**: Pattern `\bi am a[n]?\s+(.+?)(\.|,| and|$)` extracts roles (e.g., "I am a Project Finance Analyst" → confidence 0.9)
2. **User preference**: Pattern `\bi prefer\s+(.+?)(\.|,| and|$)` extracts preferences (e.g., "I prefer weekly summaries" → confidence 0.85)
3. **Org insight**: Pattern `\bour team\s+(.+?)(\.|,|$)` extracts team-level information (e.g., "Our team interfaces with Asset Management" → confidence 0.8)

Each match produces a decision structure: `{should_write: bool, target: "USER"|"COMPANY", summary: str, confidence: float}`.

### Filtering Criteria

A memory entry is written only if all conditions are met:

1. `should_write == True`
2. `confidence >= 0.75` (threshold)
3. Summary does not match sensitive patterns (passwords, secrets, SSN, API keys, tokens, credentials, credit cards, PINs, PII)
4. Summary does not already exist in the target file (deduplication)

This selective approach avoids storing:
- Raw conversation transcripts
- Low-confidence signals
- Sensitive information
- Duplicate entries

---

## Memory File Writing

Memory is persisted to two markdown files in the project root:

- **`USER_MEMORY.md`**: User-specific facts (roles, preferences)
- **`COMPANY_MEMORY.md`**: Organization-wide learnings (team insights)

### Write Process

1. **File initialization**: If a file does not exist, it is created with header `# Memory Log\n\n`.
2. **Deduplication check**: The entire file content is read and checked for the exact summary string.
3. **Append**: If not a duplicate, the entry is appended as `- {summary}\n` in markdown list format.

Files are written synchronously on each memory write operation. No locking mechanism is implemented; concurrent writes may result in race conditions.

### Format

Entries follow a consistent format:
- `USER_MEMORY.md`: `- User role: [Role].` or `- User preference: [Preference].`
- `COMPANY_MEMORY.md`: `- Org insight: [Insight].`

---

## Tradeoffs and Limitations

### Design Tradeoffs

**Hybrid retrieval**: Combines semantic and lexical search to capture both meaning and exact keyword matches. RRF fusion balances both without manual weighting, but requires maintaining two indexes (vector store + in-memory BM25).

**Section-aware chunking**: Preserves document structure for better citations but may split semantically related content across chunks if sections are large.

**Pattern-based memory extraction**: Simple and deterministic but limited to explicit patterns. Misses implicit signals (e.g., "I work in finance" does not match "I am a..." pattern).

**In-memory storage**: ChromaDB uses in-memory client for simplicity, but data is lost on server restart. BM25 index is also in-memory, requiring full re-indexing on restart.

### Known Limitations

1. **BM25 dependency**: Hybrid retrieval silently degrades to vector-only if `rank_bm25` is not installed. No error is raised.
2. **Re-indexing behavior**: When re-indexing the same file, ChromaDB entries are deleted correctly, but the BM25 index clearing logic may leave duplicates if deletion fails silently.
3. **Memory extraction scope**: Only three patterns are supported. More nuanced extraction (e.g., "I work in finance" → role) requires LLM-based extraction.
4. **No memory retrieval**: Memory is written but not used to inform answers. Past conversations are not referenced.
5. **File concurrency**: Memory file writes are not locked; concurrent requests may cause race conditions.
6. **Chunking granularity**: Fixed 800-character limit may split semantically coherent content or include irrelevant context.
7. **Reranking simplicity**: Keyword overlap is a basic heuristic; cross-encoder models would provide more accurate reranking.

### Production Considerations

For production deployment, consider:
- Persistent ChromaDB storage (e.g., ChromaDB server or PostgreSQL with pgvector)
- LLM-based memory extraction for broader coverage
- Memory retrieval integration to inform answers
- File locking for concurrent memory writes
- Semantic chunking using embeddings to split on topic boundaries
- Cross-encoder models for reranking
- Error handling and retry logic for API failures
- Rate limiting and request queuing
