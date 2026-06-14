# RAG BGE-M3 Tokenizer - Local AI Stack (v2.0.0)

A high-performance, local RAG (Retrieval-Augmented Generation) solution optimized for Fedora 44 (Sway) and Linux environments. This system utilizes exact token counting for chunking and parallel embedding processing to ensure maximum performance when indexing large legal archives or personal document collections.

## ✨ What's New in v2.0.0

- **Multi-format Support:** Ingest PDF, Markdown (.md), RST, and plain text.
- **Parallel Indexing:** Uses `asyncio.Semaphore` to process multiple files concurrently without blocking the server.
- **Force Re-indexing:** Support for the `force=true` flag to overwrite and update existing documents.
- **Environment Configurable:** Control batch sizes, chunk sizes, and timeouts via environment variables.
- **Robust ID Handling:** Normalized `doc_id` generation (e.g., `document.pdf`) to prevent collisions across different formats.

## 🏗 Architecture

- **LLM Engine:** `llama-server` (Port 11434) - Handles reasoning and chat.
- **Embedding Engine:** `llama-server` @ BGE-M3 (Port 11435) - Generates 1024-dim vectors.
- **Reranker Engine:** `llama-server` @ BGE-Reranker-v2-m3 (Port 11436) - Scores search results for accuracy.
- **Database:** SQLite with the `sqlite-vec` extension for local vector storage and metadata.

## ⚙️ Environment Variables (Optional)

| Variable               | Description                         | Default |
| :--------------------- | :---------------------------------- | :------ |
| `RAG_CHUNK_SIZE`       | Maximum tokens per segment          | 512     |
| `RAG_CHUNK_OVERLAP`    | Overlapping tokens between segments | 64      |
| `RAG_MAX_CONCURRENT`   | Max files indexed in parallel       | 3       |
| `RAG_EMBED_BATCH_SIZE` | Chunks per embedding API call       | 8       |

## 🛠 Tools

| Tool                | Description                                                                                                                                               |
| :------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `create_collection` | Create a new RAG collection in the database.                                                                                                              |
| `ingest_file`       | Index a file (Text, PDF, or Markdown) directly from disk. Optimized for large files to bypass LLM context limits. Supports re-indexing with `force=true`. |
| `ingest_directory`  | Batch index an entire directory. Supports `.txt`, `.pdf`, `.md`, and more. Features parallel processing, progress bars, and ETA reporting.                |
| `add_documents`     | Index raw text strings directly (e.g., short notes, chat snippets, or clippings).                                                                         |
| `query`             | Perform a semantic search within a collection using Vector ANN followed by a Reranking stage.                                                             |
| `list_collections`  | List all collections in the database with document counts.                                                                                                |
| `delete_documents`  | Remove specific documents by ID or clear an entire collection (by passing an empty list `[]`).                                                            |
| `delete_collection` | Permanently delete a collection, including all documents, chunks, and metadata.                                                                           |

## 🚀 Quickstart

### 1. Setup your Alias (~/.zshrc or ~/.bashrc)

```bash
alias ai-agent-rag='export OPENAI_API_KEY="sk-unused" && \
  export MCP_STREAMING_FIRST_CHUNK_TIMEOUT=3600 && \
  export MCP_STREAMING_CHUNK_TIMEOUT=3600 && \
  export MCP_STREAMING_GLOBAL_TIMEOUT=3600 && \
  uvx mcp-cli chat \
    --provider openai_compatible \
    --api-base "http://localhost:11434/v1" \
    --api-key "sk-unused" \
    --model "local" \
    --config-file ~/.config/mcp-cli/server_config.json'
```
