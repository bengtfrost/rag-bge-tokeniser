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

- `ingest_file`: Index a single file (PDF/MD/TXT) from disk.
- `ingest_directory`: Batch index directories with a progress bar and ETA.
- `query`: Hybrid search using Vector ANN + Reranking.
- `list_collections`: Statistics and document counts per collection.
- `delete_documents`: Selective deletion of documents or clearing a collection.
- `delete_collection`: Complete removal of a collection and its vectors.

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
