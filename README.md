Th# RAG BGE-M3 Tokenizer - Sovereign AI Stack (v2.0.0)

A high-performance, local RAG (Retrieval-Augmented Generation) solution optimized for Fedora 44 (Sway) and Linux power users. This system utilizes **exact token counting** for chunking and **parallel embedding processing**, orchestrated by the Rust-powered **Goose** agent via a **LiteLLM** gateway.

> **Note:** This project has migrated from `mcp-cli` to **Goose** to provide a stable, agentic environment without hardcoded session timeouts or unstable STDIO handling.

## ✨ Key Features (v2.0.0)

- **Goose Orchestration:** Uses a stable, Rust-based agentic CLI that handles long-running "missions" (like large-scale indexing) without session timeouts.
- **LiteLLM Gateway:** A central proxy that unifies Local LLMs (llama-server) and Cloud APIs (Mistral/Gemini) into a single OpenAI-compatible stream on Port 4000.
- **Multi-format Support:** Ingest PDF, Markdown (.md), RST, and plain text.
- **Exact Tokenization:** Powered by the `BAAI/bge-m3` tokenizer to ensure 1:1 parity between chunking logic and embedding model constraints.
- **Parallel Indexing:** Uses `asyncio.Semaphore` to process multiple files concurrently without blocking the server.
- **Sovereign Metadata:** Direct SQL access to the vector database for structural analysis and statistics.

## 🏗 Architecture

```mermaid
graph TD
    A[goose Agent] -->|Port 4000| B(LiteLLM Gateway)
    B -->|Port 11434| C(Local LLM: Gemma/DeepSeek)
    B -->|Cloud API| D(Mistral/Gemini)
    A --- |MCP: STDIO| E{Sovereign Tools}
    E --> G1[rag_server: v2.0 RAG]
    E --> G2[sqlite-vec: Direct SQL]
    E --> G3[filesystem/fetch]
```

## 🛠 MCP Tools

| Tool                 | Description                                                       |
| :------------------- | :---------------------------------------------------------------- |
| `create_collection`  | Create a new RAG collection.                                      |
| `ingest_file`        | Index a file (Text, PDF, or Markdown) directly from disk.         |
| `ingest_directory`   | Batch index an entire directory with progress bars and ETA.       |
| `query`              | Hybrid search using Vector ANN followed by a BGE-Reranking stage. |
| `sqlite__read_query` | Perform raw SQL queries on the metadata and vector stats.         |
| `delete_documents`   | Selective deletion or full collection purge.                      |

## 🚀 Installation & Setup

### 1. Install Goose (The Orchestrator)

Install the binary directly for maximum performance:

```bash
curl -fsSL https://github.com/aaif-goose/goose/releases/download/stable/download_cli.sh | CONFIGURE=false bash
```

### 2. Configure the Stack

Update your `~/.config/goose/config.yaml` to include your Sovereign tools. This configuration uses the `openai` provider type to redirect all traffic to your LiteLLM gateway.

```yaml
GOOSE_TELEMETRY_ENABLED: false
active_provider: openai

providers:
  openai:
    type: openai
    base_url: http://localhost:4000/v1
    api_key: sk-unused

extensions:
  rag:
    enabled: true
    name: rag
    type: stdio
    cmd: /home/USER/.config/rag-bge-tokeniser/.venv/bin/python
    args:
      - /home/USER/.config/rag-bge-tokeniser/rag_server.py
  sqlite:
    enabled: true
    name: sqlite
    type: stdio
    cmd: uvx
    args:
      - mcp-server-sqlite
      - --db-path
      - /home/USER/.local/share/rag-bge-tokeniser/vectors.db
  filesystem:
    enabled: true
    name: filesystem
    type: stdio
    cmd: npx
    args:
      - "-y"
      - "@modelcontextprotocol/server-filesystem"
      - /home/USER/ai-docs
      - /home/USER/.code
```

### 3. Setup Aliases (~/.zshrc)

To ensure stability and bypass Linux keyring issues, we use environment variables to route traffic through the LiteLLM gateway:

```bash
# Gateway ensured start
_ensure_litellm() {
    if ! ss -tulpn | grep -q ":4000 "; then
        v_litellm && litellm -c ~/.config/litellm/config.yaml > /dev/null 2>&1 &
        while ! ss -tulpn | grep -q ":4000 "; do sleep 1; done
    fi
}

alias goose-local='_ensure_litellm && export OPENAI_API_KEY="sk-unused" && export OPENAI_BASE_URL="http://localhost:4000/v1" && GOOSE_MODEL=local goose session'
alias goose-smart='_ensure_litellm && export OPENAI_API_KEY="sk-unused" && export OPENAI_BASE_URL="http://localhost:4000/v1" && GOOSE_MODEL=mistral-large-latest goose session'
```

## ⚙️ Environment Variables (Optional)

You can tune the RAG performance via environment variables in the shell where the server is launched:

| Variable             | Description                   | Default |
| :------------------- | :---------------------------- | :------ |
| `RAG_CHUNK_SIZE`     | Maximum tokens per segment    | 512     |
| `RAG_MAX_CONCURRENT` | Max files indexed in parallel | 3       |

## 📂 Project Structure

- `rag_server.py`: Core MCP logic & Exact Tokenizer.
- `vectors.db`: SQLite-vec database (Location: `~/.local/share/rag-bge-tokeniser/`).
- `config.yaml`: The primary Goose configuration file.

---

**Author:** [Bengt Frost](https://github.com/bengtfrost)\
**License:** MIT
