# RAG BGE-M3 Tokeniser - Local AI Stack

En högpresterande, lokal RAG-lösning (Retrieval-Augmented Generation) byggd för Fedora 44 (Sway) som utnyttjar MCP (Model Context Protocol) för att ge lokala LLM:er tillgång till juridisk doktrin och personlig data med exakt token-kontroll.

## 🏗 Arkitektur & Kedja

Systemet är uppbyggt som en distribuerad mikrotjänst-arkitektur lokalt:

1.  **Inference (Port 11434):** `llama-server` kör modeller (Gemma, Qwen, DeepSeek) för resonemang.
2.  **Embedding (Port 11435):** `llama-server` med `bge-m3.gguf` genererar vektorer (1024 dim).
3.  **Reranking (Port 11436):** `llama-server` med `bge-reranker-v2-m3` för poängsättning av sökresultat.
4.  **MCP Hub (`mcp-cli`):** Fungerar som brygga mellan LLM och verktyg.
5.  **RAG Server (`rag_server.py`):** 
    *   **Logic:** Custom Python MCP-server.
    *   **Chunking:** Exakt token-count via `transformers` (BGE-M3 tokeniser).
    *   **Storage:** SQLite med `sqlite-vec` extension för vektorsökning.
6.  **SQL Server (`mcp-server-sqlite`):** Ger direkt SQL-åtkomst till `vectors.db` för metadata-analys.

## 🛠 Installation & Setup

### Miljö
```bash
mkdir -p ~/.config/mcp-cli/rag-bge-tokeniser
cd ~/.config/mcp-cli/rag-bge-tokeniser
uv init --no-readme
uv venv
uv add sqlite-vec httpx mcp transformers sentencepiece
```

### Komponenter
- **Databas:** `~/.local/share/rag-bge-tokeniser/vectors.db`
- **Tokenizer Cache:** Lokalt lagrad BGE-M3 modell för offline-användning.
- **Server:** `rag_server.py` (hanterar ingest, query, rerank och management).

## 🚀 Verktyg (Tools)

| Server | Verktyg | Funktion |
| :--- | :--- | :--- |
| **rag** | `ingest_directory` | Massindexering med progressbar och ETA. |
| **rag** | `query` | Hybrid sökning (Vektor + Rerank). |
| **sqlite** | `read_query` | Direkt SQL-analys av vektordatabasen. |
| **filesystem** | `read_text_file` | Låter agenten läsa källfiler. |

## 📁 Filstruktur
- `rag_server.py`: Huvudservern för RAG-logik.
- `server_config.json`: Konfiguration för MCP-klienter.
- `tokeniser_cache/`: Lokala filer för BGE-tokenisern.
- `vectors.db`: SQLite-databas med vektortabeller.

