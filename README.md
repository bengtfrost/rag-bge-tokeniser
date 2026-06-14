# RAG BGE-M3 Tokeniser - Local AI Stack (v2.0.0)

En högoptimerad, lokal RAG-lösning för Fedora 44 (Sway). Systemet använder exakt token-räkning för chunking och parallell embedding-bearbetning för maximal prestanda vid indexering av stora juridiska arkiv.

## ✨ Nyheter i v2.0.0

- **Multi-format Support:** Ingest av PDF, Markdown (.md), RST och ren text.
- **Parallell Indexering:** Använder `asyncio.Semaphore` för att bearbeta flera filer samtidigt utan att blockera.
- **Force Re-indexing:** Stöd för `--force` flagga för att uppdatera existerande dokument.
- **Konfigurerbar via Env:** Styr batch-storlekar, chunk-storlek och timeouts via miljövariabler.
- **Robust ID-hantering:** Normaliserade `doc_id` (t.ex. `lag.pdf`) för att undvika kollisioner.

## 🏗 Arkitektur

- **LLM Engine:** `llama-server` (Port 11434)
- **Embedding Engine:** `llama-server` @ BGE-M3 (Port 11435)
- **Reranker Engine:** `llama-server` @ BGE-Reranker-v2-m3 (Port 11436)
- **Database:** SQLite med `sqlite-vec` för vektorsökning.

## ⚙️ Miljövariabler (Optional)

| Variabel               | Beskrivning                              | Standard |
| :--------------------- | :--------------------------------------- | :------- |
| `RAG_CHUNK_SIZE`       | Max tokens per segment                   | 512      |
| `RAG_CHUNK_OVERLAP`    | Överlappande tokens                      | 64       |
| `RAG_MAX_CONCURRENT`   | Max antal filer som indexeras parallellt | 3        |
| `RAG_EMBED_BATCH_SIZE` | Chunks per embedding-anrop               | 8        |

## 🛠 Verktyg (Tools)

- `ingest_file`: Indexera enskild fil (PDF/MD/TXT).
- `ingest_directory`: Massindexera kataloger med progress-bar och ETA.
- `query`: Hybrid-sökning med Vektor + Reranking.
- `list_collections`: Statistik över dokument per samling.
- `delete_documents`: Selektiv radering eller rensning av samling.

## 🚀 Snabbstart

```bash
# Starta agenten (~/.zshrc alias)
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

# Exempel: Indexera en juridisk PDF
# Inuti chatten:
# "Ingest /home/bfrost/ai-docs/juridik/doktrin/Offentlig_ratt_allmant.pdf till juridik_doktrin"
```
