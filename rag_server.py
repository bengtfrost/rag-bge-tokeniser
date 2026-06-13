#!/usr/bin/env python3
import json
import os
import re
import sqlite3
import asyncio
import httpx
import sqlite_vec
import sys
import time

# --- MILJÖINSTÄLLNINGAR ---
# Tvinga Hugging Face och Transformers att vara tysta för att inte korrumpera stdout (JSON-RPC)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import ServerCapabilities
import mcp.server.stdio
import mcp.types as types
from transformers import AutoTokenizer

# ========== KONFIGURATION ==================================================
DB_PATH = "/home/bfrost/.local/share/rag-bge-tokeniser/vectors.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

TOKENISER_CACHE = "/home/bfrost/.config/mcp-cli/rag-bge-tokeniser/tokeniser_cache"
EMBED_URL       = "http://localhost:11435/v1/embeddings"
RERANK_URL      = "http://localhost:11436/rerank"
TIMEOUT         = 7200   # 2 timmar för tunga juridiska dokument

CHUNK_SIZE        = 512  # tokens per chunk
CHUNK_OVERLAP     = 64   # tokens överlapp (token-exakt, se chunk_text_exact)
EMBED_BATCH_SIZE  = 8
RERANK_CANDIDATES = 20
RERANK_MIN_SCORE  = 0.1
# ===========================================================================


# ========== HJÄLPFUNKTIONER ================================================

def debug_log(msg: str):
    """Loggar till stderr så att MCP-protokollet på stdout inte störs."""
    print(f"[*] RAG-LOG: {msg}", file=sys.stderr, flush=True)


def progress_bar(current: int, total: int, width: int = 30) -> str:
    """Returnerar en ASCII-progressbar, t.ex. [████████░░░░░░░] 53%"""
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {int(pct * 100)}%"


def fmt_seconds(seconds: float) -> str:
    """Formaterar sekunder till läsbar sträng, t.ex. '2m 14s'."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60}s"


# ========== TOKENIZER (lazy load) ==========================================

_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        debug_log("Laddar BGE-M3 tokenizer från cache...")
        _tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/bge-m3", cache_dir=TOKENISER_CACHE
        )
        debug_log("Tokenizer laddad.")
    return _tokenizer


# ========== TEXTBEARBETNING ================================================

def split_into_sentences(text: str) -> list[str]:
    """Delar upp text i meningar med hänsyn till svenska tecken och stycken."""
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÅÄÖ])", text)
    return [s.strip() for s in sentences if len(s.strip()) > 2]


def chunk_text_exact(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Skapar chunks baserat på exakt token-count via BGE-M3-tokenizern.
    Överlapp är token-exakt: efter varje chunk tas meningar från slutet
    tills overlap_tokens uppnås, istället för en hårdkodad mening-räknare.
    """
    tk = get_tokenizer()
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    # Tokenisera alla meningar en gång – undviker upprepad tokenisering
    sentence_tokens: list[int] = [
        len(tk.encode(s, add_special_tokens=False)) for s in sentences
    ]

    chunks: list[str] = []
    current_sents: list[str] = []
    current_tokens: int = 0

    for sent, s_tok in zip(sentences, sentence_tokens):

        # Hantera extremt långa meningar – trunkera på teckennivå
        if s_tok > max_tokens:
            if current_sents:
                chunks.append(" ".join(current_sents))
                current_sents, current_tokens = [], 0
            chunks.append(sent[: max_tokens * 4])
            continue

        if current_tokens + s_tok > max_tokens and current_sents:
            chunks.append(" ".join(current_sents))

            # Token-exakt överlapp: bygg bakifrån tills vi når overlap_tokens
            overlap_sents: list[str] = []
            overlap_tok = 0
            for s, t in zip(reversed(current_sents), reversed(
                [len(tk.encode(s, add_special_tokens=False)) for s in current_sents]
            )):
                if overlap_tok + t > overlap_tokens:
                    break
                overlap_sents.insert(0, s)
                overlap_tok += t

            current_sents  = overlap_sents
            current_tokens = overlap_tok

        current_sents.append(sent)
        current_tokens += s_tok

    if current_sents:
        chunks.append(" ".join(current_sents))

    return chunks


# ========== DATABAS =========================================================

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.enable_load_extension(True)
sqlite_vec.load(conn)

with conn:
    conn.execute("CREATE TABLE IF NOT EXISTS collections (name TEXT PRIMARY KEY)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS docs (
            id          TEXT PRIMARY KEY,
            collection  TEXT,
            text        TEXT,
            parent_id   TEXT,
            chunk_index INTEGER
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS vectors USING vec0(
            id         TEXT PRIMARY KEY,
            collection TEXT,
            embedding  FLOAT[1024]
        )
    """)
    # Index för snabbare text-lookup vid query (undviker full table scan)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_docs_collection ON docs(collection)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_docs_parent ON docs(parent_id, collection)"
    )


# ========== NÄTVERKSANROP ==================================================

async def get_embeddings(
    texts: list[str],
    label: str = "",
    progress_offset: int = 0,
    progress_total: int = 0,
) -> list[list[float]]:
    """
    Skickar batchar till embedding-server (port 11435).
    Loggar progress per batch med progressbar och ETA till stderr.

    progress_offset  – antal redan bearbetade chunks (för katalogjobb)
    progress_total   – totalt antal chunks i hela jobbet (för katalogjobb)
    """
    all_embeddings: list[list[float]] = []
    total_batches = -(-len(texts) // EMBED_BATCH_SIZE)  # ceil division
    job_total     = progress_total if progress_total > 0 else len(texts)
    prefix        = f"[{label}] " if label else ""
    t_start       = time.monotonic()

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for batch_num, i in enumerate(range(0, len(texts), EMBED_BATCH_SIZE), start=1):
            batch_texts = texts[i : i + EMBED_BATCH_SIZE]

            resp = await client.post(
                EMBED_URL, json={"input": batch_texts, "model": "bge-m3"}
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            all_embeddings.extend(
                d["embedding"] for d in sorted(data, key=lambda x: x["index"])
            )

            # ETA-beräkning: visa "beräknar..." på första batchen då elapsed≈0
            elapsed       = time.monotonic() - t_start
            done_chunks   = progress_offset + i + len(batch_texts)
            bar           = progress_bar(done_chunks, job_total)

            if batch_num == 1:
                eta_str = "ETA: beräknar..."
            else:
                avg_per_chunk   = elapsed / max(i + len(batch_texts), 1)
                remaining       = job_total - done_chunks
                eta_str         = f"ETA: {fmt_seconds(remaining * avg_per_chunk)}"

            debug_log(
                f"{prefix}Embeddings batch {batch_num}/{total_batches} "
                f"{bar}  {eta_str}"
            )

    return all_embeddings


async def rerank(
    query: str, documents: list[str], top_n: int
) -> list[tuple[int, float]]:
    """Skickar kandidater till reranker-server (port 11436)."""
    if not documents:
        return []
    debug_log(f"Rerankar {len(documents)} kandidater...")
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(RERANK_URL, json={"query": query, "documents": documents})
        resp.raise_for_status()
        results = resp.json()["results"]

    indexed  = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
    filtered = [r for r in indexed if r["relevance_score"] > RERANK_MIN_SCORE]
    debug_log(f"Reranking klar: {len(filtered)} träffar över tröskel {RERANK_MIN_SCORE}.")
    return [(r["index"], r["relevance_score"]) for r in filtered[:top_n]]


# ========== INDEXERINGSHJÄLP ===============================================

def _purge_document(coll: str, parent_id: str):
    """
    Tar bort alla befintliga chunks för ett dokument innan re-ingest.
    Förhindrar att inaktuella chunks ligger kvar efter en uppdaterad fil.
    """
    conn.execute("DELETE FROM vectors WHERE id LIKE ?", (f"{parent_id}_ch%",))
    conn.execute(
        "DELETE FROM docs WHERE parent_id = ? AND collection = ?",
        (parent_id, coll),
    )


def _doc_exists(coll: str, parent_id: str) -> bool:
    """
    Kontrollerar om ett dokument redan är indexerat i en samling.

    Args:
        coll:      Samlingsnamn.
        parent_id: Dokumentets parent_id (motsvarar doc_id vid ingest).

    Returns:
        True om minst ett chunk med detta parent_id finns i samlingen.
    """
    row = conn.execute(
        "SELECT 1 FROM docs WHERE parent_id = ? AND collection = ? LIMIT 1",
        (parent_id, coll),
    ).fetchone()
    return row is not None


async def _index_chunks(
    coll: str,
    doc_id: str,
    chunks: list[str],          # redan förberedda chunks (ingen re-chunkning)
    progress_offset: int = 0,
    progress_total: int = 0,
) -> int:
    """
    Tar emot färdiga chunks, beräknar embeddings och sparar till DB.
    Rensar alltid befintliga chunks för doc_id innan inläggning (re-ingest-säker).
    Returnerar antalet indexerade segment.
    """
    if not chunks:
        debug_log(f"'{doc_id}': inga segment att indexera.")
        return 0

    t0 = time.monotonic()
    debug_log(f"'{doc_id}': {len(chunks)} segment, startar embeddings...")

    embeddings = await get_embeddings(
        chunks,
        label=doc_id,
        progress_offset=progress_offset,
        progress_total=progress_total if progress_total > 0 else len(chunks),
    )

    debug_log(f"'{doc_id}': sparar till databas...")

    # Atomisk: rensa gamla + skriv nya i samma transaktion
    with conn:
        _purge_document(coll, doc_id)
        for idx, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_ch{idx}"
            conn.execute(
                "INSERT INTO vectors (id, collection, embedding) VALUES (?, ?, ?)",
                (chunk_id, coll, json.dumps(emb)),
            )
            conn.execute(
                "INSERT INTO docs (id, collection, text, parent_id, chunk_index) "
                "VALUES (?, ?, ?, ?, ?)",
                (chunk_id, coll, chunk_text, doc_id, idx),
            )

    elapsed = time.monotonic() - t0
    debug_log(f"'{doc_id}': klar – {len(chunks)} segment på {fmt_seconds(elapsed)}.")
    return len(chunks)


# ========== MCP SERVER =====================================================

app = Server("rag-bge-tokeniser")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="create_collection",
            description="Skapa en ny RAG-samling",
            inputSchema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        ),
        types.Tool(
            name="ingest_file",
            description=(
                "Läs och indexera en textfil direkt från disk utan att skicka innehållet "
                "genom LLM-kontextfönstret. Föredra detta framför add_documents för stora filer. "
                "Stödjer re-ingest: gamla chunks rensas automatiskt."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "collection":   {"type": "string", "description": "Samlingsnamn"},
                    "file_path":    {"type": "string", "description": "Absolut sökväg till filen"},
                    "document_id":  {"type": "string", "description": "Valfritt ID (standard: filnamnet)"},
                    "encoding":     {"type": "string", "description": "Teckenkodning (standard: utf-8)"},
                },
                "required": ["collection", "file_path"],
            },
        ),
        types.Tool(
            name="ingest_directory",
            description=(
                "Indexera alla textfiler i en katalog direkt från disk. "
                "Chunkar varje fil ett enda gång, kör sedan embeddings. "
                "Rapporterar progress per fil och totalt med ETA. "
                "Stödjer re-ingest: gamla chunks rensas automatiskt."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "collection":      {"type": "string", "description": "Samlingsnamn"},
                    "directory_path":  {"type": "string", "description": "Absolut sökväg till katalogen"},
                    "file_extension":  {"type": "string", "description": "Filändelse (standard: .txt), skiftlägesokänslig"},
                    "encoding":        {"type": "string", "description": "Teckenkodning (standard: utf-8)"},
                },
                "required": ["collection", "directory_path"],
            },
        ),
        types.Tool(
            name="add_documents",
            description=(
                "Indexera textdokument som skickas direkt som strängar. "
                "OBS: Använd ingest_file för stora filer för att undvika kontextfönsterproblem."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {"type": "string"},
                    "ids":        {"type": "array", "items": {"type": "string"}},
                    "documents":  {"type": "array", "items": {"type": "string"}},
                },
                "required": ["collection", "ids", "documents"],
            },
        ),
        types.Tool(
            name="query",
            description="Sök i samlingen med semantisk sökning och reranking",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {"type": "string"},
                    "query":      {"type": "string"},
                    "top_k":      {"type": "integer", "description": "Antal resultat (standard: 5)"},
                },
                "required": ["collection", "query"],
            },
        ),
        types.Tool(
            name="list_collections",
            description="Lista alla samlingar i databasen",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="delete_documents",
            description=(
                "Ta bort ett eller flera dokument från en samling. "
                "Om ids är en tom lista ([]) tas ALLA dokument i samlingen bort "
                "(samlingen behålls men töms helt)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {"type": "string"},
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lista med dokument-ID:n. Tom lista = rensa hela samlingen.",
                    },
                },
                "required": ["collection", "ids"],
            },
        ),
        types.Tool(
            name="delete_collection",
            description=(
                "Ta bort en hel samling inklusive alla dokument, chunks och metadata. "
                "Returnerar fel om samlingen inte existerar."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Namnet på samlingen som ska tas bort"},
                },
                "required": ["name"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    # ------------------------------------------------------------------ #
    if name == "create_collection":
        coll_name = arguments["name"]
        conn.execute("INSERT OR IGNORE INTO collections (name) VALUES (?)", (coll_name,))
        conn.commit()
        return [types.TextContent(type="text", text=f"Samlingen '{coll_name}' är nu redo.")]

    # ------------------------------------------------------------------ #
    elif name == "ingest_file":
        coll      = arguments["collection"]
        file_path = arguments["file_path"]
        doc_id    = arguments.get("document_id") or os.path.basename(file_path)
        encoding  = arguments.get("encoding", "utf-8")

        if not os.path.isfile(file_path):
            return [types.TextContent(type="text", text=f"Fel: Filen hittades inte: {file_path}")]

        try:
            with open(file_path, "r", encoding=encoding) as f:
                text = f.read()
        except Exception as e:
            return [types.TextContent(type="text", text=f"Fel vid läsning av fil: {e}")]

        file_size_kb = os.path.getsize(file_path) // 1024
        debug_log(f"Startar ingest av '{os.path.basename(file_path)}' ({file_size_kb} KB)...")

        conn.execute("INSERT OR IGNORE INTO collections (name) VALUES (?)", (coll,))
        conn.commit()

        # Dubbelindexeringsskydd: varna om dokumentet redan finns
        if _doc_exists(coll, doc_id):
            return [types.TextContent(
                type="text",
                text=(
                    f"Varning: Dokumentet '{doc_id}' är redan indexerat i samlingen '{coll}'. "
                    f"Inga ändringar gjordes. Använd delete_documents för att ta bort det först "
                    f"om du vill re-indexera."
                ),
            )]

        t0     = time.monotonic()
        chunks = chunk_text_exact(text, CHUNK_SIZE, CHUNK_OVERLAP)
        total  = await _index_chunks(coll, doc_id, chunks)
        elapsed = time.monotonic() - t0

        if total == 0:
            return [types.TextContent(type="text", text="Ingen text att indexera.")]

        return [types.TextContent(
            type="text",
            text=(
                f"✓ Klar! Indexerade {total} segment från '{os.path.basename(file_path)}' "
                f"i samlingen '{coll}' på {fmt_seconds(elapsed)}."
            ),
        )]

    # ------------------------------------------------------------------ #
    elif name == "ingest_directory":
        coll     = arguments["collection"]
        dir_path = arguments["directory_path"]
        ext      = arguments.get("file_extension", ".txt").lower()
        encoding = arguments.get("encoding", "utf-8")

        if not os.path.isdir(dir_path):
            return [types.TextContent(type="text", text=f"Fel: Katalogen hittades inte: {dir_path}")]

        # Skiftlägesokänslig filtrering
        files = sorted(
            f for f in os.listdir(dir_path) if f.lower().endswith(ext)
        )
        if not files:
            return [types.TextContent(type="text", text=f"Inga {ext}-filer hittades i {dir_path}.")]

        debug_log(f"Katalogindexering startar: {len(files)} filer i '{dir_path}'...")

        conn.execute("INSERT OR IGNORE INTO collections (name) VALUES (?)", (coll,))
        conn.commit()

        # ── Första pass: läs och chunka alla filer (ingen embedding ännu) ──
        # Chunks återanvänds direkt i andra passet – ingen dubbel-tokenisering.
        debug_log("Läser och chunkar alla filer...")
        FileEntry = tuple  # (filepath, doc_id, chunks, error)
        prepared: list[FileEntry] = []

        for filename in files:
            fp     = os.path.join(dir_path, filename)
            doc_id = os.path.splitext(filename)[0]
            try:
                with open(fp, "r", encoding=encoding) as f:
                    text = f.read()
                chunks = chunk_text_exact(text, CHUNK_SIZE, CHUNK_OVERLAP)
                prepared.append((fp, doc_id, chunks, None))
                debug_log(f"  Chunkat: '{filename}' → {len(chunks)} segment")
            except Exception as e:
                prepared.append((fp, doc_id, [], str(e)))
                debug_log(f"  Fel vid läsning av '{filename}': {e}")

        grand_total = sum(len(p[2]) for p in prepared)
        debug_log(f"Totalt: {len(files)} filer, {grand_total} segment. Startar embeddings...")

        # ── Andra pass: embeddings + DB-skrivning ──
        results: list[str] = []
        total_chunks = 0
        t_all = time.monotonic()

        for file_num, (fp, doc_id, chunks, err) in enumerate(prepared, start=1):
            filename = os.path.basename(fp)

            if err:
                results.append(f"  ✗ {filename}: {err}")
                continue

            if not chunks:
                results.append(f"  ✗ {filename}: tom fil")
                continue

            debug_log(
                f"Fil {file_num}/{len(files)}: '{filename}' "
                f"{progress_bar(file_num - 1, len(files))}  ({len(chunks)} segment)"
            )

            # Dubbelindexeringsskydd per fil
            if _doc_exists(coll, doc_id):
                results.append(
                    f"  ⚠ {filename}: redan indexerad – hoppades över "
                    f"(ta bort med delete_documents för att re-indexera)"
                )
                debug_log(f"  '{filename}' redan indexerad, hoppar över.")
                continue

            try:
                n = await _index_chunks(
                    coll, doc_id, chunks,
                    progress_offset=total_chunks,
                    progress_total=grand_total,
                )
                total_chunks += n
                results.append(f"  ✓ {filename}: {n} segment")
            except Exception as e:
                results.append(f"  ✗ {filename}: fel – {e}")
                debug_log(f"  FEL vid indexering av '{filename}': {e}")

        elapsed_all = time.monotonic() - t_all
        summary = (
            f"Katalogindexering klar. "
            f"{len(files)} filer · {total_chunks} segment · "
            f"{fmt_seconds(elapsed_all)} · samling: '{coll}'\n"
        )
        debug_log(summary.strip())
        return [types.TextContent(type="text", text=summary + "\n".join(results))]

    # ------------------------------------------------------------------ #
    elif name == "add_documents":
        coll = arguments["collection"]
        ids  = arguments["ids"]
        docs = arguments["documents"]

        conn.execute("INSERT OR IGNORE INTO collections (name) VALUES (?)", (coll,))
        conn.commit()

        total_chunks = 0
        skipped: list[str] = []
        t0 = time.monotonic()

        for i, (doc_id, doc_text) in enumerate(zip(ids, docs), start=1):
            debug_log(f"Dokument {i}/{len(ids)}: '{doc_id}'")

            # Dubbelindexeringsskydd
            if _doc_exists(coll, doc_id):
                debug_log(f"  '{doc_id}' redan indexerat, hoppar över.")
                skipped.append(doc_id)
                continue

            chunks = chunk_text_exact(doc_text, CHUNK_SIZE, CHUNK_OVERLAP)
            n = await _index_chunks(coll, doc_id, chunks)
            total_chunks += n

        elapsed = time.monotonic() - t0
        if total_chunks == 0 and not skipped:
            return [types.TextContent(type="text", text="Ingen text att indexera.")]

        msg = f"✓ Indexerade {total_chunks} segment i '{coll}' på {fmt_seconds(elapsed)}."
        if skipped:
            skipped_list = ", ".join(f"'{s}'" for s in skipped)
            msg += (
                f"\nVarning: Följande dokument var redan indexerade och hoppades över: "
                f"{skipped_list}. Använd delete_documents för att ta bort dem först."
            )
        return [types.TextContent(type="text", text=msg)]

    # ------------------------------------------------------------------ #
    elif name == "query":
        coll  = arguments["collection"]
        query = arguments["query"]
        top_k = arguments.get("top_k", 5)

        # Trunkera endast om nödvändigt för loggraden
        query_preview = query[:80] + ("..." if len(query) > 80 else "")
        debug_log(f"Query: '{query_preview}' i samling '{coll}'")
        t0 = time.monotonic()

        query_emb = (await get_embeddings([query], label="query"))[0]

        debug_log(f"Hämtar {RERANK_CANDIDATES} ANN-kandidater...")
        cursor = conn.execute(
            """
            SELECT id FROM vectors
            WHERE collection = ? AND embedding MATCH ? AND k = ?
            """,
            (coll, json.dumps(query_emb), RERANK_CANDIDATES),
        )

        chunk_ids = [row[0] for row in cursor.fetchall()]
        if not chunk_ids:
            return [types.TextContent(type="text", text="Hittade inget relevant.")]

        debug_log(f"{len(chunk_ids)} kandidater hämtade, hämtar text...")
        rows = conn.execute(
            f"SELECT id, text, parent_id FROM docs "
            f"WHERE id IN ({','.join(['?'] * len(chunk_ids))})",
            chunk_ids,
        ).fetchall()
        doc_map   = {row[0]: (row[1], row[2]) for row in rows}
        valid_ids = [cid for cid in chunk_ids if cid in doc_map]

        reranked = await rerank(query, [doc_map[cid][0] for cid in valid_ids], top_k)

        if not reranked:
            return [types.TextContent(type="text", text="Inga tillräckligt relevanta träffar.")]

        elapsed = time.monotonic() - t0
        debug_log(f"Query klar på {fmt_seconds(elapsed)}, returnerar {len(reranked)} träffar.")

        res = [
            f"[{i}] (Källa: {doc_map[valid_ids[idx]][1]}) Score: {score:.4f}\n"
            f"{doc_map[valid_ids[idx]][0]}"
            for i, (idx, score) in enumerate(reranked, start=1)
        ]
        return [types.TextContent(type="text", text="\n\n---\n\n".join(res))]

    # ------------------------------------------------------------------ #
    elif name == "list_collections":
        rows = conn.execute(
            """
            SELECT c.name, COUNT(DISTINCT d.parent_id)
            FROM collections c
            LEFT JOIN docs d ON c.name = d.collection
            GROUP BY c.name
            """
        ).fetchall()
        lines = [f"• {r[0]}: {r[1]} dokument" for r in rows]
        text  = "Databasens samlingar:\n" + "\n".join(lines) if lines else "Inga samlingar än."
        return [types.TextContent(type="text", text=text)]

    # ------------------------------------------------------------------ #
    elif name == "delete_documents":
        """
        Tar bort specifika dokument eller rensar hela samlingen.

        Args:
            collection: Samlingsnamn att ta bort dokument från.
            ids: Lista med parent_id:n. Om listan är tom tas ALLA
                 dokument i samlingen bort (samlingen behålls).

        Returns:
            Bekräftelse med antal borttagna dokument.
        """
        coll = arguments["collection"]
        ids  = arguments["ids"]

        # Kontrollera att samlingen existerar
        row = conn.execute(
            "SELECT name FROM collections WHERE name = ?", (coll,)
        ).fetchone()
        if not row:
            return [types.TextContent(
                type="text",
                text=f"Fel: Samlingen '{coll}' finns inte.",
            )]

        if ids:
            # Validera alla ID:n innan något tas bort
            missing  = [pid for pid in ids if not _doc_exists(coll, pid)]
            existing = [pid for pid in ids if _doc_exists(coll, pid)]

            if missing and not existing:
                # Inga av de begärda ID:na finns – returnera varning utan att göra något
                missing_list = ", ".join(f"'{m}'" for m in missing)
                return [types.TextContent(
                    type="text",
                    text=(
                        f"Varning: Dokumenten med ID {missing_list} hittades inte "
                        f"i samlingen '{coll}'. Inga dokument togs bort."
                    ),
                )]

            # Ta bort de som finns, atomiskt
            with conn:
                for parent_id in existing:
                    _purge_document(coll, parent_id)

            # Bygg svar med info om eventuellt saknade ID:n
            msg = f"✓ Tog bort {len(existing)} dokument från samlingen '{coll}'."
            if missing:
                missing_list = ", ".join(f"'{m}'" for m in missing)
                msg += (
                    f"\nVarning: Följande ID:n hittades inte och hoppades över: {missing_list}."
                )
            return [types.TextContent(type="text", text=msg)]
        else:
            # Tom ids-lista → rensa hela samlingen
            count_row = conn.execute(
                "SELECT COUNT(DISTINCT parent_id) FROM docs WHERE collection = ?",
                (coll,),
            ).fetchone()
            doc_count = count_row[0] if count_row else 0

            with conn:
                conn.execute(
                    "DELETE FROM vectors WHERE collection = ?", (coll,)
                )
                conn.execute(
                    "DELETE FROM docs WHERE collection = ?", (coll,)
                )
            debug_log(f"Rensade hela samlingen '{coll}' ({doc_count} dokument).")
            return [types.TextContent(
                type="text",
                text=(
                    f"✓ Samlingen '{coll}' är nu tom. "
                    f"{doc_count} dokument borttagna (samlingen finns kvar)."
                ),
            )]

    # ------------------------------------------------------------------ #
    elif name == "delete_collection":
        """
        Tar bort en hel samling inklusive alla dokument och metadata.

        Args:
            name: Namnet på samlingen som ska raderas.

        Returns:
            Bekräftelse på borttagning, eller felmeddelande om
            samlingen inte existerar.
        """
        coll = arguments["name"]

        # Kontrollera att samlingen existerar innan borttagning
        row = conn.execute(
            "SELECT name FROM collections WHERE name = ?", (coll,)
        ).fetchone()
        if not row:
            return [types.TextContent(
                type="text",
                text=f"Fel: Samlingen '{coll}' finns inte.",
            )]

        # Räkna dokument och segment för bekräftelsemeddelandet
        stats = conn.execute(
            """
            SELECT
                COUNT(DISTINCT parent_id) AS doc_count,
                COUNT(*)                  AS chunk_count
            FROM docs WHERE collection = ?
            """,
            (coll,),
        ).fetchone()
        doc_count   = stats[0] if stats else 0
        chunk_count = stats[1] if stats else 0

        # Atomisk borttagning: vectors, docs och collections i en transaktion
        with conn:
            conn.execute("DELETE FROM vectors WHERE collection = ?",    (coll,))
            conn.execute("DELETE FROM docs WHERE collection = ?",       (coll,))
            conn.execute("DELETE FROM collections WHERE name = ?",      (coll,))

        debug_log(f"Samling '{coll}' borttagen ({doc_count} dok, {chunk_count} segment).")
        return [types.TextContent(
            type="text",
            text=(
                f"✓ Samlingen '{coll}' är borttagen. "
                f"{doc_count} dokument och {chunk_count} segment raderade."
            ),
        )]

    # ------------------------------------------------------------------ #
    return [types.TextContent(type="text", text="Ej implementerat verktyg.")]


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="rag-bge",
                server_version="1.7.0",
                capabilities=ServerCapabilities(),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
