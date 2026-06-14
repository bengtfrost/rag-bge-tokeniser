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
from typing import Optional, List, Tuple, Any

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
from pypdf import PdfReader

# ========== KONFIGURATION (med miljövariabler) ================================
DB_PATH = os.path.expanduser("~/.local/share/rag-bge-tokeniser/vectors.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

TOKENISER_CACHE = os.path.expanduser("~/.config/rag-bge-tokeniser/tokeniser_cache")

EMBED_URL = "http://localhost:11435/v1/embeddings"
RERANK_URL = "http://localhost:11436/rerank"
TIMEOUT = 7200  # 2 timmar för tunga juridiska dokument

# Läs konfiguration från miljövariabler (med standardvärden)
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "64"))
EMBED_BATCH_SIZE = int(os.environ.get("RAG_EMBED_BATCH_SIZE", "8"))
RERANK_CANDIDATES = int(os.environ.get("RAG_RERANK_CANDIDATES", "20"))
RERANK_MIN_SCORE = float(os.environ.get("RAG_RERANK_MIN_SCORE", "0.1"))
MAX_CONCURRENT_FILES = int(os.environ.get("RAG_MAX_CONCURRENT", "3"))

# Standardändelser som stöds av extract_text_from_file
SUPPORTED_EXTENSIONS: set[str] = {".txt", ".pdf", ".md", ".rst", ".text"}
# ===========================================================================


# ========== HJÄLPFUNKTIONER ================================================


def extract_text_from_file(file_path: str, encoding: str = "utf-8") -> str:
    """
    Extraherar text baserat på filändelse.

    Args:
        file_path: Absolut sökväg till filen.
        encoding:  Teckenkodning för textfiler (ignoreras för PDF).

    Returns:
        Extraherad text som sträng.

    Raises:
        ValueError: Om PDF-extrahering misslyckas.
        UnicodeDecodeError / OSError: Vid problem med textfiler.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        try:
            reader = PdfReader(file_path)
            text_parts = [p.extract_text() for p in reader.pages if p.extract_text()]
            return "\n\n".join(text_parts)
        except Exception as e:
            raise ValueError(f"PDF-extrahering misslyckades: {e}")
    else:
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()


def debug_log(msg: str):
    """Loggar till stderr så att MCP-protokollet på stdout inte störs."""
    print(f"[*] RAG-LOG: {msg}", file=sys.stderr, flush=True)


def progress_bar(current: int, total: int, width: int = 30) -> str:
    """Returnerar en ASCII-progressbar, t.ex. [████████░░░░░░░] 53%"""
    if total <= 0:
        return "[░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0%"
    pct = current / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {int(pct * 100)}%"


def fmt_seconds(seconds: float) -> str:
    """Formaterar sekunder till läsbar sträng, t.ex. '2m 14s'."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60}s"


def safe_doc_id(filename: str) -> str:
    """
    Genererar ett konsekvent doc_id från ett filnamn.

    Regel: stam + ändelse, allt lowercase, utan sökväg.
    Exempel: 'Testfil.PDF' → 'testfil.pdf', 'rapport.txt' → 'rapport.txt'

    Inkluderar ändelsen för att undvika kollisioner när samma stam
    förekommer i flera format (t.ex. 'lag.pdf' och 'lag.txt').

    Args:
        filename: Filnamnet (utan sökväg).

    Returns:
        Normaliserat doc_id.
    """
    return filename.lower()


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
    tills overlap_tokens uppnås. Optimering: förberäknade tokenlängder.

    Args:
        text:          Råtext att chunka.
        max_tokens:    Maximalt antal tokens per chunk.
        overlap_tokens: Antal tokens överlapp mellan chunks.

    Returns:
        Lista med textchunks.
    """
    tk = get_tokenizer()
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    # Förberäkna tokenlängder för varje mening (sparar tid)
    sentence_tokens: list[int] = [
        len(tk.encode(s, add_special_tokens=False)) for s in sentences
    ]

    chunks: list[str] = []
    current_sents: list[str] = []
    current_tokens: int = 0
    # Vi måste även hålla tokenlängder för meningarna i current_sents
    current_sent_tokens: list[int] = []

    for i, (sent, s_tok) in enumerate(zip(sentences, sentence_tokens)):
        # Hantera extremt långa meningar – trunkera på teckennivå
        if s_tok > max_tokens:
            if current_sents:
                chunks.append(" ".join(current_sents))
                current_sents, current_sent_tokens = [], []
                current_tokens = 0
            chunks.append(sent[: max_tokens * 4])
            continue

        if current_tokens + s_tok > max_tokens and current_sents:
            # Spara nuvarande chunk
            chunks.append(" ".join(current_sents))

            # Token-exakt överlapp: bygg bakifrån tills vi når overlap_tokens
            overlap_sents: list[str] = []
            overlap_tok = 0
            # Iterera baklänges över current_sents och deras tokenlängder
            for s, t in zip(reversed(current_sents), reversed(current_sent_tokens)):
                if overlap_tok + t > overlap_tokens:
                    break
                overlap_sents.insert(0, s)
                overlap_tok += t

            current_sents = overlap_sents
            current_sent_tokens = [
                sentence_tokens[idx] for idx in range(i - len(current_sents), i)
            ]  # O(m) men m är litet
            current_tokens = overlap_tok

        current_sents.append(sent)
        current_sent_tokens.append(s_tok)
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_collection ON docs(collection)")
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

    Args:
        texts:           Lista med textsträngar att embedda.
        label:           Visningsnamn i progress-logg.
        progress_offset: Antal redan bearbetade chunks (för katalogjobb).
        progress_total:  Totalt antal chunks i hela jobbet (för katalogjobb).

    Returns:
        Lista med embedding-vektorer i samma ordning som indata.
    """
    all_embeddings: list[list[float]] = []
    total_batches = -(-len(texts) // EMBED_BATCH_SIZE)  # ceil division
    job_total = progress_total if progress_total > 0 else len(texts)
    prefix = f"[{label}] " if label else ""
    t_start = time.monotonic()

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
            elapsed = time.monotonic() - t_start
            done_chunks = progress_offset + i + len(batch_texts)
            bar = progress_bar(done_chunks, job_total)

            if batch_num == 1:
                eta_str = "ETA: beräknar..."
            else:
                avg_per_chunk = elapsed / max(i + len(batch_texts), 1)
                remaining = job_total - done_chunks
                eta_str = f"ETA: {fmt_seconds(remaining * avg_per_chunk)}"

            debug_log(
                f"{prefix}Embeddings batch {batch_num}/{total_batches} {bar}  {eta_str}"
            )

    return all_embeddings


async def rerank(
    query: str, documents: list[str], top_n: int
) -> list[tuple[int, float]]:
    """
    Skickar kandidater till reranker-server (port 11436).

    Args:
        query:     Sökfrågan.
        documents: Lista med kandidattexter.
        top_n:     Maximalt antal resultat att returnera.

    Returns:
        Lista med (original_index, relevance_score)-tupler, sorterad fallande.
    """
    if not documents:
        return []
    debug_log(f"Rerankar {len(documents)} kandidater...")
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(
            RERANK_URL, json={"query": query, "documents": documents}
        )
        resp.raise_for_status()
        results = resp.json()["results"]

    indexed = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
    filtered = [r for r in indexed if r["relevance_score"] > RERANK_MIN_SCORE]
    debug_log(
        f"Reranking klar: {len(filtered)} träffar över tröskel {RERANK_MIN_SCORE}."
    )
    return [(r["index"], r["relevance_score"]) for r in filtered[:top_n]]


# ========== INDEXERINGSHJÄLP ===============================================


def _purge_document(coll: str, parent_id: str):
    """
    Tar bort alla befintliga chunks för ett dokument.

    Används vid re-ingest (force=True) och delete_documents.
    Måste köras inuti en aktiv transaktion (with conn:).

    Args:
        coll:      Samlingsnamn.
        parent_id: Dokumentets parent_id.
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
        parent_id: Dokumentets parent_id.

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
    chunks: list[str],
    progress_offset: int = 0,
    progress_total: int = 0,
) -> int:
    """
    Beräknar embeddings för färdiga chunks och sparar till DB.

    Rensar alltid befintliga chunks för doc_id innan inläggning
    (atomisk purge + insert i samma transaktion).

    Args:
        coll:            Samlingsnamn.
        doc_id:          Dokumentets unika ID.
        chunks:          Redan förberedda textchunks.
        progress_offset: Antal redan bearbetade chunks (för katalogjobb).
        progress_total:  Totalt antal chunks i hela jobbet (för katalogjobb).

    Returns:
        Antal indexerade segment.
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

    # Atomisk: rensa gamla + skriv nya i samma transaktion med executemany
    with conn:
        _purge_document(coll, doc_id)

        # Förbered data för vectors-tabellen
        vectors_data = []
        docs_data = []
        for idx, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_ch{idx}"
            vectors_data.append((chunk_id, coll, json.dumps(emb)))
            docs_data.append((chunk_id, coll, chunk_text, doc_id, idx))

        conn.executemany(
            "INSERT INTO vectors (id, collection, embedding) VALUES (?, ?, ?)",
            vectors_data,
        )
        conn.executemany(
            "INSERT INTO docs (id, collection, text, parent_id, chunk_index) "
            "VALUES (?, ?, ?, ?, ?)",
            docs_data,
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
                "Läs och indexera en fil (text, PDF eller Markdown) direkt från disk "
                "utan att skicka innehållet genom LLM-kontextfönstret. "
                "Föredra detta framför add_documents för stora filer. "
                "Sätt force=true för att tvinga re-indexering av ett redan indexerat dokument."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Samlingsnamn",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Absolut sökväg till filen",
                    },
                    "document_id": {
                        "type": "string",
                        "description": (
                            "Valfritt ID för dokumentet. Standard: filnamnet i lowercase "
                            "(t.ex. 'testfil.pdf'). Ange explicit för att undvika kollisioner."
                        ),
                    },
                    "encoding": {
                        "type": "string",
                        "description": "Teckenkodning för textfiler (standard: utf-8, ignoreras för PDF)",
                    },
                    "force": {
                        "type": "boolean",
                        "description": (
                            "Om true: ta bort och re-indexera dokumentet även om det redan finns. "
                            "Standard: false (returnerar varning om dokumentet redan är indexerat)."
                        ),
                    },
                },
                "required": ["collection", "file_path"],
            },
        ),
        types.Tool(
            name="ingest_directory",
            description=(
                "Indexera filer i en katalog direkt från disk. "
                "Stödjer .txt, .pdf, .md och andra textformat. "
                "Chunkar varje fil en enda gång, kör sedan embeddings parallellt. "
                "Rapporterar progress per fil med ETA. "
                "Sätt force=true för att tvinga re-indexering av redan indexerade filer."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Samlingsnamn",
                    },
                    "directory_path": {
                        "type": "string",
                        "description": "Absolut sökväg till katalogen",
                    },
                    "file_extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Lista med filändelser att indexera (skiftlägesokänsliga). "
                            'Standard: [".txt", ".pdf", ".md", ".rst", ".text"]. '
                            'Exempel: [".pdf"] indexerar bara PDF-filer.'
                        ),
                    },
                    "encoding": {
                        "type": "string",
                        "description": "Teckenkodning för textfiler (standard: utf-8, ignoreras för PDF)",
                    },
                    "force": {
                        "type": "boolean",
                        "description": (
                            "Om true: re-indexera alla filer, även redan indexerade. "
                            "Standard: false (hoppar över redan indexerade filer med varning)."
                        ),
                    },
                },
                "required": ["collection", "directory_path"],
            },
        ),
        types.Tool(
            name="add_documents",
            description=(
                "Indexera råtext-strängar direkt (t.ex. korta anteckningar eller urklipp). "
                "OBS: För filer på disk, använd 'ingest_file' för att undvika "
                "att belasta LLM-kontextfönstret med stora mängder text."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {"type": "string"},
                    "ids": {"type": "array", "items": {"type": "string"}},
                    "documents": {"type": "array", "items": {"type": "string"}},
                    "force": {
                        "type": "boolean",
                        "description": (
                            "Om true: re-indexera dokument även om de redan finns. "
                            "Standard: false."
                        ),
                    },
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
                    "query": {"type": "string"},
                    "top_k": {
                        "type": "integer",
                        "description": "Antal resultat (standard: 5)",
                    },
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
                "Ta bort ett eller flera indexerade dokument från en samling. "
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
                    "name": {
                        "type": "string",
                        "description": "Namnet på samlingen som ska tas bort",
                    },
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
        conn.execute(
            "INSERT OR IGNORE INTO collections (name) VALUES (?)", (coll_name,)
        )
        conn.commit()
        return [
            types.TextContent(type="text", text=f"Samlingen '{coll_name}' är nu redo.")
        ]

    # ------------------------------------------------------------------ #
    elif name == "ingest_file":
        """
        Indexerar en enskild fil från disk.
        """
        coll = arguments["collection"]
        file_path = arguments["file_path"]
        encoding = arguments.get("encoding", "utf-8")
        force = arguments.get("force", False)

        doc_id = arguments.get("document_id") or safe_doc_id(
            os.path.basename(file_path)
        )

        if not os.path.isfile(file_path):
            return [
                types.TextContent(
                    type="text", text=f"Fel: Filen hittades inte: {file_path}"
                )
            ]

        try:
            text = extract_text_from_file(file_path, encoding=encoding)
        except Exception as e:
            return [types.TextContent(type="text", text=f"Fel vid läsning av fil: {e}")]

        file_size_kb = os.path.getsize(file_path) // 1024
        debug_log(
            f"Startar ingest av '{os.path.basename(file_path)}' "
            f"({file_size_kb} KB, doc_id='{doc_id}')..."
        )

        conn.execute("INSERT OR IGNORE INTO collections (name) VALUES (?)", (coll,))
        conn.commit()

        if _doc_exists(coll, doc_id):
            if not force:
                return [
                    types.TextContent(
                        type="text",
                        text=(
                            f"Varning: Dokumentet '{doc_id}' är redan indexerat "
                            f"i samlingen '{coll}'. Inga ändringar gjordes. "
                            f"Använd force=true för att tvinga re-indexering, "
                            f"eller delete_documents för att ta bort det först."
                        ),
                    )
                ]
            debug_log(f"force=True: re-indexerar '{doc_id}'...")

        t0 = time.monotonic()
        chunks = chunk_text_exact(text, CHUNK_SIZE, CHUNK_OVERLAP)
        total = await _index_chunks(coll, doc_id, chunks)
        elapsed = time.monotonic() - t0

        if total == 0:
            return [types.TextContent(type="text", text="Ingen text att indexera.")]

        action = "Re-indexerade" if force else "Indexerade"
        return [
            types.TextContent(
                type="text",
                text=(
                    f"✓ Klar! {action} {total} segment från "
                    f"'{os.path.basename(file_path)}' (doc_id='{doc_id}') "
                    f"i samlingen '{coll}' på {fmt_seconds(elapsed)}."
                ),
            )
        ]

    # ------------------------------------------------------------------ #
    elif name == "ingest_directory":
        """
        Indexerar alla matchande filer i en katalog med parallell bearbetning.
        """
        coll = arguments["collection"]
        dir_path = arguments["directory_path"]
        encoding = arguments.get("encoding", "utf-8")
        force = arguments.get("force", False)

        raw_exts = arguments.get("file_extensions", list(SUPPORTED_EXTENSIONS))
        exts: set[str] = {
            e.lower() if e.startswith(".") else f".{e.lower()}" for e in raw_exts
        }

        if not os.path.isdir(dir_path):
            return [
                types.TextContent(
                    type="text", text=f"Fel: Katalogen hittades inte: {dir_path}"
                )
            ]

        files = sorted(
            f for f in os.listdir(dir_path) if os.path.splitext(f)[1].lower() in exts
        )

        if not files:
            ext_str = ", ".join(sorted(exts))
            return [
                types.TextContent(
                    type="text",
                    text=f"Inga filer med ändelserna [{ext_str}] hittades i {dir_path}.",
                )
            ]

        debug_log(
            f"Katalogindexering startar: {len(files)} filer i '{dir_path}' "
            f"(force={force}, max_concurrent={MAX_CONCURRENT_FILES})..."
        )

        conn.execute("INSERT OR IGNORE INTO collections (name) VALUES (?)", (coll,))
        conn.commit()

        # ── Första pass: läs och chunka alla filer (sekventiellt, snabbt) ──
        debug_log("Läser och chunkar alla filer...")
        prepared: list[tuple[str, str, list[str], str | None]] = []
        total_chunks_estimate = 0

        for filename in files:
            fp = os.path.join(dir_path, filename)
            doc_id = safe_doc_id(filename)
            try:
                text = extract_text_from_file(fp, encoding=encoding)
                chunks = chunk_text_exact(text, CHUNK_SIZE, CHUNK_OVERLAP)
                prepared.append((fp, doc_id, chunks, None))
                total_chunks_estimate += len(chunks)
                debug_log(
                    f"  Chunkat: '{filename}' (doc_id='{doc_id}') → {len(chunks)} segment"
                )
            except Exception as e:
                prepared.append((fp, doc_id, [], str(e)))
                debug_log(f"  Fel vid läsning av '{filename}': {e}")

        debug_log(
            f"Totalt: {len(files)} filer, {total_chunks_estimate} segment. "
            f"Startar parallell embeddings (max {MAX_CONCURRENT_FILES} samtidigt)..."
        )

        # ── Andra pass: parallell indexering med semafor ──
        sem = asyncio.Semaphore(MAX_CONCURRENT_FILES)

        async def index_one(
            fp: str, doc_id: str, chunks: list[str], err: Optional[str]
        ) -> tuple[str, int, str]:
            if err:
                return (os.path.basename(fp), 0, err)
            if not chunks:
                return (os.path.basename(fp), 0, "tom fil")
            async with sem:
                if _doc_exists(coll, doc_id) and not force:
                    return (os.path.basename(fp), 0, "redan indexerad (hoppades över)")
                try:
                    n = await _index_chunks(coll, doc_id, chunks)
                    action = "re-indexerad" if force else "indexerad"
                    return (os.path.basename(fp), n, f"{n} segment {action}")
                except Exception as e:
                    return (os.path.basename(fp), 0, f"fel – {e}")

        tasks = [
            index_one(fp, doc_id, chunks, err) for (fp, doc_id, chunks, err) in prepared
        ]
        results = await asyncio.gather(*tasks)

        # Sammanställ resultat
        total_segments = sum(r[1] for r in results)
        result_lines = []
        for filename, seg, msg in results:
            if seg > 0 or "fel" in msg or "redan indexerad" in msg:
                prefix = "✓" if seg > 0 else ("⚠" if "hoppades" in msg else "✗")
                result_lines.append(f"  {prefix} {filename}: {msg}")

        summary = (
            f"Katalogindexering klar. "
            f"{len(files)} filer · {total_segments} segment · "
            f"samling: '{coll}'\n"
        )
        debug_log(summary.strip())
        return [types.TextContent(type="text", text=summary + "\n".join(result_lines))]

    # ------------------------------------------------------------------ #
    elif name == "add_documents":
        """
        Indexerar råtext-strängar som skickas direkt.
        """
        coll = arguments["collection"]
        ids = arguments["ids"]
        docs = arguments["documents"]
        force = arguments.get("force", False)

        conn.execute("INSERT OR IGNORE INTO collections (name) VALUES (?)", (coll,))
        conn.commit()

        total_chunks = 0
        skipped: list[str] = []
        t0 = time.monotonic()

        for i, (doc_id, doc_text) in enumerate(zip(ids, docs), start=1):
            debug_log(f"Dokument {i}/{len(ids)}: '{doc_id}'")

            if _doc_exists(coll, doc_id):
                if not force:
                    debug_log(f"  '{doc_id}' redan indexerat, hoppar över.")
                    skipped.append(doc_id)
                    continue
                debug_log(f"  force=True: re-indexerar '{doc_id}'...")

            chunks = chunk_text_exact(doc_text, CHUNK_SIZE, CHUNK_OVERLAP)
            n = await _index_chunks(coll, doc_id, chunks)
            total_chunks += n

        elapsed = time.monotonic() - t0
        if total_chunks == 0 and not skipped:
            return [types.TextContent(type="text", text="Ingen text att indexera.")]

        msg = (
            f"✓ Indexerade {total_chunks} segment i '{coll}' på {fmt_seconds(elapsed)}."
        )
        if skipped:
            skipped_list = ", ".join(f"'{s}'" for s in skipped)
            msg += (
                f"\nVarning: Följande dokument var redan indexerade och hoppades över: "
                f"{skipped_list}. Använd force=true eller delete_documents för att ersätta dem."
            )
        return [types.TextContent(type="text", text=msg)]

    # ------------------------------------------------------------------ #
    elif name == "query":
        coll = arguments["collection"]
        query = arguments["query"]
        top_k = arguments.get("top_k", 5)

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
        doc_map = {row[0]: (row[1], row[2]) for row in rows}
        valid_ids = [cid for cid in chunk_ids if cid in doc_map]

        reranked = await rerank(query, [doc_map[cid][0] for cid in valid_ids], top_k)

        if not reranked:
            return [
                types.TextContent(
                    type="text", text="Inga tillräckligt relevanta träffar."
                )
            ]

        elapsed = time.monotonic() - t0
        debug_log(
            f"Query klar på {fmt_seconds(elapsed)}, returnerar {len(reranked)} träffar."
        )

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
        text = (
            "Databasens samlingar:\n" + "\n".join(lines)
            if lines
            else "Inga samlingar än."
        )
        return [types.TextContent(type="text", text=text)]

    # ------------------------------------------------------------------ #
    elif name == "delete_documents":
        coll = arguments["collection"]
        ids = arguments["ids"]

        row = conn.execute(
            "SELECT name FROM collections WHERE name = ?", (coll,)
        ).fetchone()
        if not row:
            return [
                types.TextContent(
                    type="text",
                    text=f"Fel: Samlingen '{coll}' finns inte.",
                )
            ]

        if ids:
            missing = [pid for pid in ids if not _doc_exists(coll, pid)]
            existing = [pid for pid in ids if _doc_exists(coll, pid)]

            if missing and not existing:
                missing_list = ", ".join(f"'{m}'" for m in missing)
                return [
                    types.TextContent(
                        type="text",
                        text=(
                            f"Varning: Dokumenten med ID {missing_list} hittades inte "
                            f"i samlingen '{coll}'. Inga dokument togs bort."
                        ),
                    )
                ]

            with conn:
                for parent_id in existing:
                    _purge_document(coll, parent_id)

            msg = f"✓ Tog bort {len(existing)} dokument från samlingen '{coll}'."
            if missing:
                missing_list = ", ".join(f"'{m}'" for m in missing)
                msg += (
                    f"\nVarning: Följande ID:n hittades inte och hoppades över: "
                    f"{missing_list}."
                )
            return [types.TextContent(type="text", text=msg)]
        else:
            count_row = conn.execute(
                "SELECT COUNT(DISTINCT parent_id) FROM docs WHERE collection = ?",
                (coll,),
            ).fetchone()
            doc_count = count_row[0] if count_row else 0

            with conn:
                conn.execute("DELETE FROM vectors WHERE collection = ?", (coll,))
                conn.execute("DELETE FROM docs WHERE collection = ?", (coll,))

            debug_log(f"Rensade hela samlingen '{coll}' ({doc_count} dokument).")
            return [
                types.TextContent(
                    type="text",
                    text=(
                        f"✓ Samlingen '{coll}' är nu tom. "
                        f"{doc_count} dokument borttagna (samlingen finns kvar)."
                    ),
                )
            ]

    # ------------------------------------------------------------------ #
    elif name == "delete_collection":
        coll = arguments["name"]

        row = conn.execute(
            "SELECT name FROM collections WHERE name = ?", (coll,)
        ).fetchone()
        if not row:
            return [
                types.TextContent(
                    type="text",
                    text=f"Fel: Samlingen '{coll}' finns inte.",
                )
            ]

        stats = conn.execute(
            """
            SELECT
                COUNT(DISTINCT parent_id) AS doc_count,
                COUNT(*)                  AS chunk_count
            FROM docs WHERE collection = ?
            """,
            (coll,),
        ).fetchone()
        doc_count = stats[0] if stats else 0
        chunk_count = stats[1] if stats else 0

        with conn:
            conn.execute("DELETE FROM vectors WHERE collection = ?", (coll,))
            conn.execute("DELETE FROM docs WHERE collection = ?", (coll,))
            conn.execute("DELETE FROM collections WHERE name = ?", (coll,))

        debug_log(
            f"Samling '{coll}' borttagen ({doc_count} dok, {chunk_count} segment)."
        )
        return [
            types.TextContent(
                type="text",
                text=(
                    f"✓ Samlingen '{coll}' är borttagen. "
                    f"{doc_count} dokument och {chunk_count} segment raderade."
                ),
            )
        ]

    # ------------------------------------------------------------------ #
    return [types.TextContent(type="text", text="Ej implementerat verktyg.")]


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="rag-bge",
                server_version="2.0.0",  # Uppdaterad version efter optimeringar
                capabilities=ServerCapabilities(),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())

