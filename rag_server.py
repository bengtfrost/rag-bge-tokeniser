#!/usr/bin/env python3
import json, os, re, sqlite3, asyncio, httpx, sqlite_vec, sys, time
from typing import Any

# --- MILJÖINSTÄLLNINGAR ---
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import ServerCapabilities
import mcp.server.stdio
import mcp.types as types
from transformers import AutoTokenizer
from pypdf import PdfReader

# ========== HJÄLPMETODER & VERKTYG ==========================================


def debug_log(msg: str):
    """Loggar till stderr så att MCP-protokollet på stdout inte störs."""
    print(f"[*] RAG-LOG: {msg}", file=sys.stderr, flush=True)


def extract_text_from_file(file_path: str, encoding: str = "utf-8") -> str:
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


def progress_bar(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0%"
    pct = current / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {int(pct * 100)}%"


def fmt_seconds(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60}s"


def safe_doc_id(filename: str) -> str:
    return filename.lower()


# ========== KONFIGURATION (med miljövariabler) ================================
DB_PATH = os.path.expanduser("~/.local/share/rag-bge-tokeniser/vectors.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
TOKENISER_CACHE = os.path.expanduser("~/.config/rag-bge-tokeniser/tokeniser_cache")
TIMEOUT = 14400

EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "local-llama-server-embed")
RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "local-llama-server-rerank")

EMBED_URL = os.environ.get("RAG_EMBED_URL", "http://localhost:4000/v1/embeddings")
RERANK_URL = os.environ.get("RAG_RERANK_URL", "http://localhost:4000/rerank")

debug_log(f"Enhetlig routing aktiv via Agentgateway (:4000)")
debug_log(f"  -> Embeddings-modell: {EMBED_MODEL}")
debug_log(f"  -> Reranking-modell:  {RERANK_MODEL}")

CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "64"))
EMBED_BATCH_SIZE = int(os.environ.get("RAG_EMBED_BATCH_SIZE", "8"))
RERANK_CANDIDATES = int(os.environ.get("RAG_RERANK_CANDIDATES", "20"))
RERANK_MIN_SCORE = float(os.environ.get("RAG_RERANK_MIN_SCORE", "0.1"))
MAX_CONCURRENT_FILES = int(os.environ.get("RAG_MAX_CONCURRENT", "4"))
SUPPORTED_EXTENSIONS: set[str] = {".txt", ".pdf", ".md", ".rst", ".text"}

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
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÅÄÖ])", text)
    return [s.strip() for s in sentences if len(s.strip()) > 2]


def chunk_text_exact(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    tk = get_tokenizer()
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    sentence_tokens: list[int] = [
        len(tk.encode(s, add_special_tokens=False)) for s in sentences
    ]
    chunks: list[str] = []
    current_sents: list[str] = []
    current_tokens: int = 0
    current_sent_tokens: list[int] = []
    for i, (sent, s_tok) in enumerate(zip(sentences, sentence_tokens)):
        if s_tok > max_tokens:
            if current_sents:
                chunks.append(" ".join(current_sents))
                current_sents, current_sent_tokens = [], []
                current_tokens = 0
            chunks.append(sent[: max_tokens * 4])
            continue
        if current_tokens + s_tok > max_tokens and current_sents:
            chunks.append(" ".join(current_sents))
            overlap_sents: list[str] = []
            overlap_tok = 0
            for s, t in zip(reversed(current_sents), reversed(current_sent_tokens)):
                if overlap_tok + t > overlap_tokens:
                    break
                overlap_sents.insert(0, s)
                overlap_tok += t
            current_sents = overlap_sents
            current_sent_tokens = [
                sentence_tokens[idx] for idx in range(i - len(current_sents), i)
            ]
            current_tokens = overlap_tok
        current_sents.append(sent)
        current_sent_tokens.append(s_tok)
        current_tokens += s_tok
    if current_sents:
        chunks.append(" ".join(current_sents))
    return chunks


# ========== SÖKEXPANSION (MIDDLEWARE) ========================================


def expand_query_middleware(query: str) -> str:
    broad_keywords = [
        "berätta om",
        "vad är",
        "förklara",
        "beskriv",
        "vad handlar om",
        "vad innebär",
        "vad betyder",
        "hur fungerar",
        "vad gör",
    ]
    query_lower = query.lower()
    needs_expansion = any(keyword in query_lower for keyword in broad_keywords)
    if not needs_expansion:
        return query
    debug_log(f"Bred sökfråga detekterad. Initierar sökexpansion för: '{query}'")
    prompt = (
        f"Du är en sökmotorsassistent. Generera 3-5 synonymer eller relaterade juridiska/tekniska sökord på svenska "
        f'för att bredda en sökning på: "{query}". Svara ENDAST med sökorden separerade med mellanslag. '
        f"Ingen introduktion, inga punktlistor och inga citattecken."
    )
    payload = {
        "model": "local-llama-server",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 50,
    }
    try:
        import urllib.request

        req = urllib.request.Request(
            "http://localhost:4000/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-unused",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3.0) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            expanded = res_data["choices"][0]["message"]["content"].strip()
            expanded = re.sub(r"[\"\']", "", expanded)
            expanded = " ".join(expanded.split())
            if expanded and expanded.lower() != "null":
                optimized_query = f"{query} {expanded}"
                debug_log(f"Sökning expanderad till: '{optimized_query}'")
                return optimized_query
    except Exception as e:
        debug_log(
            f"Sökexpansion API-anrop misslyckades ({e}). Använder statisk fallback."
        )
    if any(term in query_lower for term in ["rf", "regeringsform", "lag"]):
        optimized_query = (
            f"{query} grundlag författning lagstiftning rättskälla paragrafer riksdag"
        )
    else:
        optimized_query = (
            f"{query} definition förklaring sammanfattning bakgrund information"
        )
    debug_log(f"Sökning expanderad (fallback) till: '{optimized_query}'")
    return optimized_query


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
    texts: list[str], label: str = "", progress_offset: int = 0, progress_total: int = 0
) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    total_batches = -(-len(texts) // EMBED_BATCH_SIZE)
    job_total = progress_total if progress_total > 0 else len(texts)
    prefix = f"[{label}] " if label else ""
    t_start = time.monotonic()
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for batch_num, i in enumerate(range(0, len(texts), EMBED_BATCH_SIZE), start=1):
            batch_texts = texts[i : i + EMBED_BATCH_SIZE]
            resp = await client.post(
                EMBED_URL, json={"input": batch_texts, "model": EMBED_MODEL}
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            all_embeddings.extend(
                d["embedding"] for d in sorted(data, key=lambda x: x["index"])
            )
            elapsed = time.monotonic() - t_start
            done_chunks = progress_offset + i + len(batch_texts)
            bar = progress_bar(done_chunks, job_total)
            eta_str = (
                "ETA: beräknar..."
                if batch_num == 1
                else f"ETA: {fmt_seconds((job_total - done_chunks) * (elapsed / max(i + len(batch_texts), 1)))}"
            )
            debug_log(
                f"{prefix}Embeddings batch {batch_num}/{total_batches} {bar}  {eta_str}"
            )
    return all_embeddings


async def rerank(
    query: str, documents: list[str], top_n: int
) -> list[tuple[int, float]]:
    if not documents:
        return []
    debug_log(f"Rerankar {len(documents)} kandidater via Agentgateway...")
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(
            RERANK_URL,
            json={"model": RERANK_MODEL, "query": query, "documents": documents},
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
    conn.execute("DELETE FROM vectors WHERE id LIKE ?", (f"{parent_id}_ch%",))
    conn.execute(
        "DELETE FROM docs WHERE parent_id = ? AND collection = ?", (parent_id, coll)
    )


def _doc_exists(coll: str, parent_id: str) -> bool:
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
    with conn:
        _purge_document(coll, doc_id)
        vectors_data, docs_data = [], []
        for idx, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_ch{idx}"
            vectors_data.append((chunk_id, coll, json.dumps(emb)))
            docs_data.append((chunk_id, coll, chunk_text, doc_id, idx))
        conn.executemany(
            "INSERT INTO vectors (id, collection, embedding) VALUES (?, ?, ?)",
            vectors_data,
        )
        conn.executemany(
            "INSERT INTO docs (id, collection, text, parent_id, chunk_index) VALUES (?, ?, ?, ?, ?)",
            docs_data,
        )
    elapsed = time.monotonic() - t0
    debug_log(f"'{doc_id}': klar – {len(chunks)} segment på {fmt_seconds(elapsed)}.")
    return len(chunks)


# ========== MCP VERKTYGSDEFINITIONER ========================================
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
            description="Läs och indexera en fil direkt från disk. Sätt force=true för att tvinga re-indexering.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {"type": "string"},
                    "file_path": {"type": "string"},
                    "document_id": {"type": "string"},
                    "encoding": {"type": "string"},
                    "force": {"type": "boolean"},
                },
                "required": ["collection", "file_path"],
            },
        ),
        types.Tool(
            name="ingest_directory",
            description="Indexera alla matchande filer i en katalog med parallell bearbetning.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {"type": "string"},
                    "directory_path": {"type": "string"},
                    "file_extensions": {"type": "array", "items": {"type": "string"}},
                    "encoding": {"type": "string"},
                    "force": {"type": "boolean"},
                },
                "required": ["collection", "directory_path"],
            },
        ),
        types.Tool(
            name="add_documents",
            description="Indexera råtext-strängar direkt.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {"type": "string"},
                    "ids": {"type": "array", "items": {"type": "string"}},
                    "documents": {"type": "array", "items": {"type": "string"}},
                    "force": {"type": "boolean"},
                },
                "required": ["collection", "ids", "documents"],
            },
        ),
        types.Tool(
            name="query",
            description="Sök i samlingen med semantisk sökning, automatisk sökexpansion (middleware) och reranking",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {"type": "string"},
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
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
            description="Ta bort ett eller flera indexerade dokument från en samling. Tom lista = rensa hela samlingen.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {"type": "string"},
                    "ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["collection", "ids"],
            },
        ),
        types.Tool(
            name="delete_collection",
            description="Ta bort en hel samling inklusive alla dokument, chunks och metadata.",
            inputSchema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "create_collection":
        coll_name = arguments["name"]
        conn.execute(
            "INSERT OR IGNORE INTO collections (name) VALUES (?)", (coll_name,)
        )
        conn.commit()
        return [
            types.TextContent(type="text", text=f"Samlingen '{coll_name}' är nu redo.")
        ]

    elif name == "ingest_file":
        coll, file_path, force = (
            arguments["collection"],
            arguments["file_path"],
            arguments.get("force", False),
        )
        encoding = arguments.get("encoding", "utf-8")
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
        debug_log(
            f"Startar ingest av '{os.path.basename(file_path)}' ({os.path.getsize(file_path) // 1024} KB, doc_id='{doc_id}')...."
        )
        conn.execute("INSERT OR IGNORE INTO collections (name) VALUES (?)", (coll,))
        conn.commit()
        if _doc_exists(coll, doc_id):
            if not force:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Varning: '{doc_id}' är redan indexerat i '{coll}'. Inga ändringar gjordes.",
                    )
                ]
            debug_log(f"force=True: re-indexerar '{doc_id}'...")
        t0 = time.monotonic()
        chunks = chunk_text_exact(text, CHUNK_SIZE, CHUNK_OVERLAP)
        total = await _index_chunks(coll, doc_id, chunks)
        elapsed = time.monotonic() - t0
        if total == 0:
            return [types.TextContent(type="text", text="Ingen text att indexera.")]
        return [
            types.TextContent(
                type="text",
                text=f"✓ Klar! {'Re-indexerade' if force else 'Indexerade'} {total} segment från '{os.path.basename(file_path)}' (doc_id='{doc_id}') i '{coll}' på {fmt_seconds(elapsed)}.",
            )
        ]

    elif name == "ingest_directory":
        coll, dir_path, force = (
            arguments["collection"],
            arguments["directory_path"],
            arguments.get("force", False),
        )
        encoding = arguments.get("encoding", "utf-8")
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
            return [
                types.TextContent(
                    type="text",
                    text=f"Inga filer med ändelserna [{', '.join(sorted(exts))}] hittades i {dir_path}.",
                )
            ]
        debug_log(
            f"Katalogindexering startar: {len(files)} filer i '{dir_path}' (force={force})..."
        )
        conn.execute("INSERT OR IGNORE INTO collections (name) VALUES (?)", (coll,))
        conn.commit()
        debug_log("Läser och chunkar alla filer...")
        prepared, total_chunks_estimate = [], 0
        for filename in files:
            fp, doc_id = os.path.join(dir_path, filename), safe_doc_id(filename)
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
            f"Totalt: {len(files)} filer, {total_chunks_estimate} segment. Startar parallell embeddings (max {MAX_CONCURRENT_FILES} samtidigt)..."
        )
        sem = asyncio.Semaphore(MAX_CONCURRENT_FILES)

        async def index_one(
            fp: str, doc_id: str, chunks: list[str], err: str | None
        ) -> tuple[str, int, str]:
            if err:
                return (os.path.basename(fp), 0, err)
            if not chunks:
                return (os.path.basename(fp), 0, "tom fil")
            async with sem:
                if _doc_exists(coll, doc_id) and not force:
                    return (os.path.basename(fp), 0, "redan indexerad")
                try:
                    n = await _index_chunks(coll, doc_id, chunks)
                    return (
                        os.path.basename(fp),
                        n,
                        f"{n} segment {'re-indexerad' if force else 'indexerad'}",
                    )
                except Exception as e:
                    return (os.path.basename(fp), 0, f"fel – {e}")

        tasks = [
            index_one(fp, doc_id, chunks, err) for (fp, doc_id, chunks, err) in prepared
        ]
        results = await asyncio.gather(*tasks)
        total_segments = sum(r[1] for r in results)
        result_lines = []
        for filename, seg, msg in results:
            if seg > 0 or "fel" in msg or "redan indexerad" in msg:
                prefix = (
                    "✓"
                    if seg > 0
                    else ("⚠" if "hoppades" in msg or "redan indexerad" in msg else "✗")
                )
                result_lines.append(f"  {prefix} {filename}: {msg}")
        summary = f"Katalogindexering klar. {len(files)} filer · {total_segments} segment · samling: '{coll}'\n"
        debug_log(summary.strip())
        return [types.TextContent(type="text", text=summary + "\n".join(result_lines))]

    elif name == "add_documents":
        coll, ids, docs, force = (
            arguments["collection"],
            arguments["ids"],
            arguments["documents"],
            arguments.get("force", False),
        )
        conn.execute("INSERT OR IGNORE INTO collections (name) VALUES (?)", (coll,))
        conn.commit()
        total_chunks, skipped, t0 = 0, [], time.monotonic()
        for i, (doc_id, doc_text) in enumerate(zip(ids, docs), start=1):
            debug_log(f"Dokument {i}/{len(ids)}: '{doc_id}'")
            if _doc_exists(coll, doc_id):
                if not force:
                    skipped.append(doc_id)
                    continue
                debug_log(f"  force=True: re-indexerar '{doc_id}'...")
            chunks = chunk_text_exact(doc_text, CHUNK_SIZE, CHUNK_OVERLAP)
            total_chunks += await _index_chunks(coll, doc_id, chunks)
        elapsed = time.monotonic() - t0
        if total_chunks == 0 and not skipped:
            return [types.TextContent(type="text", text="Ingen text att indexera.")]
        msg = (
            f"✓ Indexerade {total_chunks} segment i '{coll}' på {fmt_seconds(elapsed)}."
        )
        if skipped:
            msg += f"\nVarning: Redan indexerade (hoppades över): {', '.join(f"'{s}'" for s in skipped)}."
        return [types.TextContent(type="text", text=msg)]

    elif name == "query":
        coll, query, top_k = (
            arguments["collection"],
            arguments["query"],
            arguments.get("top_k", 5),
        )
        optimized_query = expand_query_middleware(query)
        query_preview = optimized_query[:80] + (
            "..." if len(optimized_query) > 80 else ""
        )
        debug_log(f"Query: '{query_preview}' i samling '{coll}'")
        t0 = time.monotonic()
        query_emb = (await get_embeddings([optimized_query], label="query"))[0]
        debug_log(f"Hämtar {RERANK_CANDIDATES} ANN-kandidater...")
        cursor = conn.execute(
            "SELECT id FROM vectors WHERE collection = ? AND embedding MATCH ? AND k = ?",
            (coll, json.dumps(query_emb), RERANK_CANDIDATES),
        )
        chunk_ids = [row[0] for row in cursor.fetchall()]
        if not chunk_ids:
            return [types.TextContent(type="text", text="Hittade inget relevant.")]
        debug_log(f"{len(chunk_ids)} kandidater hämtade, hämtar text...")
        rows = conn.execute(
            f"SELECT id, text, parent_id FROM docs WHERE id IN ({','.join(['?'] * len(chunk_ids))})",
            chunk_ids,
        ).fetchall()
        doc_map = {row[0]: (row[1], row[2]) for row in rows}
        valid_ids = [cid for cid in chunk_ids if cid in doc_map]
        reranked = await rerank(
            optimized_query, [doc_map[cid][0] for cid in valid_ids], top_k
        )
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
            f"[{i}] (Källa: {doc_map[valid_ids[idx]][1]}) Score: {score:.4f}\n{doc_map[valid_ids[idx]][0]}"
            for i, (idx, score) in enumerate(reranked, start=1)
        ]
        return [types.TextContent(type="text", text="\n\n---\n\n".join(res))]

    elif name == "list_collections":
        rows = conn.execute(
            "SELECT c.name, COUNT(DISTINCT d.parent_id) FROM collections c LEFT JOIN docs d ON c.name = d.collection GROUP BY c.name"
        ).fetchall()
        lines = [f"• {r[0]}: {r[1]} dokument" for r in rows]
        return [
            types.TextContent(
                type="text",
                text="Databasens samlingar:\n" + "\n".join(lines)
                if lines
                else "Inga samlingar än.",
            )
        ]

    elif name == "delete_documents":
        coll, ids = arguments["collection"], arguments["ids"]
        if not conn.execute(
            "SELECT name FROM collections WHERE name = ?", (coll,)
        ).fetchone():
            return [
                types.TextContent(
                    type="text", text=f"Fel: Samlingen '{coll}' finns inte."
                )
            ]
        if ids:
            missing = [pid for pid in ids if not _doc_exists(coll, pid)]
            existing = [pid for pid in ids if _doc_exists(coll, pid)]
            if missing and not existing:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Varning: Dokumenten {', '.join(f"'{m}'" for m in missing)} hittades inte i '{coll}'.",
                    )
                ]
            with conn:
                for parent_id in existing:
                    _purge_document(coll, parent_id)
            msg = f"✓ Tog bort {len(existing)} dokument från samlingen '{coll}'."
            if missing:
                msg += f"\nVarning: Hittades inte och hoppades över: {', '.join(f"'{m}'" for m in missing)}."
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
                    text=f"✓ Samlingen '{coll}' är nu tom. {doc_count} dokument borttagna.",
                )
            ]

    elif name == "delete_collection":
        coll = arguments["name"]
        if not conn.execute(
            "SELECT name FROM collections WHERE name = ?", (coll,)
        ).fetchone():
            return [
                types.TextContent(
                    type="text", text=f"Fel: Samlingen '{coll}' finns inte."
                )
            ]
        stats = conn.execute(
            "SELECT COUNT(DISTINCT parent_id), COUNT(*) FROM docs WHERE collection = ?",
            (coll,),
        ).fetchone()
        doc_count, chunk_count = (stats[0] if stats else 0), (stats[1] if stats else 0)
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
                text=f"✓ Samlingen '{coll}' är borttagen. {doc_count} dokument och {chunk_count} segment raderade.",
            )
        ]

    return [types.TextContent(type="text", text="Ej implementerat verktyg.")]


# ========== APPLIKATIONSSTART & LIFECYCLE ===================================


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="rag-bge",
                server_version="2.0.0",
                capabilities=ServerCapabilities(),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
