"""SQLite database: schema, writing documents/chunks, reading chunks with their sources."""

import sqlite3

import pandas as pd

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,   -- = CSV `index`
    filename     TEXT,
    source_url   TEXT,
    download_url TEXT,
    source_title TEXT,
    source_type  TEXT,
    status       TEXT NOT NULL,      -- 'done' | 'failed'
    error        TEXT,
    n_chunks     INTEGER DEFAULT 0,
    ingested_at  TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    section     TEXT,
    page_start  INTEGER,
    page_end    INTEGER,
    text        TEXT NOT NULL,
    embed_text  TEXT NOT NULL,
    n_tokens    INTEGER,
    embedding   BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE VIEW IF NOT EXISTS chunk_citations AS
SELECT c.chunk_id, c.doc_id, c.chunk_index, c.section, c.page_start, c.page_end, c.text,
       d.filename, d.source_url, d.download_url, d.source_title, d.source_type
FROM chunks c JOIN documents d ON d.doc_id = c.doc_id;
"""

# SQL database metadata fields
METADATA_FIELDS = ("doc_id", "filename", "source_url", "download_url", "source_title", "source_type")


# ---------------- Connections ----------------
def init_db(path) -> sqlite3.Connection:
    """Create the database (tables, index, view) if needed and return a connection."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")   # needed for ON DELETE CASCADE
    conn.execute("PRAGMA journal_mode = WAL;")  # safer/faster writes
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def connect_db(path=DB_PATH) -> sqlite3.Connection:
    """Open the RAG database with rows accessible by column name."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ---------------- Writing ----------------
def store_document(conn, document, chunks, vectors):
    """Write the document row and all its chunks in a transaction."""
    with conn:
        # Overwrite previous instance
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (document["doc_id"],))
        conn.execute(
            f"""INSERT INTO documents ({", ".join(METADATA_FIELDS)}, status, n_chunks, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, 'done', ?, datetime('now'))""",
            (*[document[k] for k in METADATA_FIELDS], len(chunks)),
        )
        conn.executemany(
            """INSERT INTO chunks
               (doc_id, chunk_index, section, page_start, page_end,
                text, embed_text, n_tokens, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (document["doc_id"], chunk["chunk_index"], chunk["section"], chunk["page_start"],
                 chunk["page_end"], chunk["text"], chunk["extracted_text"], chunk["n_tokens"],
                 vector.tobytes())
                for chunk, vector in zip(chunks, vectors)
            ],
        )


def mark_failed(conn, doc, error):
    """Mark the document extraction as failed in the SQL database."""
    with conn:
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc["doc_id"],))
        conn.execute(
            f"""INSERT INTO documents ({", ".join(METADATA_FIELDS)}, status, error, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, 'failed', ?, datetime('now'))""",
            (*[doc[k] for k in METADATA_FIELDS], error),
        )


def already_done(conn: sqlite3.Connection, doc_id: str) -> bool:
    """True if the document was already ingested successfully."""
    row = conn.execute("SELECT status FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    return row is not None and row[0] == "done"


# ---------------- Reading ----------------
def get_citations(conn, chunk_ids: list[int]) -> dict[int, dict]:
    """Return {chunk_id: citation dict} for the chunks listed by the retrieval algorithm."""
    conn.row_factory = sqlite3.Row
    marks = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT * FROM chunk_citations WHERE chunk_id IN ({marks})", chunk_ids
    ).fetchall()
    conn.row_factory = None
    return {r["chunk_id"]: dict(r) for r in rows}


def fetch_chunk_citations(chunk_ids: list[int]) -> pd.DataFrame:
    """Return the chunks with their source metadata, in the order of chunk_ids."""
    if not chunk_ids:
        return pd.DataFrame()

    # SQLite commonly limits bound parameters; batch IDs defensively
    frames = []
    with connect_db() as conn:
        for start in range(0, len(chunk_ids), 900):
            batch = chunk_ids[start:start + 900]
            placeholders = ",".join("?" for _ in batch)
            sql_query = f"""
                SELECT chunk_id, doc_id, chunk_index, section,
                       page_start, page_end, text, filename,
                       source_url, download_url, source_title, source_type
                FROM chunk_citations
                WHERE chunk_id IN ({placeholders})
            """
            frames.append(pd.read_sql_query(sql_query, conn, params=batch))

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Re-arrange rows according to the given rank
    if not result.empty:
        order = {chunk_id: rank for rank, chunk_id in enumerate(chunk_ids)}
        result["_rank_order"] = result["chunk_id"].map(order)
        result = result.sort_values("_rank_order").drop(columns="_rank_order")
    return result


def get_db_signature() -> dict:
    """Fast structural check. Does not detect edits to existing text with same IDs."""
    with connect_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(MAX(chunk_id), 0) AS max_id FROM chunks"
        ).fetchone()
    return {"row_count": int(row["n"]), "max_chunk_id": int(row["max_id"])}
