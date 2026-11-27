"""Build the database and search indexes.

    python build_database.py                       # PDFs -> chunks -> embeddings -> SQLite
    python build_database.py --start 51 --stop 63 --db part3.db   # only rows 51-63
    python build_database.py --merge part1.db part2.db part3.db   # fix chunk IDs + merge
    python build_database.py --bm25                # rebuild the BM25 index
    python build_database.py --splade              # build the SPLADE index (skipped if it exists)
    python build_database.py --splade --force      # rebuild the SPLADE index
"""

import argparse
import csv
import os
import sqlite3
import time
import traceback

import numpy as np

from config import (
    CSV_PATH, DB_PATH, DOCS_PATH, EMBED_BATCH_SIZE, EMBED_MODEL, MAX_CHUNK_TOKENS,
)
from data import already_done, init_db, mark_failed, store_document


# =====================================================================
# Extraction (Docling)
# =====================================================================
def build_chunker(tokenizer_name: str):
    """HybridChunker that counts tokens with the SAME tokenizer as the embedder."""
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_name),
        max_tokens=MAX_CHUNK_TOKENS,
    )
    return HybridChunker(tokenizer=tokenizer, merge_peers=True)


def page_range(chunk) -> tuple[int | None, int | None]:
    """Collect page numbers of every item in the chunk."""
    pages = [prov.page_no
             for item in chunk.meta.doc_items
             for prov in (getattr(item, "prov", []) or [])]
    return (min(pages), max(pages)) if pages else (None, None)


def extract_chunks(converter, chunker, filepath: str) -> list[dict]:
    """Convert one document and return a list of chunk dicts."""
    converted_doc = converter.convert(filepath).document
    chunks = []
    for i, chunk in enumerate(chunker.chunk(dl_doc=converted_doc)):
        start, end = page_range(chunk)
        headings = chunk.meta.headings or []
        chunks.append({
            "chunk_index": i,
            "section": " > ".join(headings) if headings else None,
            "page_start": start,
            "page_end": end,
            "text": chunk.text,
            # contextualize() prepends the headings to the text -> better embeddings
            "extracted_text": chunker.contextualize(chunk=chunk),
        })
    return chunks


# =====================================================================
# Embedding (Qwen3-Embedding)
# =====================================================================
def load_embedder():
    """Load the embedding model in float16 (GPU if available)."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL, device=device,
                                model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = MAX_CHUNK_TOKENS + 64    # headroom for the heading prefix
    return model


def embed_texts(model, texts: list[str]) -> np.ndarray:
    """Embed without the query instruction prefix. Returns L2-normalised float32 (n, 1024)."""
    return model.encode(
        texts, batch_size=EMBED_BATCH_SIZE, normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)


# =====================================================================
# Build
# =====================================================================
def build(db_path=DB_PATH, start=1, stop=None):
    """Extract, chunk, embed and store every document listed in the metadata CSV."""
    from docling.document_converter import DocumentConverter

    conn = init_db(db_path)
    converter = DocumentConverter()
    chunker = build_chunker(EMBED_MODEL)
    embedder = load_embedder()

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    stop = stop or len(rows)

    # Index extracted files once
    file_lookup = {name: os.path.join(root, name)
                   for root, _, files in os.walk(DOCS_PATH) for name in files}

    for n, row in enumerate(rows, 1):
        if not start <= n <= stop:
            continue

        # Map CSV columns to document fields: 'index' becomes doc_id
        doc = {
            "doc_id": row["index"].strip(),
            "filename": row["filename"].strip(),
            "source_url": row["source_url"].strip(),
            "download_url": row["download_url"].strip(),
            "source_title": row["source_title"].strip(),
            "source_type": row["source_type"].strip(),
        }
        if already_done(conn, doc["doc_id"]):
            print(f"[{n}/{len(rows)}] skip {doc['doc_id']} (already ingested)")
            continue

        file_path = file_lookup.get(doc["filename"] + ".pdf")   # all files are PDF
        if file_path is None:
            print(f"index: {n}, file not found: {doc['filename']}.pdf")

        t0 = time.time()
        try:
            chunks = extract_chunks(converter, chunker, file_path)
            if not chunks:
                raise ValueError("Docling returned no chunks (empty or unreadable file)")
            for chunk in chunks:
                chunk["n_tokens"] = chunker.tokenizer.count_tokens(chunk["extracted_text"])

            vectors = embed_texts(embedder, [chunk["extracted_text"] for chunk in chunks])
            store_document(conn, doc, chunks, vectors)
            print(f"[{n}/{len(rows)}] {doc['doc_id']} '{doc['filename']}': "
                  f"{len(chunks)} chunks in {time.time() - t0:.1f}s")
        except Exception as e:
            traceback.print_exc()
            mark_failed(conn, doc, str(e))
            print(f"[{n}/{len(rows)}] {doc['doc_id']}: FAILED ({e})")

    conn.close()


# =====================================================================
# Merging partial databases
# =====================================================================
def fix_chunk_ids(databases: list[str]):
    """Renumber chunk IDs so they are sequential across all partial databases."""
    next_id = 1
    for db in databases:
        print(f"\nProcessing {db}")
        if not os.path.exists(db):
            raise FileNotFoundError(db)

        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            old_ids = [row[0] for row in cursor.execute(
                "SELECT chunk_id FROM chunks ORDER BY chunk_id").fetchall()]
            if not old_ids:
                print("No chunks found.")
                conn.commit()
                continue

            # First move all IDs to unique negative values (prevents primary key collisions)
            for index, old_id in enumerate(old_ids):
                cursor.execute("UPDATE chunks SET chunk_id = ? WHERE chunk_id = ?", (-(index + 1), old_id))
            # Then assign globally sequential IDs
            for index in range(len(old_ids)):
                cursor.execute("UPDATE chunks SET chunk_id = ? WHERE chunk_id = ?",
                               (next_id + index, -(index + 1)))
            conn.commit()

            minimum, maximum, count = cursor.execute(
                "SELECT MIN(chunk_id), MAX(chunk_id), COUNT(*) FROM chunks").fetchone()
            print(f"Total chunks: {count}, IDs {minimum}-{maximum}")
            next_id = maximum + 1
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    print(f"\nAll databases processed. Sequential chunk IDs: 1 to {next_id - 1}")


def merge_databases(databases: list[str], output_db=DB_PATH):
    """Concatenate partial databases into one, with the index and citation view."""
    for db in databases:
        if not os.path.exists(db):
            raise FileNotFoundError(f"Database not found: {db}")
    if os.path.exists(output_db):
        os.remove(output_db)

    conn = sqlite3.connect(output_db)
    cursor = conn.cursor()
    try:
        for i, db in enumerate(databases):
            cursor.execute(f"ATTACH DATABASE ? AS source{i}", (os.path.abspath(db),))

        for i, db in enumerate(databases):
            tables = cursor.execute(f"""
                SELECT name, sql FROM source{i}.sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            """).fetchall()
            for table_name, create_sql in tables:
                exists = cursor.execute(
                    "SELECT name FROM main.sqlite_master WHERE type = 'table' AND name = ?",
                    (table_name,)).fetchone()
                if not exists:
                    cursor.execute(create_sql)
                cursor.execute(f'INSERT INTO main."{table_name}" SELECT * FROM source{i}."{table_name}"')
                print(f"Copied {table_name} from {db}")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)")
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS chunk_citations AS
            SELECT c.chunk_id, c.doc_id, c.chunk_index, c.section, c.page_start, c.page_end,
                   c.text, d.filename, d.source_url, d.download_url, d.source_title, d.source_type
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
        """)
        conn.commit()
        print(f"\nSuccessfully created {output_db}")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=str(DB_PATH), help="database to write")
    parser.add_argument("--start", type=int, default=1, help="first CSV row to process")
    parser.add_argument("--stop", type=int, default=None, help="last CSV row to process")
    parser.add_argument("--merge", nargs="+", metavar="PART_DB", help="fix IDs and merge partial databases into --db")
    parser.add_argument("--bm25", action="store_true", help="rebuild the BM25 index")
    parser.add_argument("--splade", action="store_true", help="build the SPLADE index")
    parser.add_argument("--force", action="store_true", help="rebuild the SPLADE index even if it exists")
    args = parser.parse_args()

    if args.merge:
        fix_chunk_ids(args.merge)
        merge_databases(args.merge, args.db)
    elif args.bm25 or args.splade:
        from retrieval import build_and_save_bm25, build_splade_index, load_splade_model
        if args.bm25:
            build_and_save_bm25()
        if args.splade:
            from config import SPLADE_INDEX_PATH
            if SPLADE_INDEX_PATH.exists() and not args.force:
                print(f"SPLADE index already exists ({SPLADE_INDEX_PATH}); skipped. Use --force to rebuild.")
            else:
                build_splade_index(load_splade_model())
    else:
        build(args.db, args.start, args.stop)
