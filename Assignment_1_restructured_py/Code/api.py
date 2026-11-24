"""Demo UI and API.

    python api.py                        # http://localhost:8000
    python api.py --splade --reranker    # with SPLADE and the re-ranker

    GET  /       chat page (ui/index.html)
    POST /ask    {"question": "..."} -> {"answer", "retrieved", "seconds"}
"""

import argparse
import threading
import time
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

UI_DIR = Path(__file__).resolve().parent / "ui"


def context_to_sources(context_df) -> list[dict]:
    """Turn the retrieved chunks into a JSON-friendly list for the UI."""
    if context_df is None or context_df.empty:
        return []
    sources = []
    for row in context_df.itertuples():
        if pd.isna(row.page_start):
            pages = ""
        elif row.page_start == row.page_end:
            pages = f"{int(row.page_start)}"
        else:
            pages = f"{int(row.page_start)}-{int(row.page_end)}"
        sources.append({
            "chunk_id": int(row.chunk_id),
            "filename": row.filename,
            "section": row.section if isinstance(row.section, str) else None,
            "pages": pages,
            "url": row.download_url or row.source_url,
        })
    return sources


def create_app(pipeline) -> Flask:
    """Flask app: GET / serves the chat page, POST /ask answers a question."""
    from generation import clean_answer

    app = Flask(__name__, static_folder=str(UI_DIR), static_url_path="/ui")
    lock = threading.Lock()   # one GPU: answer one question at a time

    @app.get("/")
    def index():
        return send_from_directory(UI_DIR, "index.html")

    @app.post("/ask")
    def ask_endpoint():
        question = str((request.get_json(silent=True) or {}).get("question", "")).strip()
        if not question:
            return jsonify(error="Field 'question' is required."), 400

        started = time.time()
        with lock:
            _, answer = pipeline.conduct_rag(question)
            context_df = pipeline.last_context_df
        return jsonify(answer=clean_answer(answer),
                       retrieved=context_to_sources(context_df),
                       seconds=round(time.time() - started, 1))

    return app


def start_in_background(pipeline, port=8000):
    """Start the server in a background thread (for Colab) and show the link."""
    app = create_app(pipeline)
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=port, use_reloader=False),
                     daemon=True).start()
    print(f"UI running on port {port}")
    try:
        from google.colab import output
        output.serve_kernel_port_as_window(port)
    except ImportError:
        print(f"Open http://localhost:{port}")
    return app


if __name__ == "__main__":
    from pipeline import RAGPipeline

    parser = argparse.ArgumentParser()
    parser.add_argument("--splade", action="store_true", help="add SPLADE to the search")
    parser.add_argument("--reranker", action="store_true", help="re-rank with Qwen3-Reranker")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    rag = RAGPipeline(use_splade=args.splade, use_reranker=args.reranker, verbose=False)
    create_app(rag).run(host="0.0.0.0", port=args.port)
