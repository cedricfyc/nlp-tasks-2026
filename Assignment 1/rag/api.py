"""Demo UI and API.

    python api.py                        # http://localhost:8000
    python api.py --splade --reranker    # with SPLADE and the re-ranker
    python api.py --refusal dense        # calibrated refusal before generation
    python api.py --retriever bm25       # BM25 only (baseline)

    GET  /       chat page (ui/index.html)
    POST /ask    {"question": "...", "top_k": 3}
                 -> {"answer", "citations", "refused", "retrieved", "citation_check",
                     "refusal_method", "p_unanswerable", "refusal", "top_k", "candidates", "seconds"}
                 top_k (optional, 3/4/5, default 3): passages kept from the candidates.

    "answer" contains [n] markers; citations[n-1] is the source of [n]
    (chunk_id, doc_id, source_title, filename, section, pages, url, snippet).
"""

import argparse
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from citations import format_pages
from config import CANDIDATE_K, TOP_K, TOP_K_CHOICES

UI_DIR = Path(__file__).resolve().parent / "ui"


def context_to_sources(context_df) -> list[dict]:
    """Turn the retrieved chunks into a JSON-friendly list for the UI."""
    if context_df is None or context_df.empty:
        return []
    sources = []
    for row in context_df.itertuples():
        sources.append({
            "chunk_id": int(row.chunk_id),
            "filename": row.filename,
            "section": row.section if isinstance(row.section, str) else None,
            "pages": format_pages(row.page_start, row.page_end),
            "url": row.download_url or row.source_url,
            "score": round(float(row.rerank_score), 3) if hasattr(row, "rerank_score") else None,
        })
    return sources


def create_app(pipeline) -> Flask:
    """Flask app: GET / serves the chat page, POST /ask answers a question."""
    app = Flask(__name__, static_folder=str(UI_DIR), static_url_path="/ui")
    lock = threading.Lock()   # one GPU: answer one question at a time

    @app.get("/")
    def index():
        return send_from_directory(UI_DIR, "index.html")

    @app.post("/ask")
    def ask_endpoint():
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        if not question:
            return jsonify(error="Field 'question' is required."), 400
        try:
            top_k = int(payload.get("top_k", TOP_K))
        except (TypeError, ValueError):
            top_k = None
        if top_k not in TOP_K_CHOICES:
            return jsonify(error=f"Field 'top_k' must be one of {list(TOP_K_CHOICES)}."), 400

        started = time.time()
        with lock:
            _, result = pipeline.answer_question(question, top_k=top_k)
            context_df = pipeline.last_context_df
            decision = getattr(pipeline, "last_refusal", None)
        return jsonify(answer=result.answer,
                       citations=result.citations,
                       refused=result.refused,
                       retrieved=context_to_sources(context_df),
                       citation_check={
                           "refusal_reason": result.refusal_reason,
                           "invalid_ids": result.invalid_ids,
                           "dropped_sentences": result.dropped_sentences,
                           "retries": result.retries,
                       },
                       refusal_method=getattr(pipeline, "refusal_method", "none"),
                       p_unanswerable=None if decision is None else round(decision.p_unanswerable, 4),
                       refusal=None if decision is None else {
                           "method": decision.method,
                           "refuse": bool(decision.refuse),
                           "p_unanswerable": round(decision.p_unanswerable, 4),
                           "threshold": round(decision.threshold, 4),
                       },
                       top_k=top_k,
                       candidates=CANDIDATE_K,
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
    from config import REFUSAL_METHOD, REFUSAL_METHODS, RETRIEVER
    from pipeline import RETRIEVERS, RAGPipeline

    parser = argparse.ArgumentParser()
    parser.add_argument("--splade", action="store_true", help="add SPLADE to the search")
    parser.add_argument("--reranker", action="store_true", help="re-rank with Qwen3-Reranker")
    parser.add_argument("--refusal", choices=REFUSAL_METHODS, default=REFUSAL_METHOD,
                        help="refusal check before generation (none = prompt refusal only)")
    parser.add_argument("--retriever", choices=RETRIEVERS, default=RETRIEVER,
                        help="both = BM25 + dense; bm25 / dense for the baseline comparison")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    rag = RAGPipeline(use_splade=args.splade, use_reranker=args.reranker, verbose=False,
                      refusal_method=args.refusal, retriever=args.retriever)
    create_app(rag).run(host="0.0.0.0", port=args.port)
