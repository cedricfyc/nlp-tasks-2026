"""Run the 100 test questions through the real pipeline, then measure and judge the answers.

    python run_eval.py                                   # BM25 + dense, prompt refusal only
    python run_eval.py --retriever bm25                  # BM25 baseline
    python run_eval.py --retriever dense                 # dense only
    python run_eval.py --splade --reranker               # all searches + re-ranker
    python run_eval.py --refusal dense                   # calibrated refusal
    python run_eval.py --judge                           # also score the answers with the LLM judge
    python run_eval.py --metrics-only                    # recompute metrics from saved answers (no GPU)
    python run_eval.py --dataset ../data/test_60.json    # another question set (name added to the files)

Writes to results/ (or RAG_RESULTS_DIR / --out-dir):
    rag_<setting>.db        one row per question (answer, citations, refusal, chunks); resumable
    judge_<setting>.db      judge scores (with --judge), same table as the team's judge notebook
    metrics_<setting>.json  refusal rate, false-refusal rate, citation coverage (+ judge summary)
"""

import argparse
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import DATASET_PATH, REFUSAL_METHOD, REFUSAL_METHODS, RESULTS_DIR, RETRIEVER, TOP_K
from refusal import is_self_refusal

RETRIEVERS = ("both", "bm25", "dense")

ANSWERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_results (
    id               TEXT PRIMARY KEY,
    type             TEXT,
    domains          TEXT,
    primary_domain   TEXT,
    question         TEXT NOT NULL,
    ground_truth     TEXT,
    response         TEXT,
    fetched_context  TEXT,
    is_unanswerable  INTEGER,
    metadata         TEXT,
    created_at       TEXT,
    setting          TEXT,
    top_k            INTEGER,
    reranking_method TEXT,
    chunk_ids        TEXT,
    citations        TEXT,
    n_citations      INTEGER,
    refused          INTEGER,
    refusal_reason   TEXT,
    p_unanswerable   REAL
)
"""


def setting_name(retriever, splade, reranker, refusal) -> str:
    """e.g. 'both+splade+rerank_refusal-dense' (used in file names and the judge table)."""
    return retriever + ("+splade" if splade else "") + ("+rerank" if reranker else "") \
        + f"_refusal-{refusal}"


def generate_answers(rag, questions, answers_db, setting, reranking_method, limit=None):
    """Answer every question not yet in answers_db (resumable after a disconnect)."""
    with closing(sqlite3.connect(answers_db)) as conn:
        conn.execute(ANSWERS_SCHEMA)
        done = {r[0] for r in conn.execute("SELECT id FROM rag_results")}

    todo = [q for q in questions if q["id"] not in done][:limit]
    print(f"{len(todo)} questions to answer ({len(done)} already saved in {answers_db})")

    for position, item in enumerate(todo, start=1):
        try:
            _, result = rag.answer_question(item["question"])
        except KeyboardInterrupt:
            print("Interrupted; finished questions are saved, run again to resume.")
            break
        except Exception as exc:
            print(f"[{position}/{len(todo)}] {item['id']} FAILED: {exc}")
            continue

        context = rag.last_context_df
        decision = getattr(rag, "last_refusal", None)
        refused = bool(result.refused or is_self_refusal(result.answer))
        record = {
            "id": item["id"],
            "type": item.get("type"),
            "domains": json.dumps(item.get("domains", []), ensure_ascii=False),
            "primary_domain": item.get("primary_domain"),
            "question": item["question"],
            "ground_truth": item.get("ground_truth"),
            "response": result.answer,
            "fetched_context": json.dumps([] if context is None or context.empty
                                          else context["text"].tolist(), ensure_ascii=False),
            "is_unanswerable": int(bool(item.get("is_unanswerable"))),
            "metadata": json.dumps({k: v for k, v in item.items() if k not in {
                "id", "type", "domains", "primary_domain", "question", "ground_truth",
                "is_unanswerable"}}, ensure_ascii=False),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "setting": setting,
            "top_k": TOP_K,
            "reranking_method": reranking_method,
            "chunk_ids": json.dumps([] if context is None or context.empty
                                    else [int(c) for c in context["chunk_id"]]),
            "citations": json.dumps(result.citations, ensure_ascii=False, default=str),
            "n_citations": len(result.citations),
            "refused": int(refused),
            "refusal_reason": result.refusal_reason or ("model" if refused else None),
            "p_unanswerable": None if decision is None else float(decision.p_unanswerable),
        }
        with closing(sqlite3.connect(answers_db)) as conn, conn:
            conn.execute(f"INSERT OR REPLACE INTO rag_results ({', '.join(record)}) "
                         f"VALUES ({', '.join('?' * len(record))})", list(record.values()))
        print(f"[{position}/{len(todo)}] {item['id']} "
              f"{'REFUSED' if refused else f'{len(result.citations)} citation(s)'}")


def compute_metrics(answers: pd.DataFrame) -> dict:
    """Refusal rate on unanswerable questions, false refusals, citation coverage."""
    unanswerable = answers[answers["is_unanswerable"] == 1]
    answerable = answers[answers["is_unanswerable"] == 0]
    answered = answerable[answerable["refused"] == 0]

    def rate(part, whole):
        return {"count": int(part), "total": int(whole),
                "rate": round(part / whole, 3) if whole else None}

    metrics = {
        "n_questions": int(len(answers)),
        "refusal_rate": rate(unanswerable["refused"].sum(), len(unanswerable)),
        "false_refusal_rate": rate(answerable["refused"].sum(), len(answerable)),
        "citation_coverage": rate((answered["n_citations"] > 0).sum(), len(answered)),
        "mean_citations_per_answer": round(float(answered["n_citations"].mean()), 2)
        if len(answered) else None,
        "refused_by_type": answers.groupby("type")["refused"].mean().round(3).to_dict(),
        "refusal_reasons": answers.loc[answers["refused"] == 1, "refusal_reason"]
        .value_counts().to_dict(),
    }

    r, f, c = metrics["refusal_rate"], metrics["false_refusal_rate"], metrics["citation_coverage"]
    print(f"\nQuestions answered:            {metrics['n_questions']}")
    print(f"Refusal rate (unanswerable):   {r['count']}/{r['total']}"
          + (f" = {r['rate']:.0%}" if r["rate"] is not None else ""))
    print(f"False refusals (answerable):   {f['count']}/{f['total']}"
          + (f" = {f['rate']:.0%}" if f["rate"] is not None else ""))
    print(f"Answers with >= 1 citation:    {c['count']}/{c['total']}"
          + (f" = {c['rate']:.0%}" if c["rate"] is not None else ""))
    print("Refused by question type:      "
          + ", ".join(f"{k}={v:.0%}" for k, v in metrics["refused_by_type"].items()))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", choices=RETRIEVERS, default=RETRIEVER)
    parser.add_argument("--splade", action="store_true")
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--refusal", choices=REFUSAL_METHODS, default=REFUSAL_METHOD)
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--limit", type=int, help="only the first N open questions (quick test)")
    parser.add_argument("--judge", action="store_true", help="also score answers with the LLM judge")
    parser.add_argument("--metrics-only", action="store_true",
                        help="only recompute metrics from saved answers (no models loaded)")
    args = parser.parse_args()

    setting = setting_name(args.retriever, args.splade, args.reranker, args.refusal)
    if Path(args.dataset).resolve() != Path(DATASET_PATH).resolve():
        setting += f"_{Path(args.dataset).stem}"          # keep results of each question set apart
    reranking_method = "RRF+Qwen3-Reranker" if args.reranker else "RRF"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_db = out_dir / f"rag_{setting}.db"
    judge_db = out_dir / f"judge_{setting}.db"
    print(f"Setting: {setting}\nResults: {out_dir}")

    rag = None
    if not args.metrics_only:
        from pipeline import RAGPipeline

        questions = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
        rag = RAGPipeline(use_splade=args.splade, use_reranker=args.reranker, verbose=False,
                          refusal_method=args.refusal, retriever=args.retriever)
        generate_answers(rag, questions, answers_db, setting, reranking_method, args.limit)

    from evaluate import load_answers, load_judged, summarize

    answers = load_answers(answers_db)
    metrics = {"setting": setting, **compute_metrics(answers)}

    if args.judge and not args.metrics_only:
        from evaluate import Judge, judge_answers
        judge_answers(answers, judge_db, Judge(rag.tokenizer, rag.model), setting, TOP_K,
                      reranking_method)
    if judge_db.exists():
        metrics["judge"] = summarize(load_judged(answers_db, judge_db))

    metrics_path = out_dir / f"metrics_{setting}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {metrics_path}")


if __name__ == "__main__":
    main()
