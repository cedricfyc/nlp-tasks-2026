"""Find the refusal cut-off from labelled questions (is_unanswerable) instead of guessing it.

    python calibrate_refusal.py --method bm25          # CPU only, seconds
    python calibrate_refusal.py --method dense         # loads Qwen3-Embedding-0.6B
    python calibrate_refusal.py --method llm_judge     # loads Qwen3.5-4B (GPU)
    python calibrate_refusal.py --method bm25 --max-false-refusal 0.05

Writes:
    data/refusal_<method>.json            model + cut-off, loaded by refusal.RefusalSystem
    data/refusal_<method>_questions.csv   per-question features and P(unanswerable)

The scores printed are cross-validated (each question is scored by a model that did
not see it). Note: calibrate on one question set and report final results on another
(e.g. Cedric's 150 evaluation questions), otherwise the numbers are too optimistic.
"""

import argparse
import json

import pandas as pd

from config import DATA_DIR, DATASET_PATH
from refusal import (
    BM25_FEATURES, DENSE_FEATURES, LLM_JUDGE_FEATURES, LLMJudge, ScoreRefuser,
    bm25_features, dense_features,
)


def feature_function(method):
    """Load only what the method needs; return (feature_names, f(question) -> dict)."""
    if method == "bm25":
        from retrieval import load_bm25
        payload = load_bm25()
        return BM25_FEATURES, lambda q: bm25_features(q, payload)

    if method == "dense":
        from retrieval import load_embedding_matrix, load_embedding_model
        _, matrix = load_embedding_matrix()
        model = load_embedding_model()
        return DENSE_FEATURES, lambda q: dense_features(q, matrix, model)

    if method == "llm_judge":
        from pipeline import RAGPipeline
        rag = RAGPipeline(verbose=False)
        judge = LLMJudge(rag.tokenizer, rag.model)
        return LLM_JUDGE_FEATURES, lambda q: judge.features(q, rag.retrieve(q))

    raise ValueError(method)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["bm25", "dense", "llm_judge"], default="bm25")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--max-false-refusal", type=float, default=0.10,
                        help="highest share of answerable questions that may be refused")
    args = parser.parse_args()

    questions = json.loads(open(args.dataset, encoding="utf-8").read())
    names, features_of = feature_function(args.method)

    rows = []
    for i, item in enumerate(questions, start=1):
        rows.append({"id": item["id"], "type": item["type"],
                     "is_unanswerable": int(item["is_unanswerable"]),
                     "question": item["question"], **features_of(item["question"])})
        if i % 20 == 0:
            print(f"features: {i}/{len(questions)}")
    df = pd.DataFrame(rows)

    refuser, oof = ScoreRefuser.fit(args.method, names, df[names].values,
                                    df["is_unanswerable"].values, args.max_false_refusal)
    refuser.save()

    df["p_unanswerable_cv"] = oof
    df["refused_cv"] = oof >= refuser.threshold
    out = DATA_DIR / f"refusal_{args.method}_questions.csv"
    df.to_csv(out, index=False)

    m = refuser.metrics
    print(f"\n=== {args.method} refusal (5-fold cross-validated) ===")
    print(f"questions: {m['n_questions']} ({m['n_unanswerable']} unanswerable)")
    print(f"ROC-AUC:                     {m['cv_roc_auc']:.3f}   (0.5 = chance)")
    print(f"cut-off P(unanswerable) >=   {refuser.threshold:.3f}")
    print(f"unanswerable refused:        {m['refusal_recall']:.0%}")
    print(f"answerable wrongly refused:  {m['false_refusal_rate']:.0%}  "
          f"(limit {m['max_false_refusal']:.0%})")
    print(f"balanced accuracy:           {m['balanced_accuracy']:.3f}")
    print("\nFeature weights (positive = pushes towards refusing):")
    for name, w in sorted(zip(names, refuser.coef), key=lambda x: -abs(x[1])):
        print(f"  {name:15s} {w:+.2f}")
    print("\nRefusal rate by question type:")
    print(df.groupby("type")["refused_cv"].mean().map("{:.0%}".format).to_string())
    print(f"\nPer-question details: {out}")


if __name__ == "__main__":
    main()
