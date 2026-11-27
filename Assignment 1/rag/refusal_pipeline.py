"""Kept for compatibility: the refusal check now lives in pipeline.RAGPipeline.

    RefusalRAGPipeline(refusal_method="bm25", ...)  ==  RAGPipeline(refusal_method="bm25", ...)

    python refusal_pipeline.py --refusal bm25 "What is a blocked account?"
    (same as: python run_rag.py --refusal bm25 "What is a blocked account?")
"""

import argparse

from pipeline import RAGPipeline


class RefusalRAGPipeline(RAGPipeline):
    """RAGPipeline with a refusal method switched on (default "bm25")."""

    def __init__(self, refusal_method="bm25", **kwargs):
        super().__init__(refusal_method=refusal_method, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", nargs="+")
    parser.add_argument("--refusal", choices=["bm25", "dense", "llm_judge"], default="bm25")
    parser.add_argument("--splade", action="store_true")
    parser.add_argument("--reranker", action="store_true")
    args = parser.parse_args()

    rag = RefusalRAGPipeline(refusal_method=args.refusal, use_splade=args.splade,
                             use_reranker=args.reranker)
    for question in args.questions:
        print("\n" + "=" * 80 + f"\nQ: {question}")
        rag.answer_question(question)
