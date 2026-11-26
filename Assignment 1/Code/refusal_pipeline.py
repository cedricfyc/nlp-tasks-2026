"""RAG pipeline with the refusal check before generation, without changing pipeline.py.

RefusalRAGPipeline inherits everything from RAGPipeline and only adds one step:
retrieve -> refusal check -> (refuse) or (generate with citations as before).

    python refusal_pipeline.py --refusal bm25 "What is a blocked account?"
"""

import argparse

from citations import REFUSAL_MESSAGE, CitedAnswer
from pipeline import RAGPipeline
from refusal import RefusalSystem


class RefusalRAGPipeline(RAGPipeline):
    """RAGPipeline + refusal: method is "bm25", "dense" or "llm_judge"."""

    def __init__(self, refusal_method="bm25", **kwargs):
        super().__init__(**kwargs)
        self.refusal = RefusalSystem(refusal_method, self)
        self.last_refusal = None
        self._cached = None             # (query, chunks) so retrieval runs only once

    def retrieve(self, user_query: str):
        if self._cached is not None and self._cached[0] == user_query:
            return self._cached[1]
        return super().retrieve(user_query)

    def answer_question(self, user_query: str, think=False):
        """Refuse before generation if the question looks unanswerable; otherwise as RAGPipeline."""
        context_df = self.retrieve(user_query)
        if not context_df.empty:
            self.last_refusal = self.refusal.check(user_query, context_df)
            if self.verbose:
                print(f"\n[refusal] {self.last_refusal.method}: "
                      f"P(unanswerable)={self.last_refusal.p_unanswerable:.2f} "
                      f"(cut-off {self.last_refusal.threshold:.2f})")
            if self.last_refusal.refuse:
                self.last_context_df = context_df
                self.last_result = CitedAnswer(answer=REFUSAL_MESSAGE, refused=True,
                                               refusal_reason=self.last_refusal.method)
                return "", self.last_result

        self._cached = (user_query, context_df)
        try:
            return super().answer_question(user_query, think=think)
        finally:
            self._cached = None


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