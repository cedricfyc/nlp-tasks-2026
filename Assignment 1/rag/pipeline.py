"""RAG pipeline: BM25 + dense (+ SPLADE) -> RRF (-> re-ranker) -> refusal check -> Qwen3.5-4B
-> hard citations.

    rag = RAGPipeline()                                    # BM25 + dense, prompt refusal only
    rag = RAGPipeline(retriever="bm25")                    # BM25 baseline
    rag = RAGPipeline(use_splade=True, use_reranker=True)  # all searches + re-ranker
    rag = RAGPipeline(refusal_method="dense")              # calibrated refusal before generation
    rag = RAGPipeline(refusal_method="combined")           # majority vote of bm25, dense, llm_judge
"""

from citations import (
    REFUSAL_MESSAGE, CitedAnswer, build_cited_context, check_citations, citation_feedback,
    finalise_citations, format_references, no_context_answer,
)
from config import (
    BM25_CACHE, CANDIDATE_K, CITATION_MAX_RETRIES, CITATION_SNIPPET_CHARS, CITATION_STRICT,
    REFUSAL_METHOD, REFUSAL_METHODS, RETRIEVER, SPLADE_INDEX_PATH, TOP_K,
)
from data import fetch_chunk_citations
from generation import build_messages, build_system_prompt, clean_answer, generate, load_model
from retrieval import (
    Qwen3Reranker, bm25_search, compare_search_algorithms, dense_search, load_bm25,
    load_embedding_matrix, load_embedding_model, load_splade_index, load_splade_model,
    reciprocal_rank_fusion, rerank, splade_search,
)

RETRIEVERS = ("both", "bm25", "dense")


class MajorityVoteRefusal:
    """refusal_method="combined": refuse when at least 2 of bm25, dense, llm_judge say refuse."""

    method = "combined"

    def __init__(self, pipeline):
        from refusal import RefusalSystem
        self.systems = [RefusalSystem(name, pipeline) for name in ("bm25", "dense", "llm_judge")]

    def check(self, question, context_df):
        from refusal import RefusalDecision
        decisions = [system.check(question, context_df) for system in self.systems]
        votes = sum(d.refuse for d in decisions)
        return RefusalDecision(refuse=votes >= 2, method=self.method,
                               p_unanswerable=votes / 3, threshold=2 / 3,
                               features={d.method: {"refuse": bool(d.refuse),
                                                    "p_unanswerable": d.p_unanswerable}
                                         for d in decisions})


class RAGPipeline:
    """Loads every model and index once; answer_question() answers one question."""

    def __init__(self, bm25_cache_path=BM25_CACHE, use_splade=False, use_reranker=False,
                 verbose=True, citation_retries=CITATION_MAX_RETRIES, strict_citations=CITATION_STRICT,
                 refusal_method=REFUSAL_METHOD, retriever=RETRIEVER):
        refusal_method = "none" if refusal_method is None else refusal_method
        if refusal_method not in REFUSAL_METHODS:
            raise ValueError(f"refusal_method must be one of {REFUSAL_METHODS}")
        if retriever not in RETRIEVERS:
            raise ValueError(f"retriever must be one of {RETRIEVERS}")
        self.retriever = retriever
        self.verbose = verbose

        # Generator (Qwen3.5-4B)
        self.tokenizer, self.model = load_model()

        # Retrieval resources
        self.bm25_payload = load_bm25(cache_path=bm25_cache_path, validate=True)
        self.vector_chunk_ids, self.vector_matrix = load_embedding_matrix()
        self.embedding_model = load_embedding_model()

        # Optional: SPLADE (only if its index exists) and re-ranker
        self.splade = None
        if use_splade:
            if SPLADE_INDEX_PATH.exists():
                splade_chunk_ids, splade_matrix = load_splade_index()
                self.splade = (load_splade_model(), splade_matrix, splade_chunk_ids)
            else:
                print(f"WARNING: SPLADE index not found ({SPLADE_INDEX_PATH}); SPLADE is switched off. "
                      "Build it once with: python build_database.py --splade")
        self.reranker = Qwen3Reranker() if use_reranker else None

        # Hard citations
        self.citation_retries = citation_retries
        self.strict_citations = strict_citations

        # Refusal before generation (loaded last: it uses the model and indexes above)
        self.refusal_method = refusal_method
        self.refusal = None
        if refusal_method == "combined":
            self.refusal = MajorityVoteRefusal(self)
        elif refusal_method != "none":
            from refusal import RefusalSystem
            self.refusal = RefusalSystem(refusal_method, self)

        self.last_context_df = None
        self.last_result: CitedAnswer | None = None
        self.last_refusal = None

    def retrieve(self, user_query: str, top_k=None):
        """Return the top_k context chunks (DataFrame) for a question.

        Each search returns CANDIDATE_K candidates; RRF merges them; the best CANDIDATE_K
        go to the re-ranker, which keeps top_k (without re-ranker: the RRF top_k).
        """
        top_k = TOP_K if top_k is None else int(top_k)
        if self.retriever == "both":
            lexical, semantic = compare_search_algorithms(
                query=user_query,
                bm25_payload=self.bm25_payload,
                vector_matrix=self.vector_matrix,
                vector_chunk_ids=self.vector_chunk_ids,
                top_k=CANDIDATE_K,
                embedding_model=self.embedding_model,
                show=self.verbose,
            )
            ranked_lists = [lexical["chunk_id"].tolist(), semantic["chunk_id"].tolist()]
        elif self.retriever == "bm25":
            lexical = bm25_search(user_query, self.bm25_payload, top_k=CANDIDATE_K)
            ranked_lists = [lexical["chunk_id"].tolist() if not lexical.empty else []]
        else:  # dense
            semantic = dense_search(self.vector_matrix, self.vector_chunk_ids, user_query,
                                    top_k=CANDIDATE_K, model=self.embedding_model)
            ranked_lists = [semantic["chunk_id"].tolist() if not semantic.empty else []]

        if self.splade is not None:
            model, matrix, chunk_ids = self.splade
            ranked_lists.append(splade_search(model, matrix, chunk_ids, user_query,
                                              top_k=CANDIDATE_K)["chunk_id"].tolist())

        fused = reciprocal_rank_fusion(ranked_lists)
        if self.reranker is None:
            return fetch_chunk_citations(fused[:top_k]).reset_index(drop=True)
        context_df = fetch_chunk_citations(fused[:CANDIDATE_K])
        return rerank(self.reranker, user_query, context_df, top_n=top_k).reset_index(drop=True)

    @staticmethod
    def build_context(context_df) -> str:
        """Join the chunks as numbered sources [1], [2], ... (the numbers the answer cites)."""
        return build_cited_context(context_df)

    def check_refusal(self, user_query: str, context_df):
        """Calibrated refusal check (None when refusal_method is "none")."""
        if self.refusal is None or context_df.empty:
            return None
        decision = self.refusal.check(user_query, context_df)
        if self.verbose:
            print(f"\n[refusal] {decision.method}: P(unanswerable)={decision.p_unanswerable:.2f} "
                  f"(cut-off {decision.threshold:.2f}) -> {'refuse' if decision.refuse else 'answer'}")
        return decision

    def answer_question(self, user_query: str, think=False, top_k=None):
        """Retrieve, check refusal, generate and enforce citations. Returns (thinking, CitedAnswer).

        top_k: passages given to the generator (default config.TOP_K; the UI sends 3-5).

        1. The chunks are retrieved once.
        2. Refusal check (if a refusal method is set): refuse before generating.
        3. The chunks are given to the generator as numbered sources.
        4. The answer is checked: invalid source numbers are removed and factual
           sentences without a citation are listed.
        5. If the check fails, the generator gets one correction message and rewrites.
        6. Uncited sentences still left are removed (strict mode); an answer with no
           valid citation becomes the refusal message.
        """
        context_df = (self.retrieve(user_query) if top_k is None
                      else self.retrieve(user_query, top_k=top_k))
        self.last_context_df = context_df
        self.last_refusal = None
        if context_df.empty:
            self.last_result = no_context_answer()
            return "", self.last_result

        self.last_refusal = self.check_refusal(user_query, context_df)
        if self.last_refusal is not None and self.last_refusal.refuse:
            self.last_result = CitedAnswer(answer=REFUSAL_MESSAGE, refused=True,
                                           refusal_reason=self.last_refusal.method)
            if self.verbose:
                print("\n--- Answer ---\n" + REFUSAL_MESSAGE)
            return "", self.last_result

        messages = build_messages(build_system_prompt(self.build_context(context_df)), user_query)
        thinking, raw = generate(tokenizer=self.tokenizer, model=self.model,
                                 messages=messages, think=think)
        check = check_citations(clean_answer(raw), len(context_df))

        retries = 0
        while check.needs_retry and retries < self.citation_retries:
            retries += 1
            if self.verbose:
                print(f"\n[citations] check failed, retry {retries}/{self.citation_retries}")
            messages += [{"role": "assistant", "content": clean_answer(raw)},
                         {"role": "user", "content": citation_feedback(check)}]
            thinking, raw = generate(tokenizer=self.tokenizer, model=self.model,
                                     messages=messages, think=think)
            check = check_citations(clean_answer(raw), len(context_df))

        result = finalise_citations(check, context_df, strict=self.strict_citations,
                                    snippet_chars=CITATION_SNIPPET_CHARS)
        result.retries = retries
        self.last_result = result

        if self.verbose:
            print("\n--- Cited answer ---\n" + result.answer + "\n\n--- Sources ---\n"
                  + format_references(result))
        return thinking, result

    def conduct_rag(self, user_query: str, think=False, top_k=None):
        """Same as before: returns (thinking, answer). The answer now carries [n] citations.

        The full result (citations, refusal flag, dropped sentences) is in
        self.last_result; the chunks used as context are in self.last_context_df;
        the refusal decision (if any) is in self.last_refusal.
        """
        thinking, result = self.answer_question(user_query, think=think, top_k=top_k)
        return thinking, result.answer
