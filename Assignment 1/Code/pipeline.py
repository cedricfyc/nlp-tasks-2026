"""RAG pipeline: BM25 + dense (+ SPLADE) -> RRF (-> re-ranker) -> Qwen3.5-4B -> hard citations."""

from citations import (
    CitedAnswer, build_cited_context, check_citations, citation_feedback, finalise_citations,
    format_references, no_context_answer,
)
from config import (
    BM25_CACHE, CITATION_MAX_RETRIES, CITATION_SNIPPET_CHARS, CITATION_STRICT, TOP_K,
)
from data import fetch_chunk_citations
from generation import build_messages, build_system_prompt, clean_answer, generate, load_model
from retrieval import (
    Qwen3Reranker, compare_search_algorithms, load_bm25, load_embedding_matrix,
    load_embedding_model, load_splade_index, load_splade_model, reciprocal_rank_fusion,
    rerank, splade_search,
)


class RAGPipeline:
    """Loads every model and index once; answer_question() answers one question."""

    def __init__(self, bm25_cache_path=BM25_CACHE, use_splade=False, use_reranker=False,
                 verbose=True, citation_retries=CITATION_MAX_RETRIES, strict_citations=CITATION_STRICT):
        # Generator (Qwen3.5-4B)
        self.tokenizer, self.model = load_model()

        # Retrieval resources
        self.bm25_payload = load_bm25(cache_path=bm25_cache_path, validate=True)
        self.vector_chunk_ids, self.vector_matrix = load_embedding_matrix()
        self.embedding_model = load_embedding_model()

        # Optional: SPLADE and re-ranker
        self.splade = None
        if use_splade:
            splade_chunk_ids, splade_matrix = load_splade_index()
            self.splade = (load_splade_model(), splade_matrix, splade_chunk_ids)
        self.reranker = Qwen3Reranker() if use_reranker else None

        # Hard citations
        self.citation_retries = citation_retries
        self.strict_citations = strict_citations

        self.verbose = verbose
        self.last_context_df = None
        self.last_result: CitedAnswer | None = None

    def retrieve(self, user_query: str):
        """Return the context chunks (DataFrame) for a question."""
        lexical, semantic = compare_search_algorithms(
            query=user_query,
            bm25_payload=self.bm25_payload,
            vector_matrix=self.vector_matrix,
            vector_chunk_ids=self.vector_chunk_ids,
            top_k=TOP_K,
            embedding_model=self.embedding_model,
            show=self.verbose,
        )
        ranked_lists = [lexical["chunk_id"].tolist(), semantic["chunk_id"].tolist()]

        if self.splade is not None:
            model, matrix, chunk_ids = self.splade
            ranked_lists.append(splade_search(model, matrix, chunk_ids, user_query)["chunk_id"].tolist())

        context_df = fetch_chunk_citations(reciprocal_rank_fusion(ranked_lists))
        if self.reranker is not None:
            context_df = rerank(self.reranker, user_query, context_df)
        return context_df.reset_index(drop=True)

    @staticmethod
    def build_context(context_df) -> str:
        """Join the chunks as numbered sources [1], [2], ... (the numbers the answer cites)."""
        return build_cited_context(context_df)

    def answer_question(self, user_query: str, think=False):
        """Retrieve, generate and enforce citations. Returns (thinking, CitedAnswer).

        1. The chunks are given to the generator as numbered sources.
        2. The answer is checked: invalid source numbers are removed and factual
           sentences without a citation are listed.
        3. If the check fails, the generator gets one correction message and rewrites.
        4. Uncited sentences still left are removed (strict mode); an answer with no
           valid citation becomes the refusal message.
        """
        context_df = self.retrieve(user_query)
        self.last_context_df = context_df
        if context_df.empty:
            self.last_result = no_context_answer()
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

    def conduct_rag(self, user_query: str, think=False):
        """Same as before: returns (thinking, answer). The answer now carries [n] citations.

        The full result (citations, refusal flag, dropped sentences) is in
        self.last_result; the chunks used as context are in self.last_context_df.
        """
        thinking, result = self.answer_question(user_query, think=think)
        return thinking, result.answer
