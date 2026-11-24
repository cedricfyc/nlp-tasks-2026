"""RAG pipeline: BM25 + dense (+ SPLADE) -> RRF (-> re-ranker) -> Qwen3.5-4B."""

from config import BM25_CACHE, TOP_K
from data import fetch_chunk_citations
from generation import ask, build_system_prompt, load_model
from retrieval import (
    Qwen3Reranker, compare_search_algorithms, load_bm25, load_embedding_matrix,
    load_embedding_model, load_splade_index, load_splade_model, reciprocal_rank_fusion,
    rerank, splade_search,
)


class RAGPipeline:
    """Loads every model and index once; conduct_rag() answers one question."""

    def __init__(self, bm25_cache_path=BM25_CACHE, use_splade=False, use_reranker=False,
                 verbose=True):
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

        self.verbose = verbose
        self.last_context_df = None

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
        return context_df

    @staticmethod
    def build_context(context_df) -> str:
        """Join the chunks as [CONTEXT n] blocks."""
        return "\n\n---\n\n".join(
            f"[CONTEXT {index}]\nSection: {row.section}\nContent:\n{row.text}"
            for index, row in enumerate(context_df.itertuples(), start=1)
        )

    def conduct_rag(self, user_query: str, think=False):
        """Retrieve, build the prompt and generate. Returns (thinking, answer).

        The chunks used as context are kept in self.last_context_df.
        """
        self.last_context_df = self.retrieve(user_query)
        return ask(
            tokenizer=self.tokenizer,
            model=self.model,
            system_prompt=build_system_prompt(context=self.build_context(self.last_context_df)),
            question=user_query,
            think=think,
        )
