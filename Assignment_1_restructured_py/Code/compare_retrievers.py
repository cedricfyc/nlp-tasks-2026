"""Compare BM25 and dense search side by side for one question.

    python compare_retrievers.py
    python compare_retrievers.py "How do I get my Mietkaution back?" --top-k 10
"""

import argparse

from retrieval import compare_search_algorithms, load_bm25, load_embedding_matrix

DEFAULT_QUERY = "I am gay. Can I get married to a North Korean?"


def main(query=DEFAULT_QUERY, top_k=10):
    bm25_payload = load_bm25()
    vector_chunk_ids, vector_matrix = load_embedding_matrix()
    print("Loaded vectors:", vector_matrix.shape)
    compare_search_algorithms(query, bm25_payload, vector_matrix, vector_chunk_ids, top_k=top_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    main(args.query, args.top_k)
