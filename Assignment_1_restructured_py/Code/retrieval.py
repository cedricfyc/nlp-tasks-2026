"""Search methods: BM25, dense, SPLADE, Reciprocal Rank Fusion and the Qwen3 re-ranker.

Every search returns a DataFrame of chunks (from fetch_chunk_citations) with
its own <method>_score and <method>_rank columns, best first.
"""

import pickle
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz, vstack

from config import (
    BM25_CACHE, DB_PATH, EMBED_TYPE, EMBEDDING_MODEL_NAME, RERANK_TOP_N,
    RERANKER_MODEL_NAME, RRF_K, SPLADE_IDS_PATH, SPLADE_INDEX_PATH,
    SPLADE_MODEL_NAME, TOP_K,
)
from data import connect_db, fetch_chunk_citations, get_db_signature


def _top_k_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    """Indices of the top_k highest scores, best first (argpartition avoids a full sort)."""
    k = min(top_k, len(scores))
    if k == 0:
        return np.array([], dtype=int)
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]


def _ranked_results(chunk_ids: list[int], scores: np.ndarray, idx: np.ndarray,
                    method: str) -> pd.DataFrame:
    """Fetch the chunks at idx and add <method>_score and <method>_rank columns."""
    ranked_ids = [chunk_ids[i] for i in idx]
    results = fetch_chunk_citations(ranked_ids)
    if results.empty:
        return results
    results[f"{method}_score"] = results["chunk_id"].map(
        {chunk_ids[i]: float(scores[i]) for i in idx})
    results[f"{method}_rank"] = results["chunk_id"].map(
        {chunk_id: rank for rank, chunk_id in enumerate(ranked_ids, start=1)})
    return results.sort_values(f"{method}_rank").reset_index(drop=True)


# =====================================================================
# BM25 (lexical search)
# =====================================================================
def bm25_tokenise(text):
    """Unicode-aware tokeniser preserving words and hyphenated terms."""
    return re.findall(r"\b[\w-]+\b", str(text).lower(), flags=re.UNICODE)


def read_bm25_corpus():
    """Return (chunk_ids, embed_texts) of all non-empty chunks."""
    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT chunk_id, embed_text
            FROM chunks
            WHERE embed_text IS NOT NULL AND trim(embed_text) <> ''
            ORDER BY chunk_id
            """
        ).fetchall()
    return [int(row["chunk_id"]) for row in rows], [row["embed_text"] for row in rows]


def build_and_save_bm25(cache_path=BM25_CACHE):
    """Build the BM25 index over all chunks and save it (run once)."""
    from rank_bm25 import BM25Okapi

    chunk_ids, texts = read_bm25_corpus()
    if not texts:
        raise ValueError("No non-empty chunks.embed_text rows found.")

    payload = {
        "format_version": 1,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "db_path": str(Path(DB_PATH).resolve()),
        "signature": get_db_signature(),
        "chunk_ids": chunk_ids,
        "bm25": BM25Okapi([bm25_tokenise(text) for text in texts]),
    }

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved BM25 index: {cache_path.resolve()}")
    print(f"Indexed chunks: {len(chunk_ids):,}")
    return payload


def load_bm25(cache_path=BM25_CACHE, validate=True):
    """Load the saved BM25 index; optionally check it still matches the database."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"BM25 cache not found: {cache_path}. Run build_and_save_bm25() first.")

    with cache_path.open("rb") as f:
        payload = pickle.load(f)

    if payload.get("format_version") != 1:
        raise ValueError("Unsupported BM25 cache format. Rebuild the index.")

    if validate and get_db_signature() != payload["signature"]:
        raise RuntimeError(
            "BM25 cache may be stale. Database row count or max chunk_id changed. "
            "Run build_and_save_bm25() to refresh it."
        )

    print(f"Loaded BM25 index with {len(payload['chunk_ids']):,} chunks "
          f"(built {payload['built_at']}).")
    return payload


def bm25_search(query, bm25_payload, top_k=TOP_K) -> pd.DataFrame:
    """Tokenise the search query and search for keywords with BM25."""
    scores = np.asarray(bm25_payload["bm25"].get_scores(bm25_tokenise(query)))
    idx = _top_k_indices(scores, top_k)
    if idx.size == 0:
        return pd.DataFrame()
    return _ranked_results(bm25_payload["chunk_ids"], scores, idx, "bm25")


# =====================================================================
# Dense search (Qwen3-Embedding)
# =====================================================================
def load_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    """Load the embedding model once (pass it to dense_search)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def load_embedding_matrix():
    """Load all chunk embeddings as a row-normalised matrix. Returns (chunk_ids, matrix)."""
    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT chunk_id, embedding
            FROM chunks
            WHERE embedding IS NOT NULL
            ORDER BY chunk_id
            """
        ).fetchall()

    ids = [int(row["chunk_id"]) for row in rows]
    vectors = [np.frombuffer(row["embedding"], dtype=EMBED_TYPE) for row in rows]

    if not vectors:
        raise ValueError("No embeddings found in chunks.embedding")
    dims = {v.size for v in vectors}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent embedding dimensions found: {sorted(dims)}")

    matrix = np.vstack(vectors)
    # Normalise rows: cosine similarity then equals dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return ids, matrix / np.maximum(norms, 1e-12)


def dense_search(vector_matrix, vector_chunk_ids, query, top_k=TOP_K, model=None) -> pd.DataFrame:
    """Encode the query and rank chunks by cosine similarity."""
    model = model or load_embedding_model()
    query_vector = model.encode(
        [query], prompt_name="query", normalize_embeddings=True, convert_to_numpy=True,
    )[0].astype(np.float32)

    assert vector_matrix.shape[1] == query_vector.shape[0], (
        f"Dimension mismatch: documents={vector_matrix.shape[1]}, query={query_vector.shape[0]}"
    )
    scores = vector_matrix @ query_vector
    return _ranked_results(vector_chunk_ids, scores, _top_k_indices(scores, top_k), "dense")


def compare_search_algorithms(query, bm25_payload, vector_matrix, vector_chunk_ids,
                              top_k=TOP_K, embedding_model=None, show=True):
    """Run BM25 and dense search; optionally show which chunks both found.

    Returns (lexical, semantic).
    """
    lexical = bm25_search(query=query, bm25_payload=bm25_payload, top_k=top_k)
    semantic = dense_search(vector_matrix, vector_chunk_ids, query, top_k=top_k,
                            model=embedding_model)

    if show:
        comparison = semantic[[
            "chunk_id", "doc_id", "filename", "source_title", "section",
            "page_start", "page_end", "text", "dense_score", "dense_rank",
        ]].merge(lexical[["chunk_id", "bm25_score", "bm25_rank"]], on="chunk_id", how="outer")
        comparison["in_both"] = comparison["dense_rank"].notna() & comparison["bm25_rank"].notna()
        comparison = comparison.sort_values(
            ["in_both", "dense_rank", "bm25_rank"], ascending=[False, True, True], na_position="last",
        ).reset_index(drop=True)

        try:
            from IPython.display import display
            display(comparison)
        except ImportError:
            print(comparison.drop(columns="text").to_string())
        print(f"Top-{top_k} overlap:",
              len(set(semantic["chunk_id"]) & set(lexical["chunk_id"])), "chunks")

    return lexical, semantic


# =====================================================================
# SPLADE (learned sparse search)
# =====================================================================
def load_splade_model(model_name=SPLADE_MODEL_NAME):
    """Load the SPLADE encoder (sentence-transformers >= 5.0)."""
    from sentence_transformers import SparseEncoder

    return SparseEncoder(model_name)


def sparse_to_csr(tensor) -> csr_matrix:
    """Convert a torch sparse tensor (n, vocab_size) to a scipy CSR matrix."""
    tensor = tensor.coalesce().cpu()
    rows, cols = tensor.indices().numpy()
    values = tensor.values().float().numpy()
    return csr_matrix((values, (rows, cols)), shape=tuple(tensor.shape))


def build_splade_index(model, index_path=SPLADE_INDEX_PATH, ids_path=SPLADE_IDS_PATH,
                       batch_size=32):
    """Encode every chunk once with SPLADE and save the sparse index."""
    chunk_ids, texts = read_bm25_corpus()
    parts = []
    for start in range(0, len(texts), batch_size):
        parts.append(sparse_to_csr(model.encode_document(texts[start:start + batch_size])))
        print(f"SPLADE encoded {min(start + batch_size, len(texts)):,}/{len(texts):,}")

    matrix = vstack(parts).tocsr()
    save_npz(Path(index_path), matrix)
    np.save(Path(ids_path), np.array(chunk_ids, dtype=np.int64))
    print(f"Saved SPLADE index {matrix.shape} to {index_path}")
    return chunk_ids, matrix


def load_splade_index(index_path=SPLADE_INDEX_PATH, ids_path=SPLADE_IDS_PATH):
    """Load the saved SPLADE index. Returns (chunk_ids, sparse matrix)."""
    if not Path(index_path).exists():
        raise FileNotFoundError(f"{index_path} not found. Run build_splade_index() first.")
    return np.load(Path(ids_path)).tolist(), load_npz(Path(index_path)).tocsr()


def splade_search(model, splade_matrix, splade_chunk_ids, query, top_k=TOP_K) -> pd.DataFrame:
    """Rank chunks by the dot product of SPLADE word weights."""
    query_vector = sparse_to_csr(model.encode_query([query]))
    scores = (splade_matrix @ query_vector.T).toarray().ravel()
    return _ranked_results(splade_chunk_ids, scores, _top_k_indices(scores, top_k), "splade")


# =====================================================================
# Fusion and re-ranking
# =====================================================================
def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = RRF_K) -> list[int]:
    """Merge ranked chunk-id lists: score += 1 / (k + rank + 1). Best first."""
    rrf_scores = {}
    for ranked_ids in ranked_lists:
        for rank, chunk_id in enumerate(ranked_ids):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    return sorted(rrf_scores, key=rrf_scores.get, reverse=True)


class Qwen3Reranker:
    """Scores how well each chunk answers the question (0 to 1).

    Qwen3-Reranker model card: the model answers "yes" or "no";
    the score is the probability of "yes".
    """

    PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on "
        "the Query and the Instruct provided. Note that the answer can only be \"yes\" or "
        "\"no\".<|im_end|>\n<|im_start|>user\n"
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    INSTRUCTION = ("Given a question from an international student about studying or living "
                   "in Germany, judge whether the document helps answer the question")

    def __init__(self, model_name=RERANKER_MODEL_NAME, max_length=8192, batch_size=8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map="auto").eval()
        self.max_length, self.batch_size = max_length, batch_size
        self.token_yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_no = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_tokens = self.tokenizer.encode(self.PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.SUFFIX, add_special_tokens=False)

    def _tokenise(self, pairs):
        inputs = self.tokenizer(
            pairs, padding=False, truncation="longest_first", return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        inputs["input_ids"] = [self.prefix_tokens + ids + self.suffix_tokens
                               for ids in inputs["input_ids"]]
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt",
                                    max_length=self.max_length)
        return {key: value.to(self.model.device) for key, value in inputs.items()}

    def score(self, query, documents) -> list[float]:
        """One relevance score (0 to 1) per document."""
        torch, scores = self.torch, []
        with torch.no_grad():
            for start in range(0, len(documents), self.batch_size):
                pairs = [f"<Instruct>: {self.INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"
                         for doc in documents[start:start + self.batch_size]]
                logits = self.model(**self._tokenise(pairs)).logits[:, -1, :]
                yes_no = torch.stack([logits[:, self.token_no], logits[:, self.token_yes]], dim=1)
                scores += torch.nn.functional.log_softmax(yes_no.float(), dim=1)[:, 1].exp().tolist()
        return scores


def rerank(reranker, query, context_df: pd.DataFrame, top_n=RERANK_TOP_N) -> pd.DataFrame:
    """Re-sort chunks by re-ranker score and keep the best top_n."""
    if context_df.empty:
        return context_df
    documents = [f"{row.section or ''}\n{row.text}" for row in context_df.itertuples()]
    context_df = context_df.copy()
    context_df["rerank_score"] = reranker.score(query, documents)
    return (context_df.sort_values("rerank_score", ascending=False)
            .head(top_n).reset_index(drop=True))
