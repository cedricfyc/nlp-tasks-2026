"""Paths, model names and settings shared by all scripts.

Paths are relative to the repo, so the code runs the same in Colab and locally.
Any path can be overridden with an environment variable (RAG_...).
"""

import os
from pathlib import Path

import numpy as np

# ---------------- Paths ----------------
PROJECT_DIR = Path(__file__).resolve().parent          # Assignment 1/rag_project
ASSIGNMENT_DIR = PROJECT_DIR.parent                     # Assignment 1
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", ASSIGNMENT_DIR / "data"))

DB_PATH = Path(os.environ.get("RAG_DB_PATH", DATA_DIR / "student_assistant_rag_0_6B.db"))
BM25_CACHE = Path(os.environ.get("RAG_BM25_CACHE", DATA_DIR / "bm25_chunks.pkl"))
SPLADE_INDEX_PATH = DATA_DIR / "splade_index.npz"
SPLADE_IDS_PATH = DATA_DIR / "splade_chunk_ids.npy"
DATASET_PATH = DATA_DIR / "dataset_100.json"
CSV_PATH = DATA_DIR / "document_metadata.csv"
DOCS_PATH = ASSIGNMENT_DIR / "RAG Documents" / "rag_documents"

# ---------------- Database building ----------------
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
MAX_CHUNK_TOKENS = 512
EMBED_BATCH_SIZE = 32

# ---------------- Retrieval ----------------
EMBEDDING_MODEL_NAME = EMBED_MODEL
EMBED_TYPE = np.float32
TOP_K = 3                     # chunks per search method
RRF_K = 60                    # Reciprocal Rank Fusion constant

SPLADE_MODEL_NAME = "naver/splade-v3"           # gated: accept the licence on Hugging Face
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"  # or "Qwen/Qwen3-Reranker-4B"
RERANK_TOP_N = 6              # chunks kept after re-ranking

# ---------------- Generation ----------------
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_PATH = os.environ.get("RAG_MODEL_PATH", "/content/drive/MyDrive/models/Qwen3.5-4B")
