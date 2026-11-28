"""Paths, model names and settings shared by all scripts.

Paths are relative to the repo, so the code runs the same in Colab and locally.
Any path can be overridden with an environment variable (RAG_...).
"""

import os
from pathlib import Path

import numpy as np

# ---------------- Paths ----------------
PROJECT_DIR = Path(__file__).resolve().parent          # Assignment 1/Code
ASSIGNMENT_DIR = PROJECT_DIR.parent                     # Assignment 1
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", ASSIGNMENT_DIR / "data"))

DB_PATH = Path(os.environ.get("RAG_DB_PATH", DATA_DIR / "student_assistant_rag_0_6B.db"))
BM25_CACHE = Path(os.environ.get("RAG_BM25_CACHE", DATA_DIR / "bm25_chunks.pkl"))
SPLADE_INDEX_PATH = DATA_DIR / "splade_index.npz"
SPLADE_IDS_PATH = DATA_DIR / "splade_chunk_ids.npy"
#DATASET_PATH = DATA_DIR / "dataset_100.json"
DATASET_PATH = ASSIGNMENT_DIR / "testing" / "Test Datasets" / "dataset_100.json"
CSV_PATH = DATA_DIR / "document_metadata.csv"
#DOCS_PATH = ASSIGNMENT_DIR / "RAG Documents"            # PDFs in RAG Native/ and RAG Screenshots/
DOCS_PATH = DATA_DIR / "raw documentation"              # PDFs in RAG Native/ and RAG Screenshots/
RESULTS_DIR = Path(os.environ.get("RAG_RESULTS_DIR", ASSIGNMENT_DIR / "results"))

# ---------------- Database building ----------------
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
MAX_CHUNK_TOKENS = 512
EMBED_BATCH_SIZE = 32

# ---------------- Retrieval ----------------
EMBEDDING_MODEL_NAME = EMBED_MODEL
EMBED_TYPE = np.float32
TOP_K = 3                     # passages given to the generator (default; the UI can choose 3-5)
TOP_K_CHOICES = (3, 4, 5)     # values POST /ask accepts for top_k
CANDIDATE_K = 20              # candidates per search; the best CANDIDATE_K after fusion go to the re-ranker
RRF_K = 60                    # Reciprocal Rank Fusion constant

SPLADE_MODEL_NAME = "naver/splade-v3"           # gated: accept the licence on Hugging Face
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"  # or "Qwen/Qwen3-Reranker-4B"
RERANK_TOP_N = 6              # default for rerank() called on its own (the pipeline keeps top_k)
RETRIEVER = "both"            # "both" (BM25 + dense), "bm25" or "dense" (baseline comparison)

# ---------------- Generation ----------------
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_PATH = os.environ.get("RAG_MODEL_PATH", "/content/drive/MyDrive/models/Qwen3.5-4B")

# ---------------- Citations ----------------
CITATION_MAX_RETRIES = 1      # re-generations when citations are missing or point to no source
CITATION_STRICT = True        # remove sentences that are still uncited after the retries
CITATION_SNIPPET_CHARS = 300  # length of the source excerpt returned with each citation

# ---------------- Refusal ----------------
# "none"      : only the generator's own refusal (system prompt)
# "bm25" / "dense" / "llm_judge" : calibrated check before generation (calibrate_refusal.py)
# Change to the best method after the comparison (run_eval.py).
REFUSAL_METHOD = os.environ.get("RAG_REFUSAL_METHOD", "none")
REFUSAL_METHODS = ("none", "bm25", "dense", "llm_judge", "combined")   # combined = majority vote of the three
