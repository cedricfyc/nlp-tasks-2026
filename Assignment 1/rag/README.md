# RAG with Citations

The overall task consists in creating...

## Task Description

**Technique**: Chunking, embeddings, vector store, generation that cites the retrieved passage and refuses when nothing relevant is found; a BM25 baseline for comparison.

**Must show**: An unanswerable-question test with a refusal rate.

**Examples**: A Telegram bot for a hobby community's rulebook (board-game, climbing gym, sailing club); a chat over one municipality's public council minutes; a web app answering questions about a hiking region's trail guides.

**Ships**: POST /ask returning answer plus citations over a public corpus; the question set in the repo.


---


## Ideas for RAG with citations
- [Architecture Explained](https://chatgpt.com/share/6af25e24-7a1c-83ed-8d8f-a6b84c130f9a)
- [Student Assistant Topic Explained](https://chatgpt.com/share/6aecf7b2-9b60-83ed-b447-074c26fce784)
- [Claude Coding Prompt](https://claude.ai/share/3eff6ba1-cc04-4eed-8a1e-a5cb6fb0b149)
- Consult [this Google Doc for ideas](https://docs.google.com/document/d/16--8Z7NYCgoXtqeZQsS3nFFvpoXB7jC_g4C89cRvQqo/edit?usp=sharing)

---


## Model Choice
- **Embedder**: Qwen3-Embedding-0.6B (1024-dimensional vectors, stored in the SQLite database)
- **Retrieval**
    - BM25 (`rank_bm25`), the baseline
    - Dense search: cosine similarity over the chunk embeddings
    - SPLADE (naver/splade-v3), optional
    - Reciprocal Rank Fusion (k = 60) to merge the searches
- **Re-ranker**: Qwen3-Reranker-0.6B, optional; scores the best 20 candidates and keeps the top k (3, 4 or 5)
- **Generator**: Qwen3.5-4B, answering only from the numbered passages, with hard citations [n]


---


## Experiment Set-Up
1. **BM25 baseline**: BM25 only
2. **All retrievers**: BM25 + dense + SPLADE, RRF, re-ranker; no refusal check
3. **Full system**: setting 2 + `combined` refusal check


---


## Refusal Rate

The assistant refuses with *"I don't have enough information to answer that."* in three layers:

1. **Generator prompt**: the model is told to give that sentence when the passages do not contain the answer.
2. **Citation check**: uncited sentences are removed; an answer with no valid citation left becomes the refusal.
3. **Refusal check before generation** (optional, `refusal_method`), so the model never runs on the question:

| Method | Signal |
|---|---|
| `bm25` | BM25 scores: top score, word coverage, gap to the next results |
| `dense` | embedding similarity: top score, mean, gap, z-score |
| `llm_judge` | Qwen3.5-4B reads the passages: probability that they answer the question |
| `combined` | majority vote: refuse if at least 2 of the 3 methods say refuse |

Each of `bm25`, `dense` and `llm_judge` is a logistic regression on its signals, calibrated on the
100 questions of `data/dataset_100.json` (20 unanswerable) with a cross-validated cut-off that refuses
at most 10 % of the answerable questions (`calibrate_refusal.py`).

---

## Reproducibility

### Requirements
- **Hardware**: NVIDIA GPU with ~15 GB memory, e.g. the free **T4** in Google Colab. The full system
  uses ~12–13 GB, so only one model copy fits: run one command / one app at a time.
- **Software**: Python 3.10+, `torch` with CUDA (preinstalled in Colab), `pip install -r requirements.txt`
  (`docling` only for rebuilding the database).
- **Models**: Qwen3.5-4B (~9.3 GB) on Google Drive in `MyDrive/models/Qwen3.5-4B`.
  Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B and naver/splade-v3 download automatically each session.
- **Hugging Face token** (Read), only for SPLADE: accept the licence at
  <https://huggingface.co/naver/splade-v3> with the same account.

### How to run
On any machine the steps are the same:
1. **Get the files**: the whole `Assignment 1/` folder (`Code/`, `data/`, `requirements.txt`).
2. **Install**: `pip install -r requirements.txt`.
3. **Point to the generator**: set `RAG_MODEL_PATH` to the Qwen3.5-4B folder.
4. **Log in to Hugging Face** (only for SPLADE).
5. **One-time steps** (skip if the files exist), from `Code/`:
   ```bash
   python build_database.py --splade             # SPLADE index (--force: rebuild)
   python calibrate_refusal.py --method dense     # refusal cut-off (--method bm25 | dense | llm_judge)
   python calibrate_refusal.py --method llm_judge --retriever both --splade --reranker   # same search as the app
   ```
6. **Start the app** (full system), from `Code/`:
   ```bash
   python api.py --splade --reranker --refusal combined   # full system (--refusal none | bm25 | dense | llm_judge | combined)
   ```
   or in Python:
   ```python
   rag = RAGPipeline(use_splade=True, use_reranker=True,     # SPLADE and re-ranker on
                     refusal_method="combined",              # or "none" | "bm25" | "dense" | "llm_judge"
                     retriever="both")                       # or "bm25" | "dense"
   start_in_background(rag)                                  # chat page + POST /ask
   ```

**On your own computer** (NVIDIA GPU):
```bash
cd "Assignment 1"
pip install -r requirements.txt
export RAG_MODEL_PATH=/path/to/Qwen3.5-4B        # generator folder (Windows: set RAG_MODEL_PATH=...)
huggingface-cli login                            # token, only for SPLADE
cd Code
python api.py --splade --reranker --refusal combined   # open http://localhost:8000
```

**In Google Colab** (T4 GPU): put the folder on Drive as `MyDrive/nlp-tasks-2026/Assignment 1` and
Qwen3.5-4B in `MyDrive/models/Qwen3.5-4B`, open `run_rag_colab.ipynb`
(Runtime → Change runtime type → T4 GPU) and run its cells (numbers count markdown cells too):

| Cell | Does | When |
|---|---|---|
| 3 | mount Drive, `cd` to `Code/`, install packages | every session |
| 5 | model path, results folder, Hugging Face login | every session |
| 7 | one-time steps (skip if the files exist) | once |
| 11 | start the app (set the `RAGPipeline(...)` line above) | to use the assistant |

Run the app once per session, from an empty GPU (Colab: Runtime → Disconnect and delete runtime).

### Settings

**How to apply a setting** (three ways; an option you pass wins over `config.py`):

1. **Command-line options**, when starting a script from `Code/` (in Colab put `!` in front):
   ```bash
   python api.py --splade --reranker --refusal combined   # switches (--splade, --reranker) and --name value
   python api.py --retriever bm25 --refusal none          # baseline (--retriever both | bm25 | dense)
   python run_rag.py --refusal dense "What is a Sperrkonto?"   # ask from the terminal
   ```
   `python <script>.py --help` lists every option of a script.
2. **Arguments in Python** (e.g. the notebook's app cell):
   ```python
   rag = RAGPipeline(use_splade=True, use_reranker=True, refusal_method="combined", retriever="both")
   ```
3. **Defaults** for everything: edit the value in `Code/config.py`, or set its environment variable
   before starting (`export RAG_REFUSAL_METHOD=combined`; in Colab `os.environ["RAG_REFUSAL_METHOD"] = "combined"`).

`top_k` is chosen per question in the chat page, or sent with `POST /ask` (`"top_k": 4`).

| Python argument | Command-line option | What it does |
|---|---|---|
| `retriever="both"` (default) | `--retriever both` | BM25 (keywords) + dense (meaning) search, merged with RRF |
| `retriever="bm25"` | `--retriever bm25` | keyword search only (the baseline) |
| `retriever="dense"` | `--retriever dense` | embedding search only |
| `use_splade=True` | `--splade` | adds SPLADE (learned keyword weights) as a third search |
| `use_reranker=True` | `--reranker` | Qwen3-Reranker re-scores the best 20 candidates and keeps the top k |
| `refusal_method="none"` (default) | `--refusal none` | no check before answering; the model and the citation check can still refuse |
| `refusal_method="bm25"` / `"dense"` / `"llm_judge"` | `--refusal bm25` / `dense` / `llm_judge` | refuses before answering when that signal says the guides don't cover the question |
| `refusal_method="combined"` | `--refusal combined` | refuses when at least 2 of bm25, dense and llm_judge say refuse |
| `top_k=3` (default), 4 or 5 | (chat page / `POST /ask`) | how many passages the answer uses |

Full system: `use_splade=True, use_reranker=True, refusal_method="combined", retriever="both"`.

**`Code/config.py`** (defaults used when nothing else is passed):

| Setting | Default | What it does |
|---|---|---|
| `MODEL_PATH` (env `RAG_MODEL_PATH`) | `/content/drive/MyDrive/models/Qwen3.5-4B` | folder of the generator |
| `EMBED_MODEL`, `SPLADE_MODEL_NAME`, `RERANKER_MODEL_NAME` | Qwen3-Embedding-0.6B, naver/splade-v3, Qwen3-Reranker-0.6B | the search models |
| `DATA_DIR` (env `RAG_DATA_DIR`) | `Assignment 1/data` | database, indexes, calibration files |
| `RESULTS_DIR` (env `RAG_RESULTS_DIR`) | `Assignment 1/results` | output folder of `run_eval.py` |
| `RETRIEVER` | `"both"` | default retriever |
| `REFUSAL_METHOD` (env `RAG_REFUSAL_METHOD`) | `"none"` | default refusal check |
| `TOP_K` / `TOP_K_CHOICES` | 3 / (3, 4, 5) | passages per answer / values the page and `POST /ask` accept |
| `CANDIDATE_K` | 20 | candidates per search, and how many go to the re-ranker |
| `RRF_K` | 60 | Reciprocal Rank Fusion constant |
| `CITATION_MAX_RETRIES` | 1 | extra generations when citations are missing or wrong |
| `CITATION_STRICT` | `True` | removes sentences that are still uncited |
| `CITATION_SNIPPET_CHARS` | 300 | length of the source excerpt shown per citation |
| `MAX_CHUNK_TOKENS`, `EMBED_BATCH_SIZE` | 512, 32 | chunk size and batch size when building the database |

### Demo UI and POST /ask
The chat page shows the answer with clickable `[n]` citations, the sources, refusals (orange card),
a 3/4/5 sources switch and a history of the tab's questions (browser only).
`POST /ask {"question": "...", "top_k": 3}` returns `answer`, `citations`, `refused`, `retrieved`
(passages with re-ranker scores), the refusal decision, `top_k` and `seconds`.
Each search returns 20 candidates; the re-ranker keeps the best `top_k`.

### Files created once (skipped when they already exist)
| File in `data/` | Created by | Status |
|---|---|---|
| `student_assistant_rag_0_6B.db` (database, embeddings) | `python build_database.py` | in the repo |
| `bm25_chunks.pkl` (BM25 index) | `python build_database.py --bm25` | in the repo |
| `splade_index.npz` + chunk ids | `python build_database.py --splade` | one-time step |
| `refusal_bm25.json` | `python calibrate_refusal.py --method bm25` | in the repo |
| `refusal_dense.json` | `python calibrate_refusal.py --method dense` | one-time step |
| `refusal_llm_judge.json` | `python calibrate_refusal.py --method llm_judge --retriever both --splade --reranker` | one-time step |

`combined` needs the three refusal files and no calibration of its own. Add `--force` to redo a step.
Commit new files so nobody has to repeat them. Never run `build_database.py --merge` on the main database.

### When to rebuild or recalibrate
| What changed | Redo |
|---|---|
| Documents added or changed | `build_database.py`, then `--bm25`, `--splade --force`, and all calibrations with `--force` |
| BM25 index rebuilt | `calibrate_refusal.py --method bm25 --force` |
| Embedding model changed | the database embeddings, then `--method dense --force` |
| Generator, `TOP_K`, retriever, SPLADE or re-ranker changed | `--method llm_judge --force` |
| Question set (`data/dataset_100.json`) changed | all calibrations with `--force` |

Nothing changed → nothing to redo.

### Troubleshooting
| Problem | Fix |
|---|---|
| `OutOfMemoryError` | a model is still in the GPU: Disconnect and delete runtime, run cells 3, 5, 11 once |
| *No backend with GPU available* | GPU quota used up: try later or another account |
| `GatedRepoError 403` (SPLADE) | request access on the splade-v3 page with the token's account |
| `requirements.txt` not found / `ModuleNotFoundError` | wrong Drive path, or incomplete `Code/` folder |
| `JSONDecodeError` / `header too large` | broken download: free Drive space, keep the Hugging Face cache off Drive |

## Folder Contents

- `Code/`: the Python modules (`pipeline.py`, `retrieval.py`, `generation.py`, `citations.py`,
  `refusal.py`, `api.py`, …), the chat page `ui/`, and the team's `Notebooks/` and `Test Scripts/`
- `data/`: database, BM25 index, question sets, refusal calibrations
- `RAG Documents/`: the 78 source PDFs (only needed to rebuild the database)
- `run_rag_colab.ipynb`: Colab launcher (no logic)

---
