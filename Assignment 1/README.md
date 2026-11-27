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
- **Embedder Model**: Qwen3-Embedding-0.6B/4B/8B
- **Retrieval Model**
    - BM25 as baseline
    - FAISS
    - BM25 + FAISS + RRF
    - Qwen3-Reranker-0.6B/4B/8B
- **Generator Model**: Qwen3.8-27B


---


## Experiment Set-Up
1. **BM25 baseline**
- Lexical retrieval by BM25 algorithm
- Words and terms between query and documents are matched.
- Retrieved context is appended to the query.
- Generator can work on the added context.

2. **FAISS**
- Toolbox of indexing methods and related primitives for search, clustering, compression, and transformation of vectors.
- contains algorithms that search in sets of vectors of any size.

3. **BM25 + FAISS + RRF**
- Reciprocal Rank Fusion enables the mixing of rank results of documents retrieved by BM25 and FAISS.
- Higher RRF score means higher combined relevance.

4. **BM25 + FAISS + RRF + Reranker 0.6B**
**TO BE CONTINUED**

5. **BM25 + FAISS + RRF + Reranker 4B**
**TO BE CONTINUED**


The embedder model of choice would be Qwen3-Embedding-0.6B for a start. It embeds the **TO BE CONTINUED**


---


## Refusal Rate

**Approach A — Similarity threshold** (Simple baseline)
Use FAISS similarity scores to determine whether retrieved passages are sufficiently relevant.

**Approach B — Reranker score threshold** (Recommended)
Use Qwen3-Reranker to assess relevance between question and retrieved passages.

**Approach C — Evidence sufficiency classifier** (Strong practical option)
Introduce a separate classification step between retrieval and generation.
It could be implemented using one of the following:
- Qwen3.8-27B with structured JSON output.
- A smaller classification model.
- A fine-tuned binary or three-class classifier.

|Decision	            |   Meaning                         |
|-----------------------|-----------------------------------|
|ANSWERABLE	            |   Evidence supports answer        |
|PARTIALLY_ANSWERABLE	|   Evidence supports only part     |
|UNANSWERABLE	        |   Evidence insufficient           |

---

## Reproducibility

### How to run
Put the `Assignment 1` folder on Google Drive (`MyDrive/Assignment 1` or `MyDrive/nlp-tasks-2026/Assignment 1`),
put Qwen3.5-4B in `MyDrive/models/Qwen3.5-4B`, open `run_rag_colab.ipynb` in Colab (T4 GPU) and run it from top to bottom.
Locally, run the same commands from `Assignment 1/Code/` without the `!` (an NVIDIA GPU is needed).

### Files created once (skipped when they already exist)
| File in `data/` | Created by | Status |
|---|---|---|
| `student_assistant_rag_0_6B.db` (database, embeddings) | `python build_database.py` | in the repo |
| `bm25_chunks.pkl` (BM25 index) | `python build_database.py --bm25` | in the repo |
| `splade_index.npz` + chunk ids | `python build_database.py --splade` | one-time step |
| `refusal_bm25.json` | `python calibrate_refusal.py --method bm25` | in the repo |
| `refusal_dense.json` | `python calibrate_refusal.py --method dense` | one-time step |
| `refusal_llm_judge.json` | `python calibrate_refusal.py --method llm_judge --retriever both --splade --reranker` | one-time step |

Refusal `combined` is a majority vote of the three files above (refuse if 2 of 3 say refuse);
it needs no calibration of its own.

After a one-time step, commit the new file so nobody has to repeat it.
`build_database.py --splade` and `calibrate_refusal.py` skip an existing file; add `--force` to redo it.
`build_database.py --bm25` always rebuilds (fast). `build_database.py` with no flag only adds documents
that are not yet in the database.

### When to rebuild or recalibrate
Only redo a step when something it depends on has changed:

| What changed | Redo |
|---|---|
| Documents added or changed | `build_database.py`, then `--bm25`, `--splade --force`, and all calibrations with `--force` |
| BM25 index rebuilt | `calibrate_refusal.py --method bm25 --force` |
| Embedding model changed | the database embeddings, then `--method dense --force` |
| Generator model, `TOP_K`, retriever, SPLADE or re-ranker changed | `--method llm_judge --force` (same retrieval options as the evaluation run) |
| Question set (`data/dataset_100.json`) changed | all calibrations with `--force` |
| Evaluation shows too many / too few refusals | that method with `--force --max-false-refusal <value>` |

Nothing changed → nothing to redo. Do not run `build_database.py --merge` on the main database:
it deletes and rewrites the output file.

### Evaluation results
Questions: `data/dataset_100.json` is used to calibrate the refusal methods; `data/test_60.json`
(20 unanswerable, 15 simple, 15 multi-domain, 10 long-context, none repeated from the 100) is the
held-out test set for the final numbers (`--dataset ../data/test_60.json`).
`python run_eval.py [--retriever bm25|dense|both] [--splade] [--reranker] [--refusal none|bm25|dense|llm_judge|combined] --judge`
writes `results/rag_<setting>.db`, `judge_<setting>.db` and `metrics_<setting>.json`
(with `_test_60` added to the setting name for the test set).
Runs are resumable: run the same command again after a disconnect. Answers must be generated again
only when the pipeline setting changes (each setting has its own files).

### Hardware Requirements

### Software Requirements

## Folder Contents

- RAG Documents:

---