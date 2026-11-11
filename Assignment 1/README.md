# RAG with Citations

The overall task consists in creating 

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

### Hardware Requirements

### Software Requirements


---