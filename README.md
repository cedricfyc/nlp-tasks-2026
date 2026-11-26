# NLP Tasks 2026
Repository for the NLP tasks 2026 part of the Master of Artificial Intelligence at CAIRO THWS

# Assignment Selection

The assignments RAG with citations and tool-using agent with a budget were chosen from the [catalogue](https://i-yam.github.io/nlp_milestones/) offered by the [Prof. Dr. Ivan Yamshchikov](https://github.com/i-yam) teaching Natural Language Processing (NLP).

---

## Retrieval-Augmented Generation with Citations

The project undertaken under the topic of RAG with citations is a student assistant bot specialising in matters relevant to international students in Germany. It achieved a success rate of 88 % on a test sample of 100 questions of different domains.

### Project Description

The idea behind this student assistant bot arose due to the difficulties in keeping up to date with the hellscape of German bureacracy. It is not a certified legal advisor, just an assistant providing with facts backed by citations. It aims to refuse user queries that it cannot answer based on its sources.

The following features can be found within the project:
- Embedded documents (by Qwen/Qwen3-Embedding-0.6B) from several reliable sources (e.g. the Federal Ministry of Justice, Make It in Germany, the Federal Foreign Office, etc.) into [SQL database](./data/student_assistant_rag_0_6B.db)
- Retrieval of relevant chunks based on user query based on one or a combination:
    - BM25
    - Dense search
    - SPLADE
- Generation of text by a system-prompted model (Qwen/Qwen3.5-4B)
- Hard-refusal system of increasing complexity:
    - System prompt of the assistant itself
    - BM25 scoring
    - Dense search scoring
    - Prompted LLM judge
- Hard-coded citation system based on the fetched context
- Test on 100 different questions of relevant domains (e.g. starting studies in Germany, rent and moving in/out) and nature (e.g. long context, simple factual, non-answerable, etc.)

---

### Resources Used

The concepts and model stack used for the project are:
- **Embedding Model**: Qwen/Qwen3-Embedding-0.6B
- **Re-ranking Model**: Qwen/Qwen3-Reranker-0.6B
- **Generation Model**: Qwen/Qwen3.5-4B
- **LLM as Judge**: Qwen/Qwen3.5-4B
- **BM25 Library**: rank-bm25
- **Dense Vector Search**
- **SPLADE**: [naver/splade-v3](https://huggingface.co/naver/splade-v3)


---


###  Further Improvements

- Enhanced scraping strategy for more documents (web-scraping was avoided due to the grey area)
- Try larger embedding/reranker models:
    - Qwen/Qwen3-Embedding-4B
    - Qwen/Qwen3-Embedding-8B
    - Qwen/Qwen3-Reranker-4B
    - Qwen/Qwen3-Reranker-8B
- Try interesting/stronger generation models:
    - unsloth/Qwen3.5-4B-GGUF
    - unsloth/Ministral-3-8B-Reasoning-2512-bnb-4bit
    - unsloth/Ministral-3-8B-Instruct-2512-GGUF
    - unsloth/Qwen3.5-9B-GGUF
    - google/gemma-4-E4B
- User query enhancement: Re-write the user query in different forms and/or add more information to the user query from base LLM knowledge
- Compare cross-entropy re-ranking and LLM-reranking
- Experiment with a lightweight architecture for an assistant that runs on most PCs
- Feedback collector that stores user inputs and responses that were not answered properly by the RAG system for further development


---
---
---

## Tool-using agent with a budget

Tool-using agent with a budget can be broken down as follows:
- Point 1
- Point 2
- Point 3

## Team Composition

The team consists of the following members:
- [Cedric Fong Yee Chun](https://github.com/cedricfyc)
- [Hadera Teame Hailu](https://github.com/hadera6)
- [Kevin Thomas](https://github.com/Kevin-1001)
- [Lorena Rama](https://github.com/lorena1616)