Use ths directory for everything that is needed for the scripts to run i.e. models, chunk database, outputs, etc.

- bm25_chunks.pkl: contains the compiled BM25-indexed database. No need to build the bm25 database again, unless new documents are added or existing ones are removed.

- student_assistant_rag_0_6B.db: the SQL database can be viewed using [this SQLite browser](https://sqlitebrowser.org/dl/). It was constructed using the embeddings from qwen3-embeddings-0.6B.