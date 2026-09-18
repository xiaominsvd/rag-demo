# RAG Demo: Local Document Q&A Assistant

A minimal but complete RAG system: ask questions about the documents in `docs/`,
the system retrieves relevant passages and generates an answer grounded in them.

```
User question
  │
  ▼  embed
Retrieval: find top_k most similar chunks in Chroma
  │
  ▼  stuff into prompt
Generation: LLM answers from the sources (falls back to extractive display without an API key)
```

## 7 Steps

- **Step 0 Setup**: `pip install -r requirements.txt` (Python ≥ 3.9)
- **Step 1 Prepare docs**: `docs/` already has 3 sample documents; swap in your own `.md` files
- **Step 2 Chunking**: see `src/chunking.py`; run `python src/chunking.py` to see chunking in action
- **Step 3 Embed + index**: see `src/store.py`
- **Step 4 Retrieve**: see `src/retrieve.py`
- **Step 5 Generate**: see `src/answer.py`
- Run it end to end: `python scripts/build_index.py` then `python scripts/ask.py "What's the difference between RAG and fine-tuning?"`

## Step 6 Where to Go Next (after the basics)

1. Hybrid retrieval (vectors + BM25)
2. Reranking
3. Semantic chunking by heading, summaries on chunks
4. Eval set: prepare 20 Q&A pairs to quantify each change

## FAQ

- The embedding model downloads automatically from HuggingFace on first run (~100MB).
- For real LLM generation: `export OPENAI_API_KEY="sk-..."` before running `ask.py`.
