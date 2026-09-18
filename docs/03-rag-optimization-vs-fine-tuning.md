# Optimizing RAG

## Common Optimizations

- Chunking strategy: semantic chunking (by heading/paragraph) beats fixed-length chunking; adding titles or summaries to chunks improves retrieval.
- Hybrid retrieval: union of vector search + keyword search (BM25) covers both semantics and exact matches.
- Reranking: recall 20 candidates with vectors, then use a cross-encoder to rerank and keep the top 3.
- Query rewriting: rewrite the user's question into a more retrieval-friendly form, or use HyDE (have the model draft a hypothetical answer first, then search with it).
- Citation constraints: instruct the model in the prompt to answer only from the sources and cite them.

## RAG vs Fine-tuning

| Dimension | RAG | Fine-tuning |
|---|---|---|
| Knowledge updates | Add docs and rebuild the index — minutes | Retraining required — hours to days |
| Cost | Low | High (compute + data labeling) |
| Use cases | Knowledge Q&A, citation tracing | Style, format, task-specific behavior |
| Hallucination | Constrained by sources, less frequent | Can still fabricate |

They're not mutually exclusive: many production systems use "fine-tuning for behavior and style, RAG for knowledge."
