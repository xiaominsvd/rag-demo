# Intro to RAG

RAG (Retrieval-Augmented Generation) is a technique that lets large language models
take an "open-book exam": before answering a question, the model first retrieves
relevant material from an external knowledge base, then generates an answer grounded
in that material.

## Why RAG

1. Knowledge cutoff: a model's training data has a cutoff date; it knows nothing about what happened after.
2. Hallucination: when a model doesn't know something it fabricates with a straight face; RAG constrains it with real sources.
3. Private knowledge: internal company docs and personal notes never made it into pretraining — RAG is how you inject them.

## The Three Stages of RAG

- Indexing: chunk documents -> embed into vectors -> store in a vector database.
- Retrieval: embed the user's question -> find the top_k most similar chunks in the vector store.
- Generation: stuff the retrieved chunks into the prompt; the LLM answers based on the sources.

## Limitations of RAG

If retrieval is wrong, generation is wrong — "garbage in, garbage out."
Chunking strategy, embedding model quality, and the choice of top_k all directly
affect the final result.
