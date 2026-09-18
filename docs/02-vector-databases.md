# Vector Databases

Vector databases are purpose-built for storing high-dimensional vectors and
supporting "find the N most similar" approximate nearest neighbor (ANN) queries —
something traditional databases aren't good at.

## How They Work

1. An embedding model maps text into vectors of several hundred dimensions; texts with similar meaning end up close together in vector space.
2. At query time the question is also embedded, and distances (e.g. cosine distance) to the stored vectors are computed.
3. Index algorithms like HNSW quickly return the top_k closest results.

## Common Choices

- Chroma: lightweight, Python-native; great for prototypes and small-scale projects like this one.
- FAISS: Meta's open-source vector search engine — fast, but a library, not a full database.
- Qdrant / Weaviate / Milvus: production-grade, with distributed mode, filtering, and hybrid search.
- pgvector: a Postgres extension — zero-ops start if your team already runs PG.

## Selection Advice

Under 1M vectors on a single machine: Chroma or FAISS is enough.
Need multi-tenancy, permission filtering, high availability: consider Qdrant or Milvus.
