"""Step 4: Retrieval

Flow: user question -> same embedding model -> question vector
     -> find top_k closest chunks in Chroma
"""
import chromadb
from store import get_model


def retrieve(query, top_k=3, persist_dir="./chroma_db"):
    client = chromadb.PersistentClient(path=persist_dir)
    col = client.get_collection("rag_docs")
    model = get_model()
    q_emb = model.encode([query], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q_emb, n_results=top_k)
    # Returns (chunk id, document text, metadata, distance) — smaller distance = more relevant
    # Ids are returned as well so Step 6 can fuse/dedupe by id in hybrid retrieval
    return list(zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]))


if __name__ == "__main__":
    for cid, doc, meta, dist in retrieve("What is RAG?"):
        print(f"[{cid} | {meta['source']}] distance={dist:.3f}\n{doc[:120]}…\n")
