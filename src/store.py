"""Step 3: Embedding + storing in the vector database

Flow: chunk -> embedding model -> vector -> persisted in Chroma
"""
import chromadb
from sentence_transformers import SentenceTransformer

# Compact embedding model, ~100MB, auto-downloaded on first run
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def get_model():
    return SentenceTransformer(MODEL_NAME)


def build_collection(chunks, metadatas, persist_dir="./chroma_db"):
    client = chromadb.PersistentClient(path=persist_dir)
    model = get_model()
    print("Encoding chunks into vectors…")
    embeddings = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
    col = client.get_or_create_collection("rag_docs", metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )
    print(f"Stored {len(chunks)} chunks -> {persist_dir}")
    return col
