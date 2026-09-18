"""Step 3: 向量化（embedding）+ 存入向量数据库

流程：chunk -> embedding 模型 -> 向量 -> Chroma 持久化存储
"""
import chromadb
from sentence_transformers import SentenceTransformer

# 中文小模型，~100MB，第一次运行时自动下载
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def get_model():
    return SentenceTransformer(MODEL_NAME)


def build_collection(chunks, metadatas, persist_dir="./chroma_db"):
    client = chromadb.PersistentClient(path=persist_dir)
    model = get_model()
    print("正在把 chunk 转成向量…")
    embeddings = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
    col = client.get_or_create_collection("rag_docs", metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )
    print(f"已存入 {len(chunks)} 个 chunk -> {persist_dir}")
    return col
