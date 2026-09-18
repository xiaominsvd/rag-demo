"""Step 4: 检索（retrieval）

流程：用户问题 -> 同一个 embedding 模型 -> 问题向量
     -> 在 Chroma 里找向量最接近的 top_k 个 chunk
"""
import chromadb
from store import get_model


def retrieve(query, top_k=3, persist_dir="./chroma_db"):
    client = chromadb.PersistentClient(path=persist_dir)
    col = client.get_collection("rag_docs")
    model = get_model()
    q_emb = model.encode([query], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q_emb, n_results=top_k)
    # 返回 (chunk id, 文档内容, 元数据, 距离) —— 距离越小越相关
    # 多返回 id，是为了 Step 6 做混合检索时能按 id 融合去重
    return list(zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]))


if __name__ == "__main__":
    for cid, doc, meta, dist in retrieve("什么是 RAG？"):
        print(f"[{cid}｜{meta['source']}] 距离={dist:.3f}\n{doc[:120]}…\n")
