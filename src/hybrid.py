"""Step 6: 混合检索（hybrid retrieval）

向量检索懂语义但对精确词不敏感，BM25 关键词检索精确但不懂语义。
混合检索 = 两边各取 top_k，取并集去重，再用 RRF 融合排序。

RRF（Reciprocal Rank Fusion）：score = Σ 1 / (k + 排名)
- 只看排名不看原始分数，所以向量距离和 BM25 分数两种不同量纲可以直接融合
- 两边都命中的 chunk 得分叠加，自然排到前面
"""
import chromadb
import jieba
from rank_bm25 import BM25Okapi
from retrieve import retrieve as vector_retrieve


def _tokenize(text):
    # 中文必须先分词，BM25 才能按"词"统计词频
    return list(jieba.cut(text))


def _load_corpus(persist_dir="./chroma_db"):
    col = chromadb.PersistentClient(path=persist_dir).get_collection("rag_docs")
    data = col.get()
    # 按 chunk-id 排序，保证顺序稳定
    return sorted(
        zip(data["ids"], data["documents"], data["metadatas"]),
        key=lambda x: int(x[0].split("-")[1]),
    )


def bm25_retrieve(query, top_k=3, persist_dir="./chroma_db"):
    """纯 BM25 关键词检索，返回 (chunk id, 文档, 元数据, BM25 分数)，分数越大越相关"""
    items = _load_corpus(persist_dir)
    bm25 = BM25Okapi([_tokenize(doc) for _, doc, _ in items])
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(items, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(cid, doc, meta, float(s)) for (cid, doc, meta), s in ranked]


def hybrid_retrieve(query, top_k=3, persist_dir="./chroma_db", rrf_k=60):
    """混合检索：向量 top_k ∪ BM25 top_k，RRF 融合后取前 top_k"""
    vec_hits = vector_retrieve(query, top_k=top_k, persist_dir=persist_dir)
    bm25_hits = bm25_retrieve(query, top_k=top_k, persist_dir=persist_dir)
    fused = {}  # cid -> [doc, meta, rrf_score]
    for rank, (cid, doc, meta, _) in enumerate(vec_hits):
        entry = fused.setdefault(cid, [doc, meta, 0.0])
        entry[2] += 1.0 / (rrf_k + rank + 1)
    for rank, (cid, doc, meta, _) in enumerate(bm25_hits):
        entry = fused.setdefault(cid, [doc, meta, 0.0])
        entry[2] += 1.0 / (rrf_k + rank + 1)
    ranked = sorted(fused.items(), key=lambda x: -x[1][2])[:top_k]
    return [(cid, doc, meta, score) for cid, (doc, meta, score) in ranked]


if __name__ == "__main__":
    for cid, doc, meta, score in hybrid_retrieve("ANN"):
        print(f"[{cid}｜{meta['source']}] RRF={score:.4f}\n{doc[:80]}…\n")
