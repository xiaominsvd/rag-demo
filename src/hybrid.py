"""Step 6: Hybrid retrieval

Vector search understands semantics but is insensitive to exact terms; BM25
keyword search is precise but understands no semantics.
Hybrid retrieval = take top_k from each side, union + dedupe, then fuse
rankings with RRF.

RRF (Reciprocal Rank Fusion): score = Σ 1 / (k + rank)
- Uses only ranks, not raw scores, so vector distances and BM25 scores (different scales) fuse directly
- Chunks hit by both sides accumulate score and naturally float to the top
"""
import chromadb
import jieba
from rank_bm25 import BM25Okapi
from retrieve import retrieve as vector_retrieve


def _tokenize(text):
    # Segment text first so BM25 can count term frequencies
    return list(jieba.cut(text))


def _load_corpus(persist_dir="./chroma_db"):
    col = chromadb.PersistentClient(path=persist_dir).get_collection("rag_docs")
    data = col.get()
    # Sort by chunk-id for a stable order
    return sorted(
        zip(data["ids"], data["documents"], data["metadatas"]),
        key=lambda x: int(x[0].split("-")[1]),
    )


def bm25_retrieve(query, top_k=3, persist_dir="./chroma_db"):
    """Pure BM25 keyword search; returns (chunk id, document, metadata, BM25 score) — higher score means more relevant"""
    items = _load_corpus(persist_dir)
    bm25 = BM25Okapi([_tokenize(doc) for _, doc, _ in items])
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(items, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(cid, doc, meta, float(s)) for (cid, doc, meta), s in ranked]


def hybrid_retrieve(query, top_k=3, persist_dir="./chroma_db", rrf_k=60):
    """Hybrid search: vector top_k ∪ BM25 top_k, fused with RRF, keep top_k"""
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
        print(f"[{cid} | {meta['source']}] RRF={score:.4f}\n{doc[:80]}…\n")
