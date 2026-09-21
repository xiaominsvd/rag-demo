"""Step 7 (optional): cross-encoder reranking.

Two-stage pipeline: cheap retrieval (hybrid) fetches top-k candidates,
then the cross-encoder scores each (query, chunk) pair directly and
keeps the best few. Unlike RRF rank fusion, the cross-encoder actually
reads every query-chunk pair before scoring it.
"""
from sentence_transformers import CrossEncoder

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

_reranker = None


def _get_reranker():
    """Lazy-load the model once and reuse it."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def rerank(query, candidates, top_k=3):
    """Rerank (cid, doc, meta, score) candidates with the cross-encoder.

    Returns the top_k as (cid, doc, meta, rerank_score), highest first.
    """
    ranker = _get_reranker()
    pairs = [(query, doc) for cid, doc, meta, _ in candidates]
    scores = ranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [(cid, doc, meta, float(s)) for (cid, doc, meta, _), s in ranked[:top_k]]


def hybrid_rerank_retrieve(query, fetch_k=10, top_k=3):
    """Hybrid retrieval followed by cross-encoder reranking."""
    from hybrid import hybrid_retrieve

    candidates = hybrid_retrieve(query, top_k=fetch_k)
    return rerank(query, candidates, top_k=top_k)


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "Which vector database should I pick for a prototype?"
    print(f"Question: {q}\n")
    for cid, doc, meta, score in hybrid_rerank_retrieve(q):
        snippet = doc[:60].replace("\n", " ")
        print(f"  [{cid} | {meta['source']} | {score:.3f}] {snippet}...")
