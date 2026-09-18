"""Compare three retrieval methods: pure vector vs pure BM25 vs hybrid

Run: python scripts/compare.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from retrieve import retrieve as vector_retrieve
from hybrid import bm25_retrieve, hybrid_retrieve

QUERIES = [
    "ANN",                                                     # abbreviation: BM25 shines
    "Which vector database should I pick for a prototype?",    # paraphrased: vectors shine
    "What's the difference between RAG and fine-tuning?",      # both do well
]

METHODS = [
    ("Pure vector search", vector_retrieve, "smaller distance is better"),
    ("Pure BM25 search", bm25_retrieve, "higher score is better"),
    ("Hybrid search", hybrid_retrieve, "higher RRF score is better"),
]

for q in QUERIES:
    print(f"\n{'=' * 24}\nQuestion: {q}\n{'=' * 24}")
    for name, fn, hint in METHODS:
        print(f"\n-- {name} ({hint}) --")
        for cid, doc, meta, score in fn(q, top_k=3):
            snippet = doc[:45].replace("\n", " ")
            print(f"  [{cid} | {meta['source']} | {score:.3f}] {snippet}…")
