"""对比三种检索：纯向量 vs 纯 BM25 vs 混合

运行：python scripts/compare.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from retrieve import retrieve as vector_retrieve
from hybrid import bm25_retrieve, hybrid_retrieve

QUERIES = [
    "ANN",                            # 缩写词：BM25 擅长
    "做原型应该选哪个向量数据库？",     # 语义改写：向量擅长
    "RAG和微调有什么区别？",            # 两边都不错
]

METHODS = [
    ("纯向量检索", vector_retrieve, "距离越小越好"),
    ("纯BM25检索", bm25_retrieve, "分数越大越好"),
    ("混合检索", hybrid_retrieve, "RRF分数越大越好"),
]

for q in QUERIES:
    print(f"\n{'=' * 24}\n问题：{q}\n{'=' * 24}")
    for name, fn, hint in METHODS:
        print(f"\n-- {name}（{hint}）--")
        for cid, doc, meta, score in fn(q, top_k=3):
            snippet = doc[:45].replace("\n", " ")
            print(f"  [{cid}｜{meta['source']}｜{score:.3f}] {snippet}…")
