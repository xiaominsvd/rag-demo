"""Show the prompt context each retrieval method would feed to the LLM.

Same chunks, three scoring methods -> three different contexts.
Run: python scripts/show_contexts.py "your question"
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from retrieve import retrieve as vector_retrieve
from hybrid import bm25_retrieve, hybrid_retrieve
from answer import build_prompt

question = sys.argv[1] if len(sys.argv) > 1 else "做原型应该选哪个向量数据库？"

METHODS = [
    ("Pure vector", vector_retrieve),
    ("Pure BM25", bm25_retrieve),
    ("Hybrid (RRF)", hybrid_retrieve),
]

for name, fn in METHODS:
    hits = fn(question, top_k=3)
    print(f"\n{'=' * 22}\n{name}: context fed to the LLM\n{'=' * 22}")
    print(build_prompt(question, hits))
