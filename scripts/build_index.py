"""Build the index in one command: read docs/ -> chunk -> embed -> store

Run: python scripts/build_index.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathlib import Path
from chunking import chunk_text
from store import build_collection

docs_dir = Path(__file__).resolve().parent.parent / "docs"
chunks, metas = [], []
for f in sorted(docs_dir.glob("*.md")):
    for c in chunk_text(f.read_text(encoding="utf-8")):
        chunks.append(c)
        metas.append({"source": f.name})

print(f"Split into {len(chunks)} chunks")
build_collection(chunks, metas)
print("Index built ✅")
