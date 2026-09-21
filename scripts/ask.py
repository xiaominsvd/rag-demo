"""Ask in one command: python scripts/ask.py "your question"

For real LLM generation, set the env var first:
    export OPENAI_API_KEY="sk-..."   # never paste your key in chat
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from retrieve import retrieve
from answer import build_prompt, generate_answer

question = sys.argv[1] if len(sys.argv) > 1 else "What's the difference between RAG and fine-tuning?"
hits = retrieve(question, top_k=3)

print("🔎 Retrieved sources:")
for i, (cid, doc, meta, dist) in enumerate(hits, 1):
    print(f"  [{i}] {cid} | {meta['source']} (distance {dist:.3f}): {doc[:70]}…")
print()

answer = generate_answer(build_prompt(question, hits))
if answer:
    print("💡 Answer:\n" + answer)
else:
    print("⚠️ OPENAI_API_KEY not detected, falling back to extractive answer (showing the most relevant source):\n")
    print(hits[0][1])
