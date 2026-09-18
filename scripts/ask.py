"""一键提问：python scripts/ask.py "你的问题"

需要真实 LLM 生成时，先设置环境变量：
    export OPENAI_API_KEY="sk-..."   # 不要把 key 发在聊天里
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from retrieve import retrieve
from answer import build_prompt, generate_answer

question = sys.argv[1] if len(sys.argv) > 1 else "RAG 和微调有什么区别？"
hits = retrieve(question, top_k=3)

print("🔎 检索到的资料：")
for i, (cid, doc, meta, dist) in enumerate(hits, 1):
    print(f"  [{i}] {cid}｜{meta['source']}（距离 {dist:.3f}）：{doc[:70]}…")
print()

answer = generate_answer(build_prompt(question, hits))
if answer:
    print("💡 回答：\n" + answer)
else:
    print("⚠️ 未检测到 OPENAI_API_KEY，降级为抽取式回答（展示最相关的资料）：\n")
    print(hits[0][0])
