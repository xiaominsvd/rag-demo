# RAG Demo：本地文档问答助手

一个最小但完整的 RAG 系统：针对 `docs/` 里的文档提问，
系统检索相关段落，再生成带依据的回答。

```
用户问题
  │
  ▼  向量化
检索：Chroma 里找 top_k 个最相似的 chunk
  │
  ▼  拼进 prompt
生成：LLM 基于资料回答（无 key 时降级为抽取式展示）
```

## 7 个步骤

- **Step 0 环境准备**：`pip install -r requirements.txt`（Python ≥ 3.9）
- **Step 1 准备文档**：`docs/` 下已有 3 份中文示例文档，可换成你自己的 `.md`
- **Step 2 文本切块**：看 `src/chunking.py`，运行 `python src/chunking.py` 看切块效果
- **Step 3 向量化 + 入库**：看 `src/store.py`
- **Step 4 检索**：看 `src/retrieve.py`
- **Step 5 生成**：看 `src/answer.py`
- 一键跑通：`python scripts/build_index.py` 然后 `python scripts/ask.py "RAG和微调有什么区别？"`

## Step 6 优化方向（学完基础后）

1. 混合检索（向量 + BM25）
2. rerank 精排
3. 按标题语义切块、给 chunk 加摘要
4. 评测集：准备 20 个问答对，量化每次改动的效果

## FAQ

- embedding 模型第一次运行时会自动从 HuggingFace 下载（约 100MB）。
- 想要真实 LLM 生成：`export OPENAI_API_KEY="sk-..."` 后再运行 `ask.py`。
