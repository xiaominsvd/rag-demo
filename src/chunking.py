"""Step 2: 文本切块（chunking）

为什么切块？
- 向量模型一次能处理的文本长度有限
- 检索时我们想要的是"最相关的段落"，而不是整篇文档
- 块太大 -> 噪声多；块太小 -> 语义不完整。400~800 字是常见起点。
"""
import re


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60):
    """先按段落切，再把段落拼成固定大小的块，块之间保留一点重叠。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = (current + "\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # 单个段落超长时硬切，并保留 overlap 避免语义断裂
            while len(para) > chunk_size:
                chunks.append(para[:chunk_size])
                para = para[chunk_size - overlap:]
            current = para
    if current:
        chunks.append(current)
    return chunks


if __name__ == "__main__":
    demo = "第一段。" * 100 + "\n\n" + "第二段。" * 100
    for i, c in enumerate(chunk_text(demo)):
        print(f"--- chunk {i}（{len(c)} 字）---\n{c[:60]}…\n")
