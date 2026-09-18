"""Step 2: Chunking text

Why chunk?
- Embedding models can only handle limited text length at once
- At retrieval time we want "the most relevant passages", not whole documents
- Chunks too big -> noisy; chunks too small -> incomplete meaning. 400-800 chars is a common starting point.
"""
import re


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60):
    """Split by paragraph first, then pack paragraphs into fixed-size chunks with a little overlap between them."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = (current + "\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # If a single paragraph exceeds chunk_size, hard-split it, keeping overlap to avoid breaking meaning mid-thought
            while len(para) > chunk_size:
                chunks.append(para[:chunk_size])
                para = para[chunk_size - overlap:]
            current = para
    if current:
        chunks.append(current)
    return chunks


if __name__ == "__main__":
    demo = "Paragraph one. " * 100 + "\n\n" + "Paragraph two. " * 100
    for i, c in enumerate(chunk_text(demo)):
        print(f"--- chunk {i} ({len(c)} chars) ---\n{c[:60]}…\n")
