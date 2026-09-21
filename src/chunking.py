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


def chunk_markdown(text: str, chunk_size: int = 400, overlap: int = 60):
    """Split markdown by headers so a header never separates from its body.

    Each section becomes "header path + body" (e.g. "Vector Databases >
    Selection Advice" followed by the section text). Sections longer than
    chunk_size fall back to paragraph packing via chunk_text, with the
    header path prepended to every piece so no chunk loses its context.
    """
    parts = re.split(r"(?m)^(#{1,6}\s+.*)$", text)
    sections = []  # (level, title, body)
    if parts[0].strip():
        sections.append((0, "", parts[0].strip()))
    for i in range(1, len(parts), 2):
        hashes, title = parts[i].strip().split(None, 1)
        body = parts[i + 1].strip()
        sections.append((len(hashes), title, body))

    chunks, stack = [], []  # stack tracks [(level, title)] for the path
    for level, title, body in sections:
        while stack and stack[-1][0] >= level:
            stack.pop()
        if level:
            stack.append((level, title))
        prefix = " > ".join(t for _, t in stack)
        header = f"{prefix}\n" if prefix else ""
        section_text = header + body
        if len(section_text) <= chunk_size:
            if section_text.strip():
                chunks.append(section_text)
        else:
            budget = max(chunk_size - len(header), 100)
            for piece in chunk_text(body, chunk_size=budget, overlap=overlap):
                chunks.append(header + piece)
    return chunks


if __name__ == "__main__":
    demo = "Paragraph one. " * 100 + "\n\n" + "Paragraph two. " * 100
    for i, c in enumerate(chunk_text(demo)):
        print(f"--- chunk {i} ({len(c)} chars) ---\n{c[:60]}…\n")
