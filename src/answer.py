"""Step 5: Augmented generation

Stuff the retrieved sources into the prompt and let the LLM generate an answer.
Without OPENAI_API_KEY, falls back to "extractive" mode: just show the most
relevant sources directly.
"""
import os

PROMPT_TEMPLATE = """You are a knowledge-base Q&A assistant. Answer the question using ONLY the reference material provided below.
If the material contains no answer, say "No relevant information in the sources." Do not fabricate.

Reference material:
{context}

Question: {question}

Answer:"""


def build_prompt(question, retrieved):
    context = "\n\n---\n\n".join(
        f"[Source {i + 1} | file: {m.get('source', '?')}]\n{d}"
        for i, (cid, d, m, _) in enumerate(retrieved)
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate_answer(prompt):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content
