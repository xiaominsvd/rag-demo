"""Step 5: 增强生成（generation）

把检索到的资料拼进 prompt，再交给 LLM 生成答案。
没有 OPENAI_API_KEY 时降级为"抽取式"：直接展示最相关的资料。
"""
import os

PROMPT_TEMPLATE = """你是一个知识库问答助手。请只根据下面提供的参考资料回答问题。
如果资料中没有答案，请直接说"资料中没有相关信息"，不要编造。

参考资料：
{context}

问题：{question}

回答："""


def build_prompt(question, retrieved):
    context = "\n\n---\n\n".join(
        f"[资料{i + 1}｜来源：{m.get('source', '?')}]\n{d}"
        for i, (d, m, _) in enumerate(retrieved)
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
