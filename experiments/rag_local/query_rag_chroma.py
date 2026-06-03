from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass

from dotenv import load_dotenv
from openai import OpenAI

from experiments.rag_local.query_chroma import ChromaSearchResult, search_chroma


@dataclass
class Source:
    """
    RAG 答案引用来源。

    学习点：
    RAG 不应该只返回 answer。
    还要返回 sources，让用户知道答案依据来自哪里。

    后面做企业知识库时，sources 会变成很重要的可信度证据。
    """

    chunk_id: str
    document_id: str
    title: str
    source_path: str
    chunk_index: int
    distance: float
    preview: str
    tenant_id: str
    category: str


@dataclass
class RagResponse:
    """
    一次 RAG 问答的结构化结果。

    学习点：
    不要让 RAG 函数只返回字符串。
    工程里更推荐返回结构化对象，方便：
    - API 响应
    - 日志记录
    - eval 评测
    - 前端展示 sources
    """

    question: str
    answer: str
    retrieval_status: str
    top_distance: float | None
    sources: list[Source]


def get_llm_client() -> OpenAI:
    """
    创建 LLM client。

    学习点：
    这里继续使用 OpenAI-compatible API。
    embedding 和 chat 可以共用同一个 provider，也可以后面拆成不同 provider。
    """
    load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    if not api_key:
        raise RuntimeError(
            "Missing DASHSCOPE_API_KEY. Please set it in your .env file."
        )

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def build_context(results: list[ChromaSearchResult]) -> str:
    """
    把检索到的 chunks 拼成 prompt context。

    学习点：
    context 不是随便拼文本。
    每个 chunk 前面加 source 编号，后面模型回答时才能引用来源。

    当前 v0.1 简化成：
        [Source 1]
        title: ...
        chunk_id: ...
        content: ...

    后面可以升级成更严格的 citation 格式。
    """
    context_parts: list[str] = []

    for index, result in enumerate(results, start=1):
        context_part = f"""
        [Source {index}]
        title: {result.title}
        document_id: {result.document_id}
        chunk_id: {result.chunk_id}
        distance: {result.distance:.4f}

        content:
        {result.content}
        """.strip()

        context_parts.append(context_part)

    return "\n\n---\n\n".join(context_parts)


def build_sources(results: list[ChromaSearchResult]) -> list[Source]:
    """
    把 Chroma 检索结果转换成 sources。

    学习点：
    sources 给人看，context 给模型看。
    二者来自同一批 retrieval results，但用途不同。
    """
    sources: list[Source] = []

    for result in results:
        source = Source(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            title=result.title,
            source_path=result.source_path,
            chunk_index=result.chunk_index,
            distance=result.distance,
            preview=result.content[:160].replace("\n", " "),
            tenant_id=result.tenant_id,
            category=result.category,
        )
        sources.append(source)

    return sources


def should_refuse(results: list[ChromaSearchResult], max_distance: float = 0.9) -> bool:
    """
    根据检索距离判断是否拒答。

    学习点：
    Chroma 返回的是 distance，通常越小越相关。

    max_distance 是一个非常粗糙的拒答阈值：
        top distance <= max_distance：认为有足够相关资料
        top distance > max_distance：认为资料相关性不足

    这个阈值不是固定真理，需要后面通过 eval 调。
    当前根据你的测试输出：
        相关问题 top distance 大约 0.41 ~ 0.57
        次相关结果大约 1.0+
    所以先用 0.9 作为 v0.1 阈值。
    """
    if not results:
        return True

    top_distance = results[0].distance
    return top_distance > max_distance


def generate_answer(question: str, context: str) -> str:
    """
    调用 LLM 基于 context 回答。

    学习点：
    RAG prompt 的核心约束是：
    只能根据给定 context 回答。
    如果 context 没有依据，就必须说不知道。

    这可以降低幻觉，但不能彻底消除幻觉。
    后面仍然需要 eval 和人工检查。
    """
    client = get_llm_client()

    model = os.getenv("CHAT_MODEL", "qwen-plus")

    system_prompt = """
你是一个严谨的企业知识库问答助手。

要求：
1. 只能根据用户提供的 Context 回答。
2. 如果 Context 中没有足够依据，回答：“我在已提供资料中没有找到足够依据。”
3. 不要编造 Context 之外的信息。
4. 回答要简洁、准确。
5. 如果使用了某个来源，请在回答中提到对应的 Source 编号，例如：[Source 1]。
""".strip()

    user_prompt = f"""
Question:
{question}

Context:
{context}

请基于 Context 回答 Question。
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content or ""


def ask_rag(
    question: str,
    top_k: int = 3,
    max_distance: float = 0.9,
) -> RagResponse:
    """
    完整 RAG 问答入口。

    流程：
        1. 检索相关 chunks
        2. 判断是否应该拒答
        3. 构造 context
        4. 调用 LLM
        5. 返回 answer + sources
    """
    if not question.strip():
        raise ValueError("question must not be empty")

    results = search_chroma(query=question, top_k=top_k)
    sources = build_sources(results)

    top_distance = results[0].distance if results else None

    if should_refuse(results, max_distance=max_distance):
        return RagResponse(
            question=question,
            answer="我在已提供资料中没有找到足够依据。",
            retrieval_status="refused_low_relevance",
            top_distance=top_distance,
            sources=sources,
        )

    context = build_context(results)
    answer = generate_answer(question=question, context=context)

    return RagResponse(
        question=question,
        answer=answer,
        retrieval_status="ok",
        top_distance=top_distance,
        sources=sources,
    )


def print_rag_response(response: RagResponse) -> None:
    print("=" * 100)
    print(f"Question: {response.question}")
    print("-" * 100)
    print(f"retrieval_status: {response.retrieval_status}")
    print(f"top_distance:     {response.top_distance}")
    print()
    print("Answer:")
    print(response.answer)
    print()
    print("Sources:")

    for index, source in enumerate(response.sources, start=1):
        print(f"[{index}]")
        print(f"  chunk_id:    {source.chunk_id}")
        print(f"  document_id: {source.document_id}")
        print(f"  title:       {source.title}")
        print(f"  source_path: {source.source_path}")
        print(f"  distance:    {source.distance:.4f}")
        print(f"  tenant_id:   {source.tenant_id}")
        print(f"  category:    {source.category}")
        print(f"  preview:     {source.preview}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask local Chroma RAG.")
    parser.add_argument(
        "question",
        nargs="?",
        default="RAG 为什么需要 chunk？",
        help="Question to ask.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.9,
        help="Refuse answer if top distance is greater than this value.",
    )
    args = parser.parse_args()

    response = ask_rag(
        question=args.question,
        top_k=args.top_k,
        max_distance=args.max_distance,
    )
    print_rag_response(response)


if __name__ == "__main__":
    main()