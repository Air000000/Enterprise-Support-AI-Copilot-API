import json
import os
from typing import Any

from dotenv import load_dotenv

from chroma_search import (
    index_documents,
    search_chroma,
)
from chunk_level_rag import (
    get_client,
    build_rag_prompt,
)


load_dotenv()


# Chroma 默认 L2 distance：
# distance 越小，表示 query embedding 和 chunk embedding 越接近。
# 这个阈值需要根据你的实际检索结果调参。
SOURCE_DISTANCE_THRESHOLD = 0.8


def convert_chroma_hits_to_chunks(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    把 Chroma 返回的 hits 转成 RAG 后续可用的 chunk 格式。

    标准 L2 版本只保留 distance：
    - distance 越小，越相关
    - 不额外造 relevance / similarity
    """
    chunks = []

    for hit in hits:
        chunks.append(
            {
                "chunk_id": hit["chunk_id"],
                "document_id": hit["document_id"],
                "title": hit["title"],
                "content": hit["content"],
                "start": hit["start"],
                "end": hit["end"],
                "distance": hit["distance"],
            }
        )

    return chunks


def filter_chunks_by_distance(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    用 L2 distance 过滤弱相关 chunks。
    """
    return [
        chunk
        for chunk in chunks
        if chunk["distance"] <= SOURCE_DISTANCE_THRESHOLD
    ]


def format_context_from_chroma_chunks(chunks: list[dict[str, Any]]) -> str:
    """
    把 Chroma 检索出来的 chunks 格式化成 LLM 可读的上下文。

    注意：
    这里展示的是 distance，不是 score。
    """
    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"""[Source {index}]
document_id: {chunk["document_id"]}
title: {chunk["title"]}
chunk_id: {chunk["chunk_id"]}
distance: {chunk["distance"]:.4f}
content:
{chunk["content"]}
"""
        )

    return "\n".join(context_parts)


def generate_answer(question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    """
    使用 Qwen 根据 Chroma 检索到的 chunks 生成 RAG answer。
    """
    client = get_client()
    model = os.getenv("DASHSCOPE_MODEL", "qwen3.5-plus")

    context = format_context_from_chroma_chunks(retrieved_chunks)
    prompt = build_rag_prompt(question, context)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful RAG assistant. "
                    "Answer only based on the provided context. "
                    "If the context is insufficient, say you do not have enough evidence."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    return completion.choices[0].message.content


def answer_with_chroma_rag(question: str, top_k: int = 4) -> dict[str, Any]:
    """
    Chroma 默认 L2 版 RAG 主流程。

    1. 使用 Chroma 检索 top_k chunks
    2. 根据 L2 distance 过滤弱相关 chunks
    3. 如果没有足够资料，拒答
    4. 否则调用 Qwen 生成答案
    5. 返回 answer + sources
    """
    hits = search_chroma(question, top_k=top_k)
    retrieved_chunks = convert_chroma_hits_to_chunks(hits)

    top_distance = retrieved_chunks[0]["distance"] if retrieved_chunks else None

    filtered_chunks = filter_chunks_by_distance(retrieved_chunks)

    sources = [
        {
            "document_id": chunk["document_id"],
            "title": chunk["title"],
            "chunk_id": chunk["chunk_id"],
            "content": chunk["content"],
            "distance": round(chunk["distance"], 4),
            "start": chunk["start"],
            "end": chunk["end"],
        }
        for chunk in filtered_chunks
    ]

    if not filtered_chunks:
        return {
            "question": question,
            "answer": "我在已提供资料中没有找到足够依据。",
            "sources": sources,
            "retrieval_status": "low_confidence",
            "top_distance": round(top_distance, 4) if top_distance is not None else None,
        }

    answer = generate_answer(question, filtered_chunks)

    if "没有找到足够依据" in answer:
        retrieval_status = "insufficient_context"
    else:
        retrieval_status = "ok"

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieval_status": retrieval_status,
        "top_distance": round(top_distance, 4) if top_distance is not None else None,
    }


def run_query(question: str) -> None:
    result = answer_with_chroma_rag(question, top_k=4)

    print("\n" + "=" * 100)
    print(f"问题：{question}")
    print("=" * 100)

    print("\n回答：")
    print(result["answer"])

    print("\n检索状态：")
    print(result["retrieval_status"])

    print("\nTop Distance：")
    print(result["top_distance"])

    print("\nSources：")
    for source in result["sources"]:
        print("-" * 80)
        print(f"document_id: {source['document_id']}")
        print(f"title: {source['title']}")
        print(f"chunk_id: {source['chunk_id']}")
        print(f"distance: {source['distance']}")
        print(f"range: {source['start']} - {source['end']}")
        print(f"content: {source['content']}")

    print("\n完整 JSON：")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    """
    第一次运行先 reset=True，确保 Chroma 里是最新 chunks。
    后面如果文档和切块逻辑没变，可以改成 reset=False。
    """
    index_documents(reset=True)

    test_questions = [
        "RAG 为什么要切块？",
        "怎么保存 Todo 到 SQLite 数据库？",
        "FastAPI 里请求体和响应模型怎么定义？",
        "为什么资料库没有答案时应该拒答？",
        "Kubernetes 是什么？",
    ]

    for question in test_questions:
        run_query(question)


if __name__ == "__main__":
    main()