from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_runtime.build_rag_index import DEFAULT_INDEX_PATH, embed_texts


@dataclass
class SearchResult:
    """
    一条检索结果。

    学习点：
    搜索结果不能只返回 content。
    后续 RAG answer 需要 sources，所以这里提前保留：
    - chunk_id
    - document_id
    - title
    - source_path
    - score

    score 表示 query 和 chunk 的语义相似度。
    """

    chunk_id: str
    document_id: str
    title: str
    source_path: str
    chunk_index: int
    content: str
    score: float


def load_index(index_path: str | Path = DEFAULT_INDEX_PATH) -> list[dict[str, Any]]:
    """
    读取 build_rag_index.py 生成的 JSON index。

    学习点：
    当前阶段我们用 JSON 模拟一个最小向量库。
    后面换成 Chroma 时，这一步会变成从 Chroma collection 里查询。
    """
    path = Path(index_path) 

    if not path.exists():
        raise FileNotFoundError(
            f"Index file not found: {path}. "
            "Please run: python -m rag_runtime.build_rag_index"
        )   

    raw_text = path.read_text(encoding="utf-8")
    data = json.loads(raw_text)

    if not isinstance(data, list):
        raise ValueError(f"Invalid index format: expected list, got {type(data)}")

    return data


def dot_product(a: list[float], b: list[float]) -> float:
    """
    计算两个向量的点积。

    学习点：
    点积可以粗略理解为两个向量“朝向是否一致”。
    如果两个向量方向越接近，点积通常越大。
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} != {len(b)}")

    return sum(x * y for x, y in zip(a, b))


def vector_norm(vector: list[float]) -> float:
    """
    计算向量长度。

    学习点：
    cosine similarity 比较的是“方向相似”，不是“长度相似”。
    所以要用向量长度做归一化。
    """
    return math.sqrt(sum(x * x for x in vector))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    计算余弦相似度。

    学习点：
    embedding 检索常用 cosine similarity，因为我们更关心语义方向是否接近。

    公式：
        cos(theta) = dot(a, b) / (norm(a) * norm(b))

    直觉：
    - 1.0：方向几乎完全相同，语义非常接近
    - 0.0：方向接近垂直，语义关系弱
    - -1.0：方向相反，语义差异大

    实际 embedding 模型里，分数不一定严格按这个直觉解释，
    但排序通常有效。
    """
    norm_a = vector_norm(a)
    norm_b = vector_norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product(a, b) / (norm_a * norm_b)


def search(
    query: str,
    top_k: int = 3,
    index_path: str | Path = DEFAULT_INDEX_PATH,
) -> list[SearchResult]:
    """
    在本地 JSON index 中搜索最相关的 chunks。

    学习点：
    这是 RAG 的 retrieval 阶段。

    retrieval 不负责生成答案。
    它只负责回答：
    “哪些 chunk 最可能包含回答这个问题所需的信息？”
    """
    if not query.strip():
        raise ValueError("query must not be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    index_items = load_index(index_path)

    # 学习点：
    # 查询问题必须使用和文档 chunk 相同的 embedding 模型。
    # 否则 query 向量和 chunk 向量不在同一个语义空间里，无法比较距离。
    query_embedding = embed_texts([query])[0]

    results: list[SearchResult] = []

    for item in index_items:
        chunk_embedding = item["embedding"]
        score = cosine_similarity(query_embedding, chunk_embedding)

        result = SearchResult(
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            title=item["title"],
            source_path=item["source_path"],
            chunk_index=item["chunk_index"],
            content=item["content"],
            score=score,
        )
        results.append(result)

    results.sort(key=lambda result: result.score, reverse=True)

    return results[:top_k]


def print_search_results(query: str, results: list[SearchResult]) -> None:
    print("=" * 100)
    print(f"Query: {query}")
    print("-" * 100)

    for rank, result in enumerate(results, start=1):
        preview = result.content[:160].replace("\n", " ")

        print(f"[{rank}] score={result.score:.4f}")
        print(f"chunk_id:    {result.chunk_id}")
        print(f"document_id: {result.document_id}")
        print(f"title:       {result.title}")
        print(f"source_path: {result.source_path}")
        print(f"preview:     {preview}...")
        print("-" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search local RAG JSON index.")
    parser.add_argument(
        "query",
        nargs="?",
        help="Question to search. If omitted, built-in test questions will be used.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to return.",
    )
    args = parser.parse_args()

    if args.query:
        queries = [args.query]
    else:
        # 学习点：
        # 这些问题是最小手动验收集。
        # 后面 eval_retrieval.py 会把它们升级成正式 JSONL 测试集。
        queries = [
            "FastAPI 如何定义请求体？",
            "SQLModel 怎么把 Todo 保存到数据库？",
            "RAG 为什么需要 chunk？",
            "Embedding 是干什么的？",
            "Docker volume 有什么用？",
        ]

    for query in queries:
        results = search(query=query, top_k=args.top_k)
        print_search_results(query, results)


if __name__ == "__main__":
    main()