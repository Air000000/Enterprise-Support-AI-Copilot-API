from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from experiments.rag_local.build_chroma_index import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION_NAME,
    embed_texts,
    get_chroma_client,
)   


@dataclass
class ChromaSearchResult:
    """
    Chroma 检索结果。

    学习点：
    Chroma 返回的原始结果是一个嵌套 dict。
    我们把它转换成 dataclass，是为了让后续 query_rag_chroma.py 更好用。
    """

    chunk_id: str
    document_id: str
    title: str
    source_path: str
    chunk_index: int
    content: str
    distance: float
    tenant_id: str
    category: str


def get_collection():
    """
    获取已经构建好的 Chroma collection。

    学习点：
    这里不是 create_collection，而是 get_collection。

    build_chroma_index.py 负责创建和写入。
    query_chroma.py 负责读取和检索。

    这就是“索引构建”和“在线查询”分离。
    """
    client = get_chroma_client(DEFAULT_CHROMA_DIR)
    return client.get_collection(
        name=DEFAULT_COLLECTION_NAME,
        embedding_function=None,
    )

def build_where_filter(
    tenant_id: str | None = None,
    category: str | None = None,
) -> dict | None:
    if tenant_id and category:
        return {
            "$and": [
                {"tenant_id": tenant_id},
                {"category": category},
            ]
        }

    if tenant_id:
        return {"tenant_id": tenant_id}

    if category:
        return {"category": category}

    return None

def search_chroma(
        query: str, 
        top_k: int = 3,
        tenant_id: str | None = None,
        category: str | None = None,
    ) -> list[ChromaSearchResult]:
    """
    从 Chroma 中检索和 query 最相关的 chunks。

    学习点：
    因为 collection 创建时 embedding_function=None，
    所以 query 时也要由我们自己生成 query_embedding。

    也就是说：
        文档 chunk embedding：我们自己生成
        用户问题 embedding：我们自己生成
        相似度搜索：Chroma 负责
    """
    if not query.strip():
        raise ValueError("query must not be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    where = build_where_filter(tenant_id=tenant_id, category=category)

    collection = get_collection()

    query_embedding = embed_texts([query])[0]

    raw_results: dict[str, Any] = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    results: list[ChromaSearchResult] = []

    ids = raw_results["ids"][0]
    documents = raw_results["documents"][0]
    metadatas = raw_results["metadatas"][0]
    distances = raw_results["distances"][0]

    for chunk_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        result = ChromaSearchResult(
            chunk_id=chunk_id,
            document_id=metadata["document_id"],
            title=metadata["title"],
            source_path=metadata["source_path"],
            chunk_index=metadata["chunk_index"],
            content=document,
            distance=distance,
            tenant_id=metadata["tenant_id"],
            category=metadata["category"],
        )
        results.append(result)

    return results


def print_results(query: str, results: list[ChromaSearchResult]) -> None:
    print("=" * 100)
    print(f"Query: {query}")
    print("-" * 100)

    for rank, result in enumerate(results, start=1):
        preview = result.content[:160].replace("\n", " ")

        print(f"[{rank}] distance={result.distance:.4f}")
        print(f"chunk_id:    {result.chunk_id}")
        print(f"document_id: {result.document_id}")
        print(f"title:       {result.title}")
        print(f"source_path: {result.source_path}")
        print(f"tenant_id:   {result.tenant_id}")
        print(f"category:    {result.category}")
        print(f"preview:     {preview}...")
        print("-" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query local Chroma RAG index.")
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
    parser.add_argument(
        "--tenant-id",
        help="Tenant ID to filter results.",
    )
    parser.add_argument(
        "--category",
        help="Category to filter results.",
    )
    args = parser.parse_args()

    if args.query:
        queries = [args.query]
    else:
        queries = [
            "FastAPI 如何定义请求体？",
            "SQLModel 怎么把 Todo 保存到数据库？",
            "RAG 为什么需要 chunk？",
            "Embedding 是干什么的？",
            "Docker volume 有什么用？",
        ]

    for query in queries:
        results = search_chroma(
            query=query,
            top_k=args.top_k,
            tenant_id=args.tenant_id,
            category=args.category
        )
        print_results(query, results)


if __name__ == "__main__":
    main()