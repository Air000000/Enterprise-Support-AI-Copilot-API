import os
import json
from typing import Any

import chromadb
from dotenv import load_dotenv

from chunk_level_rag import (
    long_documents,
    build_chunks,
    embed_texts,
)


load_dotenv()


CHROMA_PATH = "data/chroma" 
COLLECTION_NAME = "rag_notes"  


def get_chroma_collection():   
    """
    创建或加载 Chroma collection。

    PersistentClient 会把数据保存到本地目录。
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)    #

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "RAG learning notes collection"},
    )   # collection 是一个向量数据库接口对象，后续可以对它进行增删改查等操作。

    return collection


def reset_collection():
    """
    删除并重建 collection，方便实验时重新建库。
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)    # PersistentClient 会把数据保存到本地目录。

    try:
        client.delete_collection(COLLECTION_NAME)   
    except Exception:
        pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "RAG learning notes collection"},
    )   # collection 是一个向量数据库接口对象，后续可以对它进行增删改查等操作。


def index_documents(reset: bool = False) -> None:
    """
    把 long_documents 切成 chunks，生成 embeddings，然后写入 Chroma。
    """
    if reset:
        collection = reset_collection()
    else:
        collection = get_chroma_collection()

    chunks = build_chunks(long_documents)   

    print(f"准备写入 {len(chunks)} 个 chunks 到 Chroma...")

    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = [chunk["content"] for chunk in chunks]
    metadatas = [
        {
            "document_id": chunk["document_id"],
            "title": chunk["title"],
            "start": chunk["start"],
            "end": chunk["end"],
        }
        for chunk in chunks
    ]

    embeddings = embed_texts([
        f"{chunk['title']}\n{chunk['content']}"
        for chunk in chunks
    ])

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )   # 把 chunks 的 id、文本内容、embedding 和元数据一起写入 Chroma。之后就可以通过 embedding 来检索相关的 chunks 了。

    print("Chroma indexing 完成。")
    print(f"Collection count: {collection.count()}")


def search_chroma(question: str, top_k: int = 4) -> list[dict[str, Any]]:
    """
    使用 Chroma 做向量检索。
    """
    collection = get_chroma_collection()    

    question_embedding = embed_texts([question])[0] 

    result = collection.query(
        query_embeddings=[question_embedding],  # 去 collection 里和已存的所有 chunk embeddings 比较距离
        n_results=top_k,    
        include=["documents", "metadatas", "distances"],
    )   #

    hits = []  
    # Chroma 的 query 接口返回的结果是一个字典
    # 里面的 "ids"、"documents"、"metadatas" 和 "distances" 都是列表
    # 因为可以同时查询多个 embedding。这里我们只查询了一个问题，所以取第一个元素。
    ids = result["ids"][0]  
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    for chunk_id, content, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):  
        hits.append(
            {
                "chunk_id": chunk_id,
                "document_id": metadata["document_id"],
                "title": metadata["title"],
                "content": content,
                "start": metadata["start"],
                "end": metadata["end"],
                "distance": distance,
            }
        )

    return hits


def run_query(question: str) -> None:
    hits = search_chroma(question, top_k=4)

    print("\n" + "=" * 100)
    print(f"问题：{question}")
    print("=" * 100)

    for index, hit in enumerate(hits, start=1):
        print(f"Top {index}")
        print(f"chunk_id: {hit['chunk_id']}")
        print(f"title: {hit['title']}")
        print(f"distance: {hit['distance']}")
        print(f"range: {hit['start']} - {hit['end']}")
        print(f"content: {hit['content']}")
        print("-" * 80)


def main() -> None:
    index_documents(reset=True)

    test_questions = [
        "RAG 为什么要切块？",
        "怎么保存 Todo 到 SQLite 数据库？",
        "FastAPI 里请求体和响应模型怎么定义？",
        "Kubernetes 是什么？",
    ]

    for question in test_questions:
        run_query(question)


if __name__ == "__main__":
    main()