from __future__ import annotations

from pathlib import Path

import chromadb

from experiments.rag_local.build_rag_index import embed_texts
from experiments.rag_local.document_loader import load_documents
from experiments.rag_local.text_splitter import (
    Chunk,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    split_documents,
)

# 学习点：
# CHROMA_DIR 是 Chroma 的本地持久化目录。
# 它相当于“本地向量数据库文件夹”。
#
# JSON index 是一个文件：
#   experiments/index/rag_index.json
#
# Chroma index 是一个目录：
#   experiments/chroma_db/
#
# 后面你重启 Python，Chroma 仍然能从这个目录恢复 collection。
DEFAULT_DOCS_DIR = Path("experiments/docs") 
DEFAULT_CHROMA_DIR = Path("experiments/chroma_db") 
DEFAULT_COLLECTION_NAME = "local_doc_rag"   


def get_chroma_client(chroma_dir: str | Path = DEFAULT_CHROMA_DIR) -> chromadb.PersistentClient:
    """
    创建本地持久化 Chroma client。

    学习点：
    PersistentClient 会把向量库保存到磁盘。
    这和普通内存 demo 不一样。

    内存版：
        程序结束，数据消失。

    持久化版：
        程序结束，数据还在 experiments/chroma_db/。
    """
    chroma_path = Path(chroma_dir)
    chroma_path.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(path=str(chroma_path))


def reset_collection(
    client: chromadb.PersistentClient,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """
    删除旧 collection，然后创建新的 collection。

    学习点：
    开发阶段最容易遇到的问题是重复 add 同一批 chunk_id。

    例如你第一次 add：
        doc_rag_notes_chunk_0

    第二次重跑脚本又 add：
        doc_rag_notes_chunk_0

    Chroma 会认为 id 重复。

    所以 v0.1 阶段我们先采用“重建索引”策略：
        删除旧 collection
        创建新 collection
        重新写入所有 chunks

    真实系统后面会升级成：
        checksum 去重
        document version
        upsert
        delete old chunks
        re-index
    """
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted old collection: {collection_name}")
    except Exception:
        # 学习点：
        # 第一次运行时 collection 不存在，删除失败是正常情况。
        # 这里不让脚本因为“旧 collection 不存在”而中断。
        print(f"No existing collection to delete: {collection_name}")

    # 学习点：
    # embedding_function=None 表示：
    # Chroma 不负责帮我们调用 embedding 模型。
    #
    # 我们自己用 DashScope 生成 embeddings，
    # 然后通过 collection.add(..., embeddings=...) 写进去。
    collection = client.create_collection(
        name=collection_name,
        embedding_function=None,
        metadata={
            "description": "Local document RAG collection built from experiments/docs",
        },
    )

    return collection


def chunk_to_chroma_document(chunk: Chunk) -> str:
    """
    构造存入 Chroma 的 document 文本。

    学习点：
    Chroma 里通常会同时存：
        1. embedding：用于向量检索
        2. document：用于返回原文
        3. metadata：用于过滤和 citation

    document 字段最好保存 chunk 原文，而不是保存 embedding 输入文本。

    因为用户最终看到 sources preview 时，
    更希望看到原始 chunk 内容，而不是额外拼接过的文本。
    """
    return chunk.content


def chunk_to_embedding_text(chunk: Chunk) -> str:
    """
    构造送去 embedding 的文本。

    学习点：
    embedding 用的文本可以和最终返回的 document 不完全一样。

    这里我们把 title 拼进去，是为了增强短 chunk 的语义。
    例如：
        content = "默认地址是 /docs。"

    单独 embedding 这句话，语义不够清楚。
    加上 title：
        fastapi_notes

        默认地址是 /docs。

    模型更容易知道它和 FastAPI 文档相关。
    """
    return f"{chunk.title}\n\n{chunk.content}"


def chunk_to_metadata(chunk: Chunk) -> dict:
    """
    构造 Chroma metadata。

    学习点：
    metadata 是后面工程化 RAG 的关键。

    它可以用于：
        1. sources 引用
        2. tenant_id 权限过滤
        3. document_id 过滤
        4. eval 命中判断
        5. 删除文档时定位旧 chunks

    当前 v0.1 先保存最小必要字段。
    """
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "title": chunk.title,
        "source_path": chunk.source_path,
        "chunk_index": chunk.chunk_index,
        "tenant_id": chunk.tenant_id,
        "category": chunk.category,
    }


def build_chroma_index(
    docs_dir: str | Path = DEFAULT_DOCS_DIR,
    chroma_dir: str | Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chunk_size: int = 500,
    overlap: int = 100,
) -> None:
    """
    构建 local-doc Chroma index。

    学习点：
    这个函数是你当前 RAG pipeline 的“入库流程”。

    它对应真实知识库系统里的：
        文档解析任务
        chunking 任务
        embedding 任务
        vector store 写入任务
    """
    print("Loading documents...")
    documents = load_documents(docs_dir)
    print(f"Loaded documents: {len(documents)}")

    print("Splitting documents...")
    chunks = split_documents(
        documents=documents,
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_CHUNK_OVERLAP,
        min_chunk_size=MIN_CHUNK_SIZE,
    )
    print(f"Generated chunks: {len(chunks)}")

    if not chunks:
        raise RuntimeError("No chunks generated. Please check your docs directory.")

    print("Embedding chunks...")
    embedding_texts = [chunk_to_embedding_text(chunk) for chunk in chunks]  
    embeddings = embed_texts(embedding_texts)
    print(f"Generated embeddings: {len(embeddings)}")

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: chunks={len(chunks)}, embeddings={len(embeddings)}"
        )

    print("Creating Chroma client...")
    client = get_chroma_client(chroma_dir)

    print("Resetting collection...")
    collection = reset_collection(
        client=client,
        collection_name=collection_name,
    )

    ids = [chunk.chunk_id for chunk in chunks]  
    documents_for_chroma = [chunk_to_chroma_document(chunk) for chunk in chunks]
    metadatas = [chunk_to_metadata(chunk) for chunk in chunks]

    print("Adding chunks to Chroma...")

    # 学习点：
    # collection.add 是真正“入库”的地方。
    #
    # ids:
    #   每个 chunk 的唯一 ID。
    #
    # documents:
    #   chunk 原文，后面检索结果会返回它。
    #
    # embeddings:
    #   chunk 的语义向量，Chroma 用它做相似度检索。
    #
    # metadatas:
    #   来源信息，后面用于 citations、filter、eval。
    collection.add(
        ids=ids,
        documents=documents_for_chroma,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print("Chroma index build completed.")
    print(f"Collection name: {collection_name}")
    print(f"Persist dir:     {Path(chroma_dir)}")
    print(f"Added chunks:    {len(chunks)}")

    # 学习点：
    # count() 可以快速确认 collection 里到底有多少条记录。
    print(f"Collection count: {collection.count()}")

    print("\nFirst chunk summary:")
    first_chunk = chunks[0]
    print(f"chunk_id:     {first_chunk.chunk_id}")
    print(f"document_id:  {first_chunk.document_id}")
    print(f"title:        {first_chunk.title}")
    print(f"source_path:  {first_chunk.source_path}")
    print(f"tenant_id:    {first_chunk.tenant_id}")
    print(f"category:     {first_chunk.category}")
    print(f"content_len:  {len(first_chunk.content)}")

def main() -> None:
    build_chroma_index()


if __name__ == "__main__":
    main()