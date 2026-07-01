from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from rag_runtime.document_loader import load_documents
from rag_runtime.text_splitter import Chunk, split_documents


# 学习点：
# docs_dir 是原始知识来源。
# index_path 是构建好的“检索索引”。
#
# 原始文档和索引文件要分开放：
# - docs/ 里面是人能读的 md/txt
# - index/ 里面是机器检索用的 JSON
DEFAULT_DOCS_DIR = Path("experiments/docs") 
DEFAULT_INDEX_PATH = Path("experiments/index/rag_index.json")
EMBEDDING_BATCH_SIZE = 10

@dataclass
class IndexItem:
    """
    一个 IndexItem = 一个可以被检索的 chunk + 它的 embedding。

    学习点：
    后面 query_index.py 检索时，不是直接搜原始文档，
    而是搜索这些 IndexItem。

    这里一定不能只保存 embedding。
    如果只保存向量，检索命中后你会不知道：
    - 它来自哪篇文档
    - 它是第几个 chunk
    - 原文内容是什么
    - sources 应该怎么返回
    """

    chunk_id: str
    document_id: str
    title: str
    source_path: str
    chunk_index: int
    content: str
    embedding: list[float]


def get_client() -> OpenAI:
    """
    创建 OpenAI client。

    学习点：
    这里虽然用的是 OpenAI SDK，但 base_url 指向 DashScope 兼容接口。
    这叫 OpenAI-compatible API。
    很多国产模型服务都会提供这种兼容模式。

    这样做的好处是：
    - 业务代码不用强绑定某一家 provider
    - 以后可以相对容易切换 OpenAI / DashScope / DeepSeek 等服务
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


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    把多段文本转换成 embedding 向量。
    """
    if not texts:
        return []
    
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        end = start + EMBEDDING_BATCH_SIZE
        batch = texts[start:end]

        response = get_client().embeddings.create(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
            input=batch,
            encoding_format="float",
        )
        
        batch_embeddings = [item.embedding for item in response.data]
        
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings


def chunk_to_embedding_text(chunk: Chunk) -> str:
    """
    构造真正送去 embedding 的文本。

    学习点：
    这里不要只 embedding content，也可以把 title 拼进去。

    原因：
    有些 chunk 内容很短，如果只看 content，语义可能不够清楚。
    加上 title 后，相当于给 chunk 补了一个“小标题”。

    例如 content 是：
    “默认地址是 /docs。”

    只看这句话，模型不知道在说谁。
    但如果加上 title：
    “fastapi_notes\\n默认地址是 /docs。”

    语义就更清晰。
    """
    return f"{chunk.title}\n\n{chunk.content}"


def build_index(chunks: list[Chunk]) -> list[IndexItem]:
    """
    给所有 chunk 生成 embedding，并组装成 IndexItem。

    学习点：
    这是 RAG 的离线阶段。

    离线阶段：
    - 文档读取
    - 文档切块
    - 生成 embedding
    - 保存索引

    在线阶段：
    - 用户提问
    - 问题 embedding
    - 相似度搜索
    - 构造 prompt
    - LLM 回答
    """
    texts = [chunk_to_embedding_text(chunk) for chunk in chunks]    

    print(f"Embedding chunks: {len(texts)}")
    embeddings = embed_texts(texts)

    if len(embeddings) != len(chunks):  
        raise RuntimeError(
            f"Embedding count mismatch: chunks={len(chunks)}, embeddings={len(embeddings)}"
        )

    index_items: list[IndexItem] = []   

    for chunk, embedding in zip(chunks, embeddings):
        item = IndexItem(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            title=chunk.title,
            source_path=chunk.source_path,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            embedding=embedding,
        )
        index_items.append(item)

    return index_items


def save_index(index_items: list[IndexItem], index_path: Path) -> None:
    """
    保存索引到 JSON 文件。

    学习点：
    JSON index 是最简单的向量库替代品。

    它的优点：
    - 透明
    - 好调试
    - 能看见每个 chunk 和 embedding

    它的缺点：
    - 数据量大时加载慢
    - 每次搜索都要全量遍历
    - 没有高效向量索引结构

    所以后面我们才会换 Chroma。
    但现在先用 JSON，非常适合理解 RAG 底层流程。
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)

    data: list[dict[str, Any]] = [asdict(item) for item in index_items]

    index_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    docs_dir = DEFAULT_DOCS_DIR
    index_path = DEFAULT_INDEX_PATH

    print("Loading documents...")
    documents = load_documents(docs_dir)
    print(f"Loaded documents: {len(documents)}")
    
    print("Splitting documents...")
    chunks = split_documents(
        documents=documents,
        chunk_size=500,
        overlap=100,
    )
    print(f"Generated chunks: {len(chunks)}")

    print("Building embedding index...")
    index_items = build_index(chunks)

    print(f"Saving index to: {index_path}")
    save_index(index_items, index_path)

    print("Index build completed.")
    print(f"Index items: {len(index_items)}")

    if index_items:
        first_item = index_items[0]
        print("\nFirst index item summary:")
        print(f"chunk_id:      {first_item.chunk_id}")
        print(f"document_id:   {first_item.document_id}")
        print(f"title:         {first_item.title}")
        print(f"content_len:   {len(first_item.content)}")
        print(f"embedding_dim: {len(first_item.embedding)}")


if __name__ == "__main__":
    main()