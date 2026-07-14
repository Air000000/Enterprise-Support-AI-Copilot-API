import os
import json
import math
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

import re

load_dotenv()


MIN_RETRIEVAL_SCORE = 0.35
SOURCE_SCORE_THRESHOLD = 0.35


long_documents = [
    {
        "document_id": "doc_fastapi_notes",
        "title": "FastAPI 学习笔记",
        "text": """
FastAPI 是一个用于构建 API 的现代 Python Web 框架。它支持类型提示，可以自动生成 OpenAPI 文档，并且适合构建后端服务。

在 FastAPI 中，请求体通常使用 Pydantic 模型定义。Pydantic 可以进行数据校验、类型转换和序列化。例如 TodoCreate 可以用于创建任务，TodoUpdate 可以用于更新任务，TodoResponse 可以用于返回给客户端。

SQLModel 结合了 SQLAlchemy 和 Pydantic 的能力，可以同时定义数据库表结构和数据模型。在 Todo API 中，可以使用 SQLModel 定义 Todo 表，包括 id、title、completed 和 due_time 字段。

SQLite 是一个轻量级关系型数据库，适合本地开发和小型项目。通过 SQLModel 的 Session，可以把 Todo 对象添加到数据库中，提交事务后数据库会生成 id。这样 Todo 就可以被持久化保存，而不是只存在内存中。

FastAPI 的 HTTPException 可以用于返回错误响应。例如用户请求一个不存在的 Todo 时，可以抛出 404 错误。日志 logging 可以帮助开发者观察接口调用、数据库操作和 AI 调用失败等情况。

环境变量通常放在 .env 文件中，例如 DASHSCOPE_API_KEY、DASHSCOPE_BASE_URL、DATABASE_URL。代码中可以使用 python-dotenv 加载这些配置，避免把密钥写死在源码里。
""",
    },
    {
        "document_id": "doc_rag_notes",
        "title": "RAG 学习笔记",
        "text": """
RAG 是 Retrieval-Augmented Generation，也就是检索增强生成。它的核心流程是先检索相关资料，再让大模型基于这些资料生成答案。

普通大模型只能使用训练中学到的通用知识和当前 prompt 中提供的信息。它不会自动知道用户的私人笔记、公司文档或项目代码。RAG 的作用就是在模型回答之前，先从外部知识库中找出相关内容。

Embedding 会把文本转换成高维向量。语义相近的文本，在向量空间中的方向或距离通常更接近。通过计算用户问题向量和文档 chunk 向量之间的相似度，系统可以找出最相关的文本片段。

文档通常需要切成 chunk。因为整篇文档可能太长，直接放进 prompt 会浪费 token，也会引入无关信息。chunk 太大可能包含太多噪声，chunk 太小可能丢失上下文。实际项目中通常会设置 chunk_size 和 overlap。

RAG 系统应该返回 sources。sources 可以包含 document_id、title、chunk_id、content 和 score。这样用户可以知道答案依据来自哪里，开发者也可以判断错误到底来自检索阶段还是生成阶段。

如果资料库中没有足够依据，严格 RAG 系统应该回答“我在已提供资料中没有找到足够依据”，而不是使用通用知识编造答案。这对于公司制度、法律、医疗和项目文档等场景尤其重要。
""",
    },
]


def get_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY. Please set it in .env")

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


EMBEDDING_BATCH_SIZE = 10


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    把多段文本转换成 embedding 向量。

    注意：
    DashScope 的 embedding API 单次 batch size 不能超过 10。
    所以这里要分批调用。
    """
    client = get_client()
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")

    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):    
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]

        response = client.embeddings.create(
            model=model,
            input=batch,
            encoding_format="float",
        )

        # 为了保证顺序稳定，按 index 排序
        batch_data = sorted(response.data, key=lambda item: item.index)

        batch_embeddings = [
            item.embedding
            for item in batch_data
        ]

        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def split_into_sentences(text: str) -> list[str]:
    """
    把文本切成句子，尽量保留中文/英文标点。

    例子：
    "FastAPI 是框架。它支持类型提示。" 
    -> ["FastAPI 是框架。", "它支持类型提示。"]
    """
    cleaned_text = text.strip()

    # 按中文句号、问号、感叹号、分号，以及英文 .?!; 后面的空白切分
    # (?<=...) 是正向后顾：表示在这些标点后面切
    parts = re.split(r"(?<=[。！？；.!?;])\s*", cleaned_text)

    sentences = [
        part.strip()
        for part in parts
        if part.strip()
    ]

    return sentences


def chunk_text(
    text: str,
    chunk_size: int = 260,
    overlap_sentences: int = 1,
) -> list[dict[str, Any]]:
    """
    句子级 chunk 切分函数。

    chunk_size:
        每个 chunk 尽量不超过多少字符。

    overlap_sentences:
        相邻 chunk 之间重叠多少个完整句子。
        注意：这里不是重叠多少字符，而是重叠多少句。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap_sentences < 0:
        raise ValueError("overlap_sentences cannot be negative")

    cleaned_text = text.strip()
    sentences = split_into_sentences(cleaned_text)

    chunks = []
    current_sentences: list[str] = []
    chunk_index = 0
    cursor = 0

    def add_chunk(sentences_for_chunk: list[str]) -> None:
        nonlocal chunk_index, cursor

        content = "".join(sentences_for_chunk).strip()
        if not content:
            return

        # 尽量在原文中找这个 chunk 的位置
        start = cleaned_text.find(content[:20], cursor)
        if start == -1:
            start = cursor

        end = start + len(content)
        cursor = max(cursor, end)

        chunks.append(
            {
                "chunk_index": chunk_index,
                "content": content,
                "start": start,
                "end": end,
            }
        )
        chunk_index += 1

    for sentence in sentences:
        # 如果单个句子本身就超过 chunk_size，那只能硬切这个长句
        if len(sentence) > chunk_size:
            if current_sentences:
                add_chunk(current_sentences)
                current_sentences = []

            start = 0
            while start < len(sentence):
                end = min(start + chunk_size, len(sentence))
                part = sentence[start:end].strip()

                if part:
                    add_chunk([part])

                if end == len(sentence):
                    break

                start = end

            continue

        candidate_sentences = current_sentences + [sentence]
        candidate_content = "".join(candidate_sentences)

        if len(candidate_content) <= chunk_size:
            current_sentences.append(sentence)
        else:
            # 当前 chunk 已经满了，先保存
            add_chunk(current_sentences)

            # 用完整句子做 overlap
            if overlap_sentences > 0:
                overlap = current_sentences[-overlap_sentences:]
            else:
                overlap = []

            current_sentences = overlap + [sentence]

    if current_sentences:
        add_chunk(current_sentences)

    return chunks

def build_chunks(documents: list[dict[str, str]]) -> list[dict[str, Any]]:
    """
    把多个长文档切成 chunk，并保留来源 metadata。
    """
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["text"], chunk_size=260, overlap_sentences=1)

        for chunk in chunks:
            chunk_id = f"{doc['document_id']}_chunk_{chunk['chunk_index']}"

            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": doc["document_id"],
                    "title": doc["title"],
                    "content": chunk["content"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                }
            )

    return all_chunks


def build_chunk_index(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    给每个 chunk 生成 embedding，形成 chunk-level index。
    """
    texts = [
        f"{chunk['title']}\n{chunk['content']}"
        for chunk in chunks
    ]

    embeddings = embed_texts(texts)

    index = []

    for chunk, embedding in zip(chunks, embeddings):
        index.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "title": chunk["title"],
                "content": chunk["content"],
                "start": chunk["start"],
                "end": chunk["end"],
                "embedding": embedding,
            }
        )

    return index


def search_chunks(
    question: str,
    index: list[dict[str, Any]],
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """
    对用户问题做 embedding，然后检索最相关的 chunks。
    """
    question_embedding = embed_texts([question])[0]

    results = []

    for item in index:
        score = cosine_similarity(question_embedding, item["embedding"])

        results.append(
            {
                "chunk_id": item["chunk_id"],
                "document_id": item["document_id"],
                "title": item["title"],
                "content": item["content"],
                "start": item["start"],
                "end": item["end"],
                "score": score,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:top_k]


def format_context(chunks: list[dict[str, Any]]) -> str:    
    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"""[Source {index}]
            document_id: {chunk["document_id"]}
            title: {chunk["title"]}
            chunk_id: {chunk["chunk_id"]}
            score: {chunk["score"]:.4f}
            content:
            {chunk["content"]}
        """
        )

    return "\n".join(context_parts)


def build_rag_prompt(question: str, context: str) -> str:   
    return f"""
请你作为一个严谨的 RAG 问答助手回答问题。

规则：
1. 你只能根据下面的【资料】回答问题。
2. 如果【资料】中没有足够信息，请回答：“我在已提供资料中没有找到足够依据。”
3. 不要使用资料之外的通用知识补充答案。
4. 回答要简洁、准确。
5. 回答时说明你主要依据了哪些 Source。

【资料】
{context}

【问题】
{question}
""".strip()


def generate_answer(question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    '''
    根据检索到的 chunks 生成答案。
    '''
    client = get_client()   
    model = os.getenv("DASHSCOPE_MODEL", "qwen3.5-plus")    

    context = format_context(retrieved_chunks)
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


def generate_answer_stream(question: str, retrieved_chunks: list[dict[str, Any]]):
    """
    流式生成答案。

    和 generate_answer() 的区别：
    - generate_answer() 等完整答案生成完再 return
    - generate_answer_stream() 每拿到一个文本片段就 yield 出去
    """
    client = get_client()
    model = os.getenv("DASHSCOPE_MODEL", "qwen3.5-plus")

    context = format_context(retrieved_chunks)
    prompt = build_rag_prompt(question, context)

    stream = client.chat.completions.create(
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
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        # 有些最后的 chunk 可能只包含 usage，没有 choices
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        piece = getattr(delta, "content", None)

        if piece:
            yield piece # yield: 每拿到一个文本片段就返回给调用方，调用方可以边生成边显示，而不需要等整个答案生成完

def answer_with_chunk_rag(
    question: str,
    index: list[dict[str, Any]],
    top_k: int = 4,
) -> dict[str, Any]:    # RAG 主流程：先检索相关 chunks，再生成答案，并返回来源和检索状态
    
    retrieved_chunks = search_chunks(question, index, top_k=top_k)  # 检索相关 chunks
    
    top_score = retrieved_chunks[0]["score"] if retrieved_chunks else 0.0   # 记录最高分，判断是否达到最低检索质量要求
    
    filtered_chunks = [
        chunk for chunk in retrieved_chunks
        if chunk["score"] >= SOURCE_SCORE_THRESHOLD
    ]   # 过滤掉得分过低的 chunks，避免它们对生成答案造成干扰

    sources = [
        {
            "document_id": chunk["document_id"],
            "title": chunk["title"],
            "chunk_id": chunk["chunk_id"],
            "content": chunk["content"],
            "score": round(chunk["score"], 4),
            "start": chunk["start"],
            "end": chunk["end"],
        }
        for chunk in filtered_chunks
    ]   # 构建 sources 列表，包含每个 chunk 的来源信息和得分，方便后续返回给用户

    if top_score < MIN_RETRIEVAL_SCORE or not filtered_chunks:
        return {
            "question": question,
            "answer": "我在已提供资料中没有找到足够依据。",
            "sources": sources,
            "retrieval_status": "low_confidence",
            "top_score": round(top_score, 4),
        }   # 如果最高分都达不到最低要求，或者没有任何 chunk 通过过滤，直接拒答，不调用生成模型，节省成本并避免错误答案

    answer = generate_answer(question, filtered_chunks) # 调用生成模型，根据过滤后的 chunks 生成答案

    if "没有找到足够依据" in answer:
        retrieval_status = "insufficient_context"
    else:
        retrieval_status = "ok"

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieval_status": retrieval_status,
        "top_score": round(top_score, 4),
    }


def run_query(question: str, index: list[dict[str, Any]]) -> None:
    result = answer_with_chunk_rag(question, index, top_k=4)

    print("\n" + "=" * 100)
    print(f"问题：{question}")
    print("=" * 100)

    print("\n回答：")
    print(result["answer"])

    print("\n检索状态：")
    print(result["retrieval_status"])

    print("\nTop Score：")
    print(result["top_score"])

    print("\nSources：")
    for source in result["sources"]:
        print("-" * 80)
        print(f"document_id: {source['document_id']}")
        print(f"title: {source['title']}")
        print(f"chunk_id: {source['chunk_id']}")
        print(f"score: {source['score']}")
        print(f"range: {source['start']} - {source['end']}")
        print(f"content: {source['content']}")

    print("\n完整 JSON：")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_query_stream(question: str, index: list[dict[str, Any]]) -> None:
    retrieved_chunks = search_chunks(question, index, top_k=4)

    top_score = retrieved_chunks[0]["score"] if retrieved_chunks else 0.0

    filtered_chunks = [
        chunk for chunk in retrieved_chunks
        if chunk["score"] >= SOURCE_SCORE_THRESHOLD
    ]

    sources = [
        {
            "document_id": chunk["document_id"],
            "title": chunk["title"],
            "chunk_id": chunk["chunk_id"],
            "content": chunk["content"],
            "score": round(chunk["score"], 4),
            "start": chunk["start"],
            "end": chunk["end"],
        }
        for chunk in filtered_chunks
    ]

    print("\n" + "=" * 100)
    print(f"问题：{question}")
    print("=" * 100)

    print("\nTop Score：")
    print(round(top_score, 4))

    print("\nSources：")
    for source in sources:
        print("-" * 80)
        print(f"document_id: {source['document_id']}")
        print(f"title: {source['title']}")
        print(f"chunk_id: {source['chunk_id']}")
        print(f"score: {source['score']}")
        print(f"range: {source['start']} - {source['end']}")
        print(f"content: {source['content']}")

    if top_score < MIN_RETRIEVAL_SCORE or not filtered_chunks:
        print("\n回答：")
        print("我在已提供资料中没有找到足够依据。")
        print("\n检索状态：")
        print("low_confidence")
        return

    print("\n回答：")
    answer_parts = []

    for piece in generate_answer_stream(question, filtered_chunks):
        print(piece, end="", flush=True)
        answer_parts.append(piece)

    answer = "".join(answer_parts)

    if "没有找到足够依据" in answer:
        retrieval_status = "insufficient_context"
    else:
        retrieval_status = "ok"

    print("\n\n检索状态：")
    print(retrieval_status)


def main() -> None:
    print("正在切分文档...")
    chunks = build_chunks(long_documents)
    print(f"切分完成，共生成 {len(chunks)} 个 chunks。")

    print("正在构建 chunk-level embedding index...")
    index = build_chunk_index(chunks)
    print("index 构建完成。")

    test_questions = [
        "RAG 为什么要切块？",
        "怎么保存 Todo 到 SQLite 数据库？",
        "FastAPI 里请求体和响应模型怎么定义？",
        "为什么资料库没有答案时应该拒答？",
        "Kubernetes 是什么？",
    ]
    
    for question in test_questions:
        run_query_stream(question, index)

if __name__ == "__main__":
    main()