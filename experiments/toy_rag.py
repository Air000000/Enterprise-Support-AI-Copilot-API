import re
from typing import List, Dict, Any


STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "或", "也", "都", "就",
    "一个", "什么", "怎么", "如何", "可以", "用于", "进行",
    "？", "?", "，", ",", "。", "."
}

documents = [
    {
        "id": "doc_1",
        "title": "FastAPI 基础",
        "text": "FastAPI 是一个用于构建 API 的现代 Python Web 框架，支持类型提示和自动生成接口文档。"
    },
    {
        "id": "doc_2",
        "title": "Pydantic 基础",
        "text": "Pydantic 用于数据校验和序列化，可以定义请求体和响应模型。"
    },
    {
        "id": "doc_3",
        "title": "SQLModel 基础",
        "text": "SQLModel 结合了 SQLAlchemy 和 Pydantic，可以用于定义数据库表并执行查询。"
    },
    {
        "id": "doc_4",
        "title": "RAG 基础",
        "text": "RAG 是检索增强生成，系统会先检索相关资料，再让大模型基于资料生成答案。"
    },
    {
        "id": "doc_5",
        "title": "Embedding 基础",
        "text": "Embedding 会把文本转换成向量，让系统可以根据语义相似度检索相关内容。"
    }
]


def tokenize(text: str) -> List[str]:
    """
    一个非常简陋的中文/英文 token 切分函数。

    注意：
    这不是生产级分词器。
    它只是为了让我们理解 retrieval 的基本流程。

    对英文和数字：按连续单词切。
    对中文：按单个汉字切。
    """
    text = text.lower()

    tokens = re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text)

    return tokens

# 根据关键词重叠数量，返回最相关的 top_k 个文档。
def retrieve_by_keyword(
    question: str,
    documents: List[Dict[str, str]],
    top_k: int = 2
) -> List[Dict[str, Any]]:
    """
    根据关键词重叠数量，返回最相关的 top_k 个文档。

    这不是语义搜索。
    这只是一个玩具版关键词检索。
    """

    question_tokens = {
    token for token in tokenize(question)
    if token not in STOPWORDS
}

    results = []

    for doc in documents:
        doc_text = doc["title"] + " " + doc["text"]
        doc_tokens = {
            token for token in tokenize(doc_text)
            if token not in STOPWORDS
        }

        matched_tokens = question_tokens.intersection(doc_tokens)
        score = len(matched_tokens)

        result = {
            "id": doc["id"],
            "title": doc["title"],
            "text": doc["text"],
            "score": score,
            "matched_tokens": list(matched_tokens)
        }

        results.append(result)

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:top_k]


def main():
    question = "怎么保存 Todo 到数据库？"

    results = retrieve_by_keyword(question, documents, top_k=2)

    print(f"问题：{question}")
    print("=" * 50)

    for index, result in enumerate(results, start=1):   
        print(f"Top {index}")
        print(f"文档 ID: {result['id']}")
        print(f"标题: {result['title']}")
        print(f"分数: {result['score']}")
        print(f"匹配 token: {result['matched_tokens']}")
        print(f"内容: {result['text']}")
        print("-" * 50)


if __name__ == "__main__":
    main()