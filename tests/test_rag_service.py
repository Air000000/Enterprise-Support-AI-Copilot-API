from types import SimpleNamespace

import services.rag_service as rag_service


def test_search_documents_calls_search_chroma(monkeypatch):
    # 1. 准备 fake return
    fake_results = [
        SimpleNamespace(
            document_id="doc1",
            chunk_id="chunk1",
            title="Document 1",
            source_path="path/to/doc1",
            chunk_index=0,
            distance=0.8272,
            content="这是文档 1 的内容文本。",
        )
    ]

    # 2. 用一个 dict 记录 fake_search_chroma 收到的参数
    calls = {}

    def fake_search_chroma(
        query: str, 
        top_k: int,
        tenant_id: str | None = None,
        category: str | None = None,
    ):
        calls["query"] = query
        calls["top_k"] = top_k
        calls["tenant_id"] = tenant_id
        calls["category"] = category
        return fake_results

    # 3. monkeypatch rag_service.search_chroma
    monkeypatch.setattr(rag_service, "search_chroma", fake_search_chroma)

    # 4. 调用 search_documents
    query = "test query"
    top_k = 5

    result = rag_service.search_documents(
        query=query, 
        top_k=top_k,
        tenant_id="tenant_demo",
        category="general",
    )

    # 5. assert
    assert result == fake_results
    assert calls["query"] == query
    assert calls["top_k"] == top_k
    assert calls["tenant_id"] == "tenant_demo"
    assert calls["category"] == "general"


def test_answer_question_calls_ask_rag(monkeypatch):
    # 1. 准备 fake_rag_result
    fake_rag_result = SimpleNamespace(
        question="RAG 为什么需要 chunk？",
        answer="RAG 通过检索外部文档，把相关上下文提供给大模型。",
        retrieval_status="ok",
        top_distance=0.512345,
        sources=[],
    )

    # 2. 用 calls 记录 fake_ask_rag 收到的参数
    calls = {}

    def fake_ask_rag(
        question: str, 
        top_k: int, 
        max_distance: float,
        tenant_id: str | None = None,
        category: str | None = None,
    ):
        calls["question"] = question
        calls["top_k"] = top_k
        calls["max_distance"] = max_distance
        calls["tenant_id"] = tenant_id
        calls["category"] = category
        return fake_rag_result

    # 3. monkeypatch rag_service.ask_rag
    monkeypatch.setattr(rag_service, "ask_rag", fake_ask_rag)

    # 4. 准备输入并调用 answer_question
    question = "RAG 为什么需要 chunk？"
    top_k = 5
    max_distance = 0.8

    result = rag_service.answer_question(
        question=question, 
        top_k=top_k, 
        max_distance=max_distance,
        tenant_id="tenant_demo",
        category="general"
    )

    # 5. assert 返回值和参数传递
    assert result == fake_rag_result
    assert calls["question"] == question
    assert calls["top_k"] == top_k
    assert calls["max_distance"] == max_distance
    assert calls["tenant_id"] == "tenant_demo"
    assert calls["category"] == "general"