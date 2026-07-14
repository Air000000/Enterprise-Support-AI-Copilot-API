from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from auth import CurrentUser, get_current_user
import main
import routers.rag as rag_router


@pytest.fixture(autouse=True)
def default_auth_user():
    user = CurrentUser(
        sub="employee",
        user_id="user_demo",
        tenant_id="tenant_demo",
        role="employee",
    )
    main.app.dependency_overrides[get_current_user] = lambda: user
    yield
    main.app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture(autouse=True)
def default_retrieval_log(monkeypatch):
    monkeypatch.setattr(
        rag_router,
        "create_retrieval_log_service",
        lambda retrieval_log_create: None,
    )


@pytest.fixture()
def client():
    with TestClient(main.app) as test_client:
        yield test_client

def test_rag_search_returns_results(client, monkeypatch):
    fake_results = [
        SimpleNamespace(
            document_id="doc_embedding_notes",
            chunk_id="doc_embedding_notes_chunk_0",
            title="Embedding Notes",
            source_path="experiments/docs/embedding.md",
            chunk_index=0,
            distance=0.827159,
            tenant_id="tenant_demo",
            category="general",
            content="# Embedding Notes\n\nEmbedding 是把文本转换成向量的技术。",
        )
    ]

    calls = {}
    log_calls = {}

    def fake_search_documents(
    query: str,
    top_k: int,
    tenant_id: str,
    category: str | None,
    ):
        calls["query"] = query
        calls["top_k"] = top_k
        calls["tenant_id"] = tenant_id
        calls["category"] = category
        return fake_results

    def fake_create_retrieval_log_service(retrieval_log_create):
        log_calls["tenant_id"] = retrieval_log_create.tenant_id
        log_calls["endpoint"] = retrieval_log_create.endpoint
        log_calls["query_text"] = retrieval_log_create.query_text
        log_calls["top_k"] = retrieval_log_create.top_k
        log_calls["category"] = retrieval_log_create.category
        log_calls["retrieval_status"] = retrieval_log_create.retrieval_status
        log_calls["total_hits"] = retrieval_log_create.total_hits
        log_calls["top_distance"] = retrieval_log_create.top_distance
        log_calls["source_documents_json"] = (
            retrieval_log_create.source_documents_json
        )
        log_calls["scores_json"] = retrieval_log_create.scores_json
        log_calls["latency_ms"] = retrieval_log_create.latency_ms

    monkeypatch.setattr(rag_router, "search_documents", fake_search_documents)
    monkeypatch.setattr(
        rag_router,
        "create_retrieval_log_service",
        fake_create_retrieval_log_service,
    )

    response = client.post(
        "/rag/search",
        json={
            "query": "为什么系统能判断两段文字语义相近？",
            "top_k": 3,
            "category": "general",
        },
    )

    assert response.status_code == 200

    assert calls["query"] == "为什么系统能判断两段文字语义相近？"
    assert calls["top_k"] == 3
    assert calls["tenant_id"] == "tenant_demo"
    assert calls["category"] == "general"

    data = response.json()
    assert data["query"] == "为什么系统能判断两段文字语义相近？"
    assert data["top_k"] == 3
    assert data["total_hits"] == 1
    assert data["results"][0]["document_id"] == "doc_embedding_notes"
    assert data["results"][0]["chunk_id"] == "doc_embedding_notes_chunk_0"
    assert data["results"][0]["distance"] == 0.8272
    assert data["results"][0]["tenant_id"] == "tenant_demo"
    assert data["results"][0]["category"] == "general"
    assert "\n" not in data["results"][0]["preview"]


    assert log_calls["tenant_id"] == "tenant_demo"
    assert log_calls["endpoint"] == "search"
    assert log_calls["query_text"] == "为什么系统能判断两段文字语义相近？"
    assert log_calls["top_k"] == 3
    assert log_calls["category"] == "general"
    assert log_calls["retrieval_status"] == "ok"
    assert log_calls["total_hits"] == 1
    assert log_calls["top_distance"] == 0.8272
    assert "doc_embedding_notes" in log_calls["source_documents_json"]
    assert "0.8272" in log_calls["scores_json"]
    assert isinstance(log_calls["latency_ms"], int)

def test_rag_ask_returns_answer_and_sources(client, monkeypatch):
    fake_source = SimpleNamespace(
        document_id="doc_rag_notes",
        chunk_id="doc_rag_notes_chunk_0",
        title="RAG Notes",
        source_path="experiments/docs/rag.md",
        chunk_index=0,
        distance=0.512345,
        tenant_id="tenant_demo",
        category="general",
        preview="# RAG Notes\n\nRAG 通过检索外部文档，把相关上下文提供给大模型。",
    )

    fake_rag_result = SimpleNamespace(
        question="RAG 为什么需要 chunk？",
        answer="RAG 需要 chunk 是为了把长文档切成更适合检索和上下文拼接的小片段。",
        retrieval_status="ok",
        top_distance=0.512345,
        sources=[fake_source],
    )

    calls = {}
    log_calls = {}

    def fake_answer_question(
    question: str,
    top_k: int,
    max_distance: float,
    tenant_id: str,
    category: str | None,
    ):
        calls["question"] = question
        calls["top_k"] = top_k
        calls["max_distance"] = max_distance
        calls["tenant_id"] = tenant_id
        calls["category"] = category
        return fake_rag_result

    def fake_create_retrieval_log_service(retrieval_log_create):
        log_calls["tenant_id"] = retrieval_log_create.tenant_id
        log_calls["endpoint"] = retrieval_log_create.endpoint
        log_calls["query_text"] = retrieval_log_create.query_text
        log_calls["top_k"] = retrieval_log_create.top_k
        log_calls["category"] = retrieval_log_create.category
        log_calls["retrieval_status"] = retrieval_log_create.retrieval_status
        log_calls["total_hits"] = retrieval_log_create.total_hits
        log_calls["top_distance"] = retrieval_log_create.top_distance
        log_calls["source_documents_json"] = (
            retrieval_log_create.source_documents_json
        )
        log_calls["scores_json"] = retrieval_log_create.scores_json
        log_calls["latency_ms"] = retrieval_log_create.latency_ms
    
    monkeypatch.setattr(rag_router, "answer_question", fake_answer_question)
    monkeypatch.setattr(
        rag_router,
        "create_retrieval_log_service",
        fake_create_retrieval_log_service,
    )

    response = client.post(
        "/rag/ask",
        json={
            "question": "RAG 为什么需要 chunk？",
            "top_k": 3,
            "max_distance": 0.9,
            "category": "general",
        },
    )

    assert response.status_code == 200

    assert calls["question"] == "RAG 为什么需要 chunk？"
    assert calls["top_k"] == 3
    assert calls["max_distance"] == 0.9
    assert calls["tenant_id"] == "tenant_demo"
    assert calls["category"] == "general"

    data = response.json()
    assert data["question"] == "RAG 为什么需要 chunk？"
    assert data["answer"] == "RAG 需要 chunk 是为了把长文档切成更适合检索和上下文拼接的小片段。"
    assert data["retrieval_status"] == "ok"
    assert data["top_distance"] == 0.5123
    assert len(data["sources"]) == 1
    assert data["sources"][0]["document_id"] == "doc_rag_notes"
    assert data["sources"][0]["chunk_id"] == "doc_rag_notes_chunk_0"
    assert data["sources"][0]["distance"] == 0.5123
    assert data["sources"][0]["tenant_id"] == "tenant_demo"
    assert data["sources"][0]["category"] == "general"
    assert "\n" not in data["sources"][0]["preview"]

    assert log_calls["tenant_id"] == "tenant_demo"
    assert log_calls["endpoint"] == "ask"
    assert log_calls["query_text"] == "RAG 为什么需要 chunk？"
    assert log_calls["top_k"] == 3
    assert log_calls["category"] == "general"
    assert log_calls["retrieval_status"] == "ok"
    assert log_calls["total_hits"] == 1
    assert log_calls["top_distance"] == 0.5123
    assert "doc_rag_notes" in log_calls["source_documents_json"]
    assert "0.5123" in log_calls["scores_json"]
    assert isinstance(log_calls["latency_ms"], int)


def test_rag_search_rejects_empty_query(client):
    response = client.post(
        "/rag/search",
        json={
            "query": "",
            "top_k": 3,
        },
    )

    assert response.status_code == 422  # 请求参数不合法，query 不能为空

def test_rag_ask_rejects_empty_question(client):
    response = client.post(
        "/rag/ask",
        json={
            "question": "",
            "top_k": 3,
            "max_distance": 0.9,
        },
    )

    assert response.status_code == 422  # 请求参数不合法，question 不能为空

def test_rag_search_rejects_invalid_top_k(client):
    response = client.post(
        "/rag/search",
        json={
            "query": "RAG 为什么需要 chunk？",
            "top_k": 0,
        },
    )

    assert response.status_code == 422  # top_k 不合法，必须大于等于1

def test_rag_ask_rejects_invalid_max_distance(client):
    response = client.post(
        "/rag/ask",
        json={
            "question": "RAG 为什么需要 chunk？",
            "top_k": 3,
            "max_distance": 0,
        },
    )

    assert response.status_code == 422  # max_distance 不合法，必须大于0

def test_rag_search_returns_500_when_service_fails(client, monkeypatch):
    def fake_search_documents(
        query: str,
        top_k: int,
        tenant_id: str,
        category: str | None,
    ):
        raise RuntimeError("Chroma is unavailable")

    monkeypatch.setattr(rag_router, "search_documents", fake_search_documents)

    response = client.post(
        "/rag/search",
        json={
            "query": "RAG 为什么需要 chunk？",
            "top_k": 3,
        },
    )

    assert response.status_code == 500
    assert "RAG search failed" in response.json()["detail"] 

def test_rag_ask_returns_500_when_service_fails(client, monkeypatch):
    def fake_answer_question(
        question: str,
        top_k: int,
        max_distance: float,
        tenant_id: str,
        category: str | None,
    ):
        raise RuntimeError("LLM is unavailable")

    log_calls = {}

    def fake_create_retrieval_log_service(retrieval_log_create):
        log_calls["tenant_id"] = retrieval_log_create.tenant_id
        log_calls["endpoint"] = retrieval_log_create.endpoint
        log_calls["query_text"] = retrieval_log_create.query_text
        log_calls["top_k"] = retrieval_log_create.top_k
        log_calls["category"] = retrieval_log_create.category
        log_calls["retrieval_status"] = retrieval_log_create.retrieval_status
        log_calls["total_hits"] = retrieval_log_create.total_hits
        log_calls["latency_ms"] = retrieval_log_create.latency_ms
        log_calls["error_message"] = retrieval_log_create.error_message

    monkeypatch.setattr(rag_router, "answer_question", fake_answer_question)
    monkeypatch.setattr(
        rag_router,
        "create_retrieval_log_service",
        fake_create_retrieval_log_service,
    )

    response = client.post(
        "/rag/ask",
        json={
            "question": "RAG 为什么需要 chunk？",
            "top_k": 3,
            "max_distance": 0.9,
        },
    )

    assert response.status_code == 500
    assert "RAG ask failed" in response.json()["detail"]

    assert log_calls["tenant_id"] == "tenant_demo"
    assert log_calls["endpoint"] == "ask"
    assert log_calls["query_text"] == "RAG 为什么需要 chunk？"
    assert log_calls["top_k"] == 3
    assert log_calls["category"] is None
    assert log_calls["retrieval_status"] == "failed"
    assert log_calls["total_hits"] == 0
    assert isinstance(log_calls["latency_ms"], int)
    assert log_calls["error_message"] == "LLM is unavailable"