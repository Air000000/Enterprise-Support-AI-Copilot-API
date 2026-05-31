from types import SimpleNamespace

from fastapi.testclient import TestClient

import main


client = TestClient(main.app)

def test_rag_search_returns_results(monkeypatch):
    fake_results = [
        SimpleNamespace(
            document_id="doc_embedding_notes",
            chunk_id="doc_embedding_notes_chunk_0",
            title="Embedding Notes",
            source_path="experiments/docs/embedding.md",
            chunk_index=0,
            distance=0.827159,
            content="# Embedding Notes\n\nEmbedding 是把文本转换成向量的技术。",
        )
    ]

    def fake_search_chroma(query: str, top_k: int):
        return fake_results

    monkeypatch.setattr(main, "search_chroma", fake_search_chroma)  

    response = client.post(
        "/rag/search",
        json={
            "query": "为什么系统能判断两段文字语义相近？",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["query"] == "为什么系统能判断两段文字语义相近？"
    assert data["top_k"] == 3
    assert data["total_hits"] == 1
    assert data["results"][0]["document_id"] == "doc_embedding_notes"
    assert data["results"][0]["chunk_id"] == "doc_embedding_notes_chunk_0"
    assert data["results"][0]["distance"] == 0.8272
    assert "\n" not in data["results"][0]["preview"]


def test_rag_ask_returns_answer_and_sources(monkeypatch):
    fake_source = SimpleNamespace(
        document_id="doc_rag_notes",
        chunk_id="doc_rag_notes_chunk_0",
        title="RAG Notes",
        source_path="experiments/docs/rag.md",
        chunk_index=0,
        distance=0.512345,
        preview="# RAG Notes\n\nRAG 通过检索外部文档，把相关上下文提供给大模型。",
    )

    fake_rag_result = SimpleNamespace(
        question="RAG 为什么需要 chunk？",
        answer="RAG 需要 chunk 是为了把长文档切成更适合检索和上下文拼接的小片段。",
        retrieval_status="ok",
        top_distance=0.512345,
        sources=[fake_source],
    )

    def fake_ask_rag(question: str, top_k: int, max_distance: float):
        return fake_rag_result

    monkeypatch.setattr(main, "ask_rag", fake_ask_rag)

    response = client.post(
        "/rag/ask",
        json={
            "question": "RAG 为什么需要 chunk？",
            "top_k": 3,
            "max_distance": 0.9,
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["question"] == "RAG 为什么需要 chunk？"
    assert data["answer"] == "RAG 需要 chunk 是为了把长文档切成更适合检索和上下文拼接的小片段。"
    assert data["retrieval_status"] == "ok"
    assert data["top_distance"] == 0.5123
    assert len(data["sources"]) == 1
    assert data["sources"][0]["document_id"] == "doc_rag_notes"
    assert data["sources"][0]["chunk_id"] == "doc_rag_notes_chunk_0"
    assert data["sources"][0]["distance"] == 0.5123
    assert "\n" not in data["sources"][0]["preview"]