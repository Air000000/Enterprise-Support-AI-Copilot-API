"""Offline runtime checks: real SQL/BM25/RRF/Chroma, mocked external model transport."""

import json
from types import SimpleNamespace
from uuid import uuid4

import chromadb
import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI
from sqlmodel import Session, SQLModel, create_engine, select

import main
import rag_runtime.hybrid_search as hybrid
import routers.rag as router
import services.document_service as documents
import services.rag_service as service
from models.document import Document, DocumentChunk


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'hybrid.db'}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(hybrid, "engine", engine)
    monkeypatch.setattr(documents, "engine", engine)
    monkeypatch.setenv("RAG_HYBRID_SEARCH_ENABLED", "true")
    calls = []

    def transport(request):
        body = json.loads(request.content)
        calls.append(body)
        order = sorted(range(len(body["documents"])), key=lambda i: "needle" not in body["documents"][i])
        return httpx.Response(200, json={"results": [
            {"index": i, "relevance_score": 1 / (rank + 1)} for rank, i in enumerate(order)
        ]})

    monkeypatch.setattr(hybrid, "get_rerank_client", lambda: OpenAI(
        api_key="offline", base_url="https://rerank.invalid/v1", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    ))
    monkeypatch.setattr(hybrid, "embed_texts", lambda texts: [[1.0, 0.0] for _ in texts])

    def query(**kwargs):
        ids = kwargs["ids"][:kwargs["n_results"]]
        return {"ids": [ids], "distances": [[0.1] * len(ids)]}

    monkeypatch.setattr(hybrid, "get_collection", lambda: SimpleNamespace(query=query))
    yield engine, calls
    engine.dispose()


def add_document(engine, *, tenant="tenant_demo", category="it", status="indexed", content="VPN guide", count=1):
    doc = Document(tenant_id=tenant, uploaded_by="user_demo", filename="guide.md", file_type="md",
                   source_path="managed/guide.md", checksum="test", category=category, status=status)
    with Session(engine) as session:
        session.add(doc)
        for i in range(count):
            session.add(DocumentChunk(tenant_id=tenant, document_id=doc.id, category=category,
                                      chunk_index=i, content=content,
                                      embedding_id=f"doc:{doc.id}:v1:chunk:{i}"))
        session.commit()
        session.refresh(doc)
    return doc


def test_frozen_pipeline_caps_and_bm25_only_score(runtime):
    engine, calls = runtime
    doc = add_document(engine, count=120, content="generic network manual")
    with Session(engine) as session:
        chunk = session.exec(select(DocumentChunk).where(DocumentChunk.document_id == doc.id, DocumentChunk.chunk_index == 119)).one()
        chunk.content = "needle exact error-code correction"
        session.add(chunk)
        session.commit()
    results = hybrid.search_hybrid("needle", 14, "tenant_demo", "it")
    assert len(results) == 14
    assert results[0].chunk_index == 119
    assert results[0].distance is None and results[0].dense_rank is None
    assert results[0].bm25_rank == 1 and results[0].fusion_score == pytest.approx(1 / 61)
    assert results[0].rerank_score == 1.0
    assert len(calls) == 1 and len(calls[0]["documents"]) == 100
    assert calls[0]["top_n"] == 100 and calls[0]["model"] == "qwen3-rerank"


def test_sql_visibility_scopes_both_channels_before_provider(runtime, monkeypatch):
    engine, calls = runtime
    good = add_document(engine, content="needle authorized")
    for status in ("uploaded", "indexing", "failed", "deleted"):
        add_document(engine, status=status, content="forbidden " + status)
    add_document(engine, tenant="foreign", content="forbidden tenant")
    add_document(engine, category="hr", content="forbidden category")
    stale = add_document(engine, content="forbidden version")
    corrupt = add_document(engine, content="forbidden chunk tenant")
    with Session(engine) as session:
        row = session.get(Document, stale.id)
        row.version = 2
        session.add(row)
        chunk = session.exec(select(DocumentChunk).where(DocumentChunk.document_id == corrupt.id)).one()
        chunk.tenant_id = "foreign"
        session.add(chunk)
        session.commit()
    expected = [f"doc:{good.id}:v1:chunk:0"]

    def query(**kwargs):
        assert kwargs["ids"] == expected
        assert kwargs["where"] == {"$and": [{"tenant_id": "tenant_demo"}, {"category": "it"}]}
        return {"ids": [expected], "distances": [[0.2]]}

    monkeypatch.setattr(hybrid, "get_collection", lambda: SimpleNamespace(query=query))
    assert [r.document_id for r in hybrid.search_hybrid("needle", 14, "tenant_demo", "it")] == [good.id]
    assert calls[0]["documents"] == ["needle authorized"]


def test_empty_and_oversize_corpus_never_embed(runtime, monkeypatch):
    engine, calls = runtime
    monkeypatch.setattr(hybrid, "embed_texts", lambda _: pytest.fail("embedding must not run"))
    assert hybrid.search_hybrid("VPN", 14, "empty") == []
    monkeypatch.setattr(hybrid, "MAX_VISIBLE_CHUNKS", 1)
    add_document(engine, count=2)
    with pytest.raises(RuntimeError, match="exceeds"):
        hybrid.search_hybrid("VPN", 14, "tenant_demo")
    assert calls == []


@pytest.mark.parametrize("invalid", ["foreign", "duplicate", "nan"])
def test_invalid_dense_ranking_never_reaches_provider(runtime, monkeypatch, invalid):
    engine, calls = runtime
    doc = add_document(engine)
    cid = f"doc:{doc.id}:v1:chunk:0"
    ids = ["foreign"] if invalid == "foreign" else [cid, cid] if invalid == "duplicate" else [cid]
    scores = [float("nan")] if invalid == "nan" else [0.1] * len(ids)
    monkeypatch.setattr(hybrid, "get_collection", lambda: SimpleNamespace(
        query=lambda **_: {"ids": [ids], "distances": [scores]}))
    with pytest.raises(RuntimeError, match="dense ranking"):
        hybrid.search_hybrid("VPN", 14, "tenant_demo")
    assert calls == []


@pytest.mark.parametrize("stage", ["before_provider", "after_provider"])
def test_document_change_aborts_search(runtime, monkeypatch, stage):
    engine, calls = runtime
    doc = add_document(engine)

    def delete():
        with Session(engine) as session:
            row = session.get(Document, doc.id)
            row.status = "deleted"
            session.add(row)
            session.commit()

    if stage == "before_provider":
        def query(**_):
            delete()
            return {"ids": [[f"doc:{doc.id}:v1:chunk:0"]], "distances": [[0.1]]}
        monkeypatch.setattr(hybrid, "get_collection", lambda: SimpleNamespace(query=query))
    else:
        original = hybrid.rerank_candidates

        def rerank(*args, **kwargs):
            result = original(*args, **kwargs)
            delete()
            return result
        monkeypatch.setattr(hybrid, "rerank_candidates", rerank)
    with pytest.raises(RuntimeError, match="changed"):
        hybrid.search_hybrid("VPN", 14, "tenant_demo")
    assert len(calls) == (stage == "after_provider")


def test_chinese_only_corpus_uses_dense_without_empty_bm25_crash(runtime):
    engine, calls = runtime
    add_document(engine, content="这是网络配置说明")
    results = hybrid.search_hybrid("网络配置", 14, "tenant_demo")
    assert len(results) == 1 and results[0].bm25_rank is None
    assert results[0].dense_rank == 1 and len(calls) == 1


def test_real_chroma_upload_reindex_failure_delete_lifecycle(runtime, tmp_path, monkeypatch):
    _, calls = runtime
    client = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
    name = "hybrid_" + uuid4().hex
    collection = client.create_collection(name, embedding_function=None)
    monkeypatch.setattr(hybrid, "get_collection", lambda: collection)
    def embed(texts):
        return [[1.0, 0.0] for _ in texts]
    try:
        doc = documents.create_document_from_bytes(filename="vpn.md", content=b"VPN needle troubleshooting guide.",
            tenant_id="tenant_demo", uploaded_by="user_demo", category="it", storage_root=tmp_path)
        assert hybrid.search_hybrid("needle", 14, "tenant_demo") == []
        options = dict(document_id=doc.id, tenant_id="tenant_demo", embedding_function=embed, chroma_collection=collection)
        documents.index_document(**options)
        collection.add(ids=["foreign-vector"], embeddings=[[1.0, 0.0]],
                       metadatas=[{"tenant_id": "foreign", "category": "it"}])
        assert hybrid.search_hybrid("needle", 14, "tenant_demo")[0].document_id == doc.id
        documents.index_document(**options)
        assert len(hybrid.search_hybrid("needle", 14, "tenant_demo")) == 1

        # Simulate a provider add that writes vectors then fails; SQL must hide leftovers.
        def partial_add(**kwargs):
            collection.add(**kwargs)
            raise RuntimeError("simulated partial vector write")
        with pytest.raises(Exception):
            documents.index_document(**(options | {"chroma_collection": SimpleNamespace(delete=collection.delete, add=partial_add)}))
        previous = len(calls)
        assert hybrid.search_hybrid("needle", 14, "tenant_demo") == []
        assert len(calls) == previous
        documents.index_document(**options)
        documents.delete_document(document_id=doc.id, tenant_id="tenant_demo", chroma_collection=collection)
        assert hybrid.search_hybrid("needle", 14, "tenant_demo") == []
    finally:
        client.delete_collection(name)


def test_api_mode_audit_and_tenant_authority(runtime, monkeypatch):
    engine, _ = runtime
    good = add_document(engine, content="needle scoped")
    add_document(engine, tenant="spoofed", content="forbidden")
    logs = []
    monkeypatch.setattr(router, "create_retrieval_log_service", logs.append)
    monkeypatch.setattr(hybrid, "get_collection", lambda: SimpleNamespace(
        query=lambda **_: {"ids": [[]], "distances": [[]]}))
    with TestClient(main.app) as client:
        response = client.post("/rag/search", json={"query": "needle", "retrieval_mode": "hybrid", "tenant_id": "spoofed"})
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["top_k"] == 14 and data["retrieval_mode"] == "hybrid"
        assert data["results"][0]["document_id"] == good.id
        assert data["results"][0]["distance"] is None
        assert data["results"][0]["rerank_score"] == 1.0
        assert logs[-1].top_distance is None
        assert json.loads(logs[-1].scores_json)["score_type"] == "rerank_relevance"
        assert json.loads(logs[-1].source_documents_json)[0]["dense_rank"] is None
        assert json.loads(logs[-1].source_documents_json)[0]["bm25_rank"] == 1
        for body in ({"query": "  "}, {"query": "x", "top_k": 11},
                     {"query": "x", "retrieval_mode": "hybrid", "top_k": 15},
                     {"query": "x", "retrieval_mode": "unknown"}):
            assert client.post("/rag/search", json=body).status_code == 422


@pytest.mark.parametrize("failure", ["disabled", "provider", "incomplete", "nan"])
def test_hybrid_failure_is_sanitized_without_dense_fallback(runtime, monkeypatch, failure):
    engine, calls = runtime
    add_document(engine)
    logs = []
    monkeypatch.setattr(router, "create_retrieval_log_service", logs.append)
    monkeypatch.setattr(service, "search_chroma", lambda **_: pytest.fail("no silent Dense fallback"))
    if failure == "disabled":
        monkeypatch.setenv("RAG_HYBRID_SEARCH_ENABLED", "false")
    elif failure == "provider":
        def fail(*args, **kwargs):
            raise RuntimeError("private-provider-body secret-key")
        monkeypatch.setattr(hybrid, "rerank_candidates", fail)
    else:
        original = hybrid.rerank_candidates

        def malformed(*args, **kwargs):
            result = original(*args, **kwargs)
            if failure == "incomplete":
                return SimpleNamespace(results=[])
            from dataclasses import replace
            return replace(result, results=(replace(result.results[0], relevance_score=float("nan")),))
        monkeypatch.setattr(hybrid, "rerank_candidates", malformed)
    with TestClient(main.app) as client:
        response = client.post("/rag/search", json={"query": "VPN", "retrieval_mode": "hybrid"})
    assert response.status_code == 503
    assert logs[-1].retrieval_status == "failed"
    assert "private-provider-body" not in response.text + logs[-1].error_message
    assert "secret-key" not in response.text + logs[-1].error_message
    if failure == "disabled":
        assert calls == []
