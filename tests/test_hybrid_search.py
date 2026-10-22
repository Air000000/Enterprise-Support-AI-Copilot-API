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
        assert "ids" not in kwargs
        where = kwargs["where"]["$and"][0]
        scope = {key: value for item in where.get("$and", [where]) for key, value in item.items()}
        ids = list(hybrid.load_visible_chunks(scope["tenant_id"], scope.get("category")))[:kwargs["n_results"]]
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
        assert "ids" not in kwargs
        assert kwargs["where"] == {"$and": [
            {"$and": [{"tenant_id": "tenant_demo"}, {"category": "it"}]},
            {"source_type": {"$in": ["uploaded_document", "archival_benchmark"]}}]}
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


@pytest.mark.parametrize("limit", ["0", "200001", "invalid"])
def test_invalid_capacity_fails_before_embedding(runtime, monkeypatch, limit):
    monkeypatch.setenv("RAG_HYBRID_MAX_VISIBLE_CHUNKS", limit)
    monkeypatch.setattr(hybrid, "embed_texts", lambda _: pytest.fail("No embedding with invalid capacity"))
    with pytest.raises(ValueError):
        hybrid.search_hybrid("needle", 14, "tenant_demo")
    assert runtime[1] == []


def test_explicit_capacity_and_search_slot_release(runtime, monkeypatch):
    engine, _ = runtime
    add_document(engine, count=2)
    monkeypatch.setattr(hybrid, "MAX_VISIBLE_CHUNKS", 1)
    monkeypatch.setenv("RAG_HYBRID_MAX_VISIBLE_CHUNKS", "2")
    assert hybrid._SEARCH_SLOT.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="capacity busy"):
            hybrid.search_hybrid("needle", 14, "tenant_demo")
    finally:
        hybrid._SEARCH_SLOT.release()
    with pytest.raises(ValueError):
        hybrid.search_hybrid("needle", 15, "tenant_demo")
    assert len(hybrid.search_hybrid("needle", 14, "tenant_demo")) == 2


def test_cache_uses_fresh_contents_not_parent_timestamps(runtime):
    engine, calls = runtime
    doc = add_document(engine, content="needle before")
    hybrid._bm25_index.cache_clear()
    first = hybrid.search_hybrid("needle", 14, "tenant_demo")
    second = hybrid.search_hybrid("needle", 14, "tenant_demo")
    assert first == second
    assert hybrid._bm25_index.cache_info().hits == 1
    with Session(engine) as session:
        chunk = session.exec(select(DocumentChunk).where(DocumentChunk.document_id == doc.id)).one()
        chunk.content = "needle after direct SQL change"
        session.add(chunk)
        session.commit()  # Deliberately no parent timestamp/version update.
    assert hybrid.search_hybrid("needle", 14, "tenant_demo")[0].content == "needle after direct SQL change"
    assert hybrid._bm25_index.cache_info().misses == 2
    assert calls[-1]["documents"] == ["needle after direct SQL change"]
    assert hybrid.search_hybrid("needle", 14, "foreign") == []
    assert hybrid.search_hybrid("needle", 14, "tenant_demo", "hr") == []


def test_native_filtered_orphan_vectors_fail_closed(runtime, monkeypatch):
    engine, calls = runtime
    add_document(engine, content="needle managed")
    client = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
    collection = client.create_collection("orphan_" + uuid4().hex, embedding_function=None)
    monkeypatch.setattr(hybrid, "get_collection", lambda: collection)
    try:
        collection.add(ids=["unmanaged-leftover"], embeddings=[[1.0, 0.0]],
                       metadatas=[{"tenant_id": "tenant_demo", "category": "it", "source_type": "uploaded_document"}])
        with pytest.raises(RuntimeError, match="out-of-scope dense ranking"):
            hybrid.search_hybrid("needle", 14, "tenant_demo", "it")
        assert calls == []  # No orphan body reaches the external reranker.
    finally:
        client.delete_collection(collection.name)


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
        collection.add(ids=["same-tenant-legacy-static"], embeddings=[[1.0, 0.0]],
                       metadatas=[{"tenant_id": "tenant_demo", "category": "it"}])
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


def test_full_synthetic_smoke_flow_offline(runtime, tmp_path, monkeypatch):
    from auth import get_current_user
    from scripts.smoke_hybrid_runtime import run_flow
    import services.agent_ops_service as agent_ops

    engine, _ = runtime
    monkeypatch.setattr(agent_ops, "engine", engine)
    monkeypatch.setenv("DOCUMENT_STORAGE_ROOT", str(tmp_path / "uploads"))
    monkeypatch.setenv("AUTH_DEMO_USERS_JSON", "")
    main.app.dependency_overrides.pop(get_current_user, None)
    client = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
    collection = client.create_collection("smoke_" + uuid4().hex, embedding_function=None)
    import rag_runtime.query_chroma as dense
    monkeypatch.setattr(dense, "get_collection", lambda: collection)
    monkeypatch.setattr(hybrid, "get_collection", lambda: collection)
    monkeypatch.setattr(documents, "get_document_chroma_collection", lambda **_: collection)
    monkeypatch.setattr(dense, "embed_texts", hybrid.embed_texts)
    # index_document binds its embedding default at definition time.
    original_index = documents.index_document
    import routers.documents as document_router
    monkeypatch.setattr(document_router, "index_document_service", lambda **kwargs: original_index(
        **kwargs, embedding_function=hybrid.embed_texts))

    def response(request):
        body = json.loads(request.content)
        order = sorted(range(len(body["documents"])), key=lambda i: "VPN error 809" not in body["documents"][i])
        return httpx.Response(200, json={"results": [
            {"index": i, "relevance_score": 1 / (rank + 1)} for rank, i in enumerate(order)]})
    monkeypatch.setattr(hybrid, "get_rerank_client", lambda: OpenAI(api_key="offline",
        base_url="https://rerank.invalid/v1", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(response))))
    try:
        with TestClient(main.app) as api:
            assert run_flow(api)["status"] == "PASS"
    finally:
        client.delete_collection(collection.name)


def test_smoke_transport_bounds_real_attempts_and_synthetic_egress():
    from scripts.smoke_hybrid_runtime import BoundedTransport, LIMITS, VPN_QUERY

    records, sent = [], []

    def respond(request):
        sent.append(request)
        return httpx.Response(200, json={"usage": {"total_tokens": 5}})
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    with httpx.Client(transport=BoundedTransport("embedding", url, records, httpx.MockTransport(respond))) as client:
        for _ in range(LIMITS["embedding"]):
            assert client.post(url + "/embeddings", json={"model": "text-embedding-v4", "input": [VPN_QUERY]}).status_code == 200
        with pytest.raises(RuntimeError, match="budget"):
            client.post(url + "/embeddings", json={"model": "text-embedding-v4", "input": [VPN_QUERY]})
        with pytest.raises(RuntimeError, match="Unapproved embedding text"):
            client.post(url + "/embeddings", json={"model": "text-embedding-v4", "input": ["private data"]})
        with pytest.raises(RuntimeError, match="endpoint"):
            client.post("https://foreign.invalid/embeddings", json={})
    assert len(sent) == LIMITS["embedding"] and all(r["total_tokens"] == 5 for r in records)


def test_smoke_transport_counts_failed_sends_and_rejects_foreign_rerank():
    from scripts.smoke_hybrid_runtime import BoundedTransport, DOCS, VPN_QUERY

    records = []
    url = "https://ws-test.ap-southeast-1.maas.aliyuncs.com/compatible-api/v1"

    def fail(request):
        raise httpx.ConnectError("simulated transport failure")
    with httpx.Client(transport=BoundedTransport("rerank", url, records, httpx.MockTransport(fail))) as client:
        with pytest.raises(RuntimeError, match="Unapproved rerank text"):
            client.post(url + "/reranks", json={"model": "qwen3-rerank", "query": VPN_QUERY, "documents": [DOCS[-1][-1]]})
        assert records == []
        with pytest.raises(httpx.ConnectError):
            client.post(url + "/reranks", json={"model": "qwen3-rerank", "query": VPN_QUERY, "documents": [DOCS[0][-1]]})
    assert len(records) == 1 and records[0]["status"] is None


@pytest.fixture
def ask_runtime(runtime, monkeypatch):
    import rag_runtime.hybrid_ask as ask
    import services.agent_ops_service as agent_ops

    engine, retrieval_calls = runtime
    monkeypatch.setattr(agent_ops, "engine", engine)
    monkeypatch.setenv("RAG_HYBRID_ASK_ENABLED", "true")
    support = {"status": "SUPPORTED", "evidence": [{"source_id": "Source 1", "segment_ids": [1]}], "reason": "Fictional support."}
    state = {"gate": {"requirements": [{"requirement": "Resolve the requested problem", **support}],
        "coverage_checks": {"requested_outcomes": support, "applicability": support,
            "version_and_time": {"status": "NOT_REQUIRED", "evidence": [], "reason": "No version constraint."}},
        "unresolved_ambiguities": []}, "answer": "Follow the VPN guide. [Source 1]", "events": [], "hooks": {},
        "gate_finish": "stop", "generation_finish": "stop", "real_classifier_factory": ask.get_classifier_client}

    def client(stage):
        def respond(request):
            body = json.loads(request.content)
            state["events"].append((stage, body))
            state.setdefault("timeouts", []).append((stage, request.extensions.get("timeout")))
            if stage in state["hooks"]:
                state["hooks"][stage]()
            if state.get("error_stage") == stage:
                return httpx.Response(500, json={"error": {"message": "private-provider-body secret-key"}})
            if state.get("timeout_stage") == stage:
                raise state["timeout_kind"]("private-provider-body secret-key", request=request)
            content = json.dumps(state["gate"]) if stage == "gate" else state["answer"]
            return httpx.Response(200, json={"id": "offline", "object": "chat.completion", "model": body["model"],
                "choices": [{"index": 0, "finish_reason": state[stage + "_finish"],
                             "message": {"role": "assistant", "content": content}}]})
        return OpenAI(api_key="offline", base_url="https://chat.invalid/v1", max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(ask, "get_classifier_client", lambda: client("gate"))
    monkeypatch.setattr(ask, "get_llm_client", lambda: client("generation"))
    monkeypatch.setattr(service, "ask_rag", lambda **_: pytest.fail("Hybrid must never fall back to Dense ask"))
    yield engine, retrieval_calls, state, ask


ASK_BODY = {"question": "How to resolve VPN?", "retrieval_mode": "hybrid", "refusal_policy": "v5_3_experimental", "category": "it"}


@pytest.mark.parametrize("policy", ["v5_3_experimental", "v5_3_1_experimental"])
def test_hybrid_ask_admits_with_null_distance_and_audits_gate(ask_runtime, monkeypatch, policy):
    engine, _, state, ask = ask_runtime
    good = add_document(engine, content="VPN guidance with necessary conditions.")
    add_document(engine, tenant="foreign", content="Never send foreign body")
    add_document(engine, category="hr", content="Never send other category")
    monkeypatch.setattr(hybrid, "get_collection", lambda: SimpleNamespace(query=lambda **_: {"ids": [[]], "distances": [[]]}))
    with TestClient(main.app) as api:
        response = api.post("/rag/ask", json=ASK_BODY | {"tenant_id": "foreign", "refusal_policy": policy})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["retrieval_status"] == "ok" and body["top_distance"] is None
        assert body["refusal_policy"] == policy and body["retrieval_mode"] == "hybrid"
        assert body["evidence_gate"]["decision"] == "SUFFICIENT"
        assert body["sources"][0]["distance"] is None and body["sources"][0]["document_id"] == good.id
        assert body["sources"][0]["bm25_rank"] == 1 and body["sources"][0]["rerank_score"] == 1
        logs = api.get("/agent-ops/retrieval-logs", params={"endpoint": "ask"}).json()
        audit = json.loads(logs[0]["scores_json"])
        assert audit["refusal_policy"] == policy and audit["evidence_gate"]["decision"] == "SUFFICIENT"
        assert audit["score_type"] == "rerank_relevance" and logs[0]["top_distance"] is None
    assert [stage for stage, _ in state["events"]] == ["gate", "generation"]
    classifier, generation = [body for _, body in state["events"]]
    assert classifier["model"] == ask.CLASSIFIER_MODEL and classifier["max_tokens"] == 2048
    assert generation["max_tokens"] == 1024
    assert generation["enable_thinking"] is False
    assert next(timeout for stage, timeout in state["timeouts"] if stage == "generation")["read"] == 30
    from experiments.evals.refusal_question_scoped_coverage import CLASSIFIER_SYSTEM_PROMPT_V5_3_DEV
    from experiments.evals.refusal_version_audit_clarification import CLASSIFIER_SYSTEM_PROMPT_V5_3_1_DEV
    expected_prompt = CLASSIFIER_SYSTEM_PROMPT_V5_3_DEV if policy == ask.POLICY else CLASSIFIER_SYSTEM_PROMPT_V5_3_1_DEV
    assert classifier["messages"][0]["content"] == expected_prompt
    payload = json.loads(classifier["messages"][1]["content"])
    assert set(payload) == {"question", "sources"}
    assert payload["sources"][0]["segments"][0]["content"] == "VPN guidance with necessary conditions."
    assert "Never send" not in json.dumps(state["events"])


def test_hybrid_ask_does_not_apply_dense_distance_threshold(ask_runtime, monkeypatch):
    engine, _, _, _ = ask_runtime
    doc = add_document(engine)
    monkeypatch.setattr(hybrid, "get_collection", lambda: SimpleNamespace(query=lambda **_: {
        "ids": [[f"doc:{doc.id}:v1:chunk:0"]], "distances": [[42.0]]}))
    with TestClient(main.app) as api:
        response = api.post("/rag/ask", json=ASK_BODY)
    assert response.status_code == 200 and response.json()["retrieval_status"] == "ok"
    assert response.json()["sources"][0]["distance"] == 42.0
    assert response.json()["top_distance"] is None


@pytest.mark.parametrize("status", ["MISSING", "CONDITIONAL", "CONFLICTING", "ambiguous"])
def test_hybrid_ask_insufficient_never_generates_and_is_filterable(ask_runtime, status):
    engine, _, state, _ = ask_runtime
    add_document(engine)
    if status == "ambiguous":
        state["gate"]["unresolved_ambiguities"] = ["Unresolved material target"]
    else:
        state["gate"]["requirements"][0]["status"] = status
        if status == "MISSING":
            state["gate"]["requirements"][0]["evidence"] = []
    with TestClient(main.app) as api:
        response = api.post("/rag/ask", json=ASK_BODY)
        assert response.status_code == 200, response.text
        assert response.json()["retrieval_status"] == "refused_insufficient_evidence"
        logs = api.get("/agent-ops/retrieval-logs", params={"retrieval_status": "refused_insufficient_evidence"}).json()
        assert len(logs) == 1
        assert api.get("/agent-ops/metrics/retrieval").json()["retrieval_status_counts"] == {"refused_insufficient_evidence": 1}
    assert [stage for stage, _ in state["events"]] == ["gate"]


@pytest.mark.parametrize("failure", ["bad_gate", "truncated_gate", "gate_provider", "generation_provider", "truncated_answer", "empty_answer", "no_citation", "unknown_citation", "uncited_support"])
@pytest.mark.parametrize("policy", ["v5_3_experimental", "v5_3_1_experimental"])
def test_hybrid_ask_failures_are_not_success_or_normal_refusal(ask_runtime, failure, policy):
    engine, _, state, _ = ask_runtime
    add_document(engine)
    add_document(engine)  # Source 2 is present but not admitted by the gate.
    if failure == "bad_gate":
        state["gate"]["requirements"][0]["evidence"] = [{"source_id": "Source 1", "segment_ids": [999]}]
    elif failure == "truncated_gate":
        state["gate_finish"] = "length"
    elif failure in {"gate_provider", "generation_provider"}:
        state["error_stage"] = failure.removesuffix("_provider")
    elif failure == "truncated_answer":
        state["generation_finish"] = "length"
    else:
        state["answer"] = {"empty_answer": "", "no_citation": "Unsupported answer", "unknown_citation": "Try it. [Source 99]",
                           "uncited_support": "Try it. [Source 2]"}[failure]
    with TestClient(main.app) as api:
        response = api.post("/rag/ask", json=ASK_BODY | {"refusal_policy": policy})
        assert response.status_code == 503
        logs = api.get("/agent-ops/retrieval-logs").json()
        assert logs[0]["retrieval_status"] == "failed"
        assert "private-provider-body" not in response.text + logs[0]["error_message"]
        assert "secret-key" not in response.text + logs[0]["error_message"]
        assert json.loads(logs[0]["scores_json"])["refusal_policy"] == policy
    assert sum(stage == "gate" for stage, _ in state["events"]) == 1
    assert sum(stage == "generation" for stage, _ in state["events"]) == (failure not in {"bad_gate", "truncated_gate", "gate_provider"})


@pytest.mark.parametrize("policy", ["v5_3_experimental", "v5_3_1_experimental"])
def test_hybrid_ask_empty_and_generator_refusal_are_distinct(ask_runtime, policy):
    engine, calls, state, ask = ask_runtime
    with TestClient(main.app) as api:
        response = api.post("/rag/ask", json=ASK_BODY | {"refusal_policy": policy})
        assert response.status_code == 200 and response.json()["retrieval_status"] == "no_context"
        assert response.json()["refusal_policy"] == policy
        assert calls == state["events"] == []
        add_document(engine)
        state["answer"] = ask.REFUSAL
        response = api.post("/rag/ask", json=ASK_BODY | {"refusal_policy": policy})
        assert response.json()["retrieval_status"] == "refused_by_generator"
        assert response.json()["refusal_policy"] == policy
        assert response.json()["evidence_gate"]["decision"] == "SUFFICIENT"


@pytest.mark.parametrize("stage", ["before_gate", "gate", "generation"])
def test_hybrid_ask_doc_mutation_blocks_stale_answer(ask_runtime, monkeypatch, stage):
    engine, _, state, ask = ask_runtime
    doc = add_document(engine)

    def mutate():
        with Session(engine) as session:
            row = session.get(Document, doc.id)
            row.status = "deleted"
            session.add(row)
            session.commit()
    if stage == "before_gate":
        search = ask.search_hybrid

        def changed(*args):
            result = search(*args)
            mutate()
            return result
        monkeypatch.setattr(ask, "search_hybrid", changed)
    else:
        state["hooks"][stage] = mutate
    with TestClient(main.app) as api:
        assert api.post("/rag/ask", json=ASK_BODY).status_code == 503
    assert len(state["events"]) == {"before_gate": 0, "gate": 1, "generation": 2}[stage]


@pytest.mark.parametrize("policy", ["v5_3_experimental", "v5_3_1_experimental"])
def test_hybrid_ask_flags_policy_and_missing_config_stop_early(ask_runtime, monkeypatch, policy):
    _, calls, state, ask = ask_runtime
    body = ASK_BODY | {"refusal_policy": policy}
    with TestClient(main.app) as api:
        for invalid in (body | {"max_distance": 0.9}, body | {"top_k": 15}, body | {"question": "  "},
                        body | {"refusal_policy": None}, body | {"refusal_policy": "unknown"},
                        {"question": "x", "top_k": 11}, {"question": "x", "refusal_policy": policy}):
            assert api.post("/rag/ask", json=invalid).status_code == 422
        for flag in ("RAG_HYBRID_ASK_ENABLED", "RAG_HYBRID_SEARCH_ENABLED"):
            monkeypatch.setenv(flag, "false")
            assert api.post("/rag/ask", json=body).status_code == 503
            monkeypatch.setenv(flag, "true")
        monkeypatch.setattr(ask, "get_classifier_client", state["real_classifier_factory"])
        monkeypatch.delenv("DASHSCOPE_PHASE_B_API_KEY", raising=False)
        assert api.post("/rag/ask", json=body).status_code == 503
    with pytest.raises(ValueError, match="Unknown experimental refusal policy"):
        ask.ask_hybrid("x", 14, "tenant_demo", refusal_policy="unknown")
    assert calls == state["events"] == []


def test_ask_acceptance_guard_rejects_labels_and_foreign_text():
    import copy
    from experiments.evals.refusal_evidence_sufficiency import ClassifierInput, ContextSource
    from experiments.evals.refusal_question_scoped_coverage import build_classifier_messages_v5_3_dev
    from scripts import hybrid_ask_cases as cases
    from scripts.smoke_hybrid_runtime import BoundedTransport

    question, text = cases.CASES[0][1], cases.DOCS[0][-1]
    body = {"model": "qwen3.5-plus-2026-04-20", "temperature": 0, "max_tokens": 2048,
            "response_format": {"type": "json_object"}, "enable_thinking": False,
            "messages": build_classifier_messages_v5_3_dev(ClassifierInput(question, (ContextSource("Source 1", "c", "d", text),)))}
    sent, records = [], []

    def respond(request):
        sent.append(request)
        return httpx.Response(200, json={"usage": {"total_tokens": 6, "prompt_tokens": 5, "completion_tokens": 1},
            "choices": [{"finish_reason": "stop", "message": {"content": "{}", "reasoning_content": "PRIVATE_REASONING"}}]})
    url = "https://ws-offline.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    with httpx.Client(transport=BoundedTransport("classifier", url, records, httpx.MockTransport(respond), ask=True)) as client:
        for bad in ("labels", "foreign", "extra_field"):
            payload = copy.deepcopy(body)
            content = json.loads(payload["messages"][1]["content"])
            if bad == "labels":
                content["expected_status"] = "ok"
            elif bad == "foreign":
                content["sources"][0]["segments"][0]["content"] = cases.DOCS[-1][-1]
            else:
                payload["gold_answer"] = "private"
            payload["messages"][1]["content"] = json.dumps(content)
            with pytest.raises(RuntimeError, match="Unapproved"):
                client.post(url + "/chat/completions", json=payload)
        assert sent == records == []
        for _ in range(cases.LIMITS["classifier"]):
            assert client.post(url + "/chat/completions", json=body).status_code == 200
        with pytest.raises(RuntimeError, match="budget"):
            client.post(url + "/chat/completions", json=body)
    assert len(sent) == len(records) == 6
    assert all(r["finish_reason"] == "stop" and r["prompt_tokens"] == 5 and r["completion_tokens"] == 1
               and r["transport_phase"] == "complete" and not r["error_types"] for r in records)
    assert "PRIVATE_REASONING" not in json.dumps(records)


@pytest.mark.parametrize("model, disabled", [("qwen3.5-plus", True), ("qwen3.5-plus-2026-02-15", True), ("other-model", False)])
def test_generation_thinking_switch_is_model_scoped(ask_runtime, monkeypatch, model, disabled):
    engine, _, state, _ = ask_runtime
    add_document(engine)
    monkeypatch.setenv("CHAT_MODEL", model)
    with TestClient(main.app) as api:
        assert api.post("/rag/ask", json=ASK_BODY).status_code == 200
    body = next(body for stage, body in state["events"] if stage == "generation")
    assert body["model"] == model
    assert (body.get("enable_thinking") is False) == disabled
    assert body["max_tokens"] == 1024


@pytest.mark.parametrize("stage", ["gate", "generation"])
@pytest.mark.parametrize("timeout_kind", [httpx.ReadTimeout, httpx.ConnectTimeout])
def test_timeout_audit_keeps_stage_and_cause_without_error_text(ask_runtime, stage, timeout_kind):
    engine, _, state, _ = ask_runtime
    add_document(engine)
    state.update(timeout_stage=stage, timeout_kind=timeout_kind)
    with TestClient(main.app) as api:
        response = api.post("/rag/ask", json=ASK_BODY)
        assert response.status_code == 503
        log = api.get("/agent-ops/retrieval-logs").json()[0]
    details = json.loads(log["scores_json"])
    assert details["error_stage"] == ("classification" if stage == "gate" else "generation")
    assert any(name.endswith(".APITimeoutError") for name in details["error_types"])
    assert any(name.endswith("." + timeout_kind.__name__) for name in details["error_types"])
    assert log["retrieval_status"] == "failed"
    assert "private-provider-body" not in json.dumps(log) + response.text
    assert "secret-key" not in json.dumps(log) + response.text
    assert sum(s == stage for s, _ in state["events"]) == 1  # No automatic retry.


def test_safe_error_details_bound_cycles_and_do_not_copy_notes():
    from rag_runtime.hybrid_ask import safe_error_details

    error = RuntimeError("private-provider-body")
    cause = ValueError("secret-key")
    error.__cause__ = cause
    cause.__cause__ = error
    error.add_note("hybrid_stage:generation?secret-key")
    result = safe_error_details(error)
    assert result == {"error_stage": "service", "error_types": ["builtins.RuntimeError", "builtins.ValueError"]}
    for _ in range(10):
        previous = error
        error = RuntimeError("private")
        error.__cause__ = previous
    assert len(safe_error_details(error)["error_types"]) == 6


@pytest.mark.parametrize("phase", ["awaiting_headers", "reading_body"])
def test_transport_distinguishes_header_and_body_timeout(phase):
    from scripts.smoke_hybrid_runtime import BoundedTransport, VPN_QUERY

    class BrokenBody(httpx.SyncByteStream):
        def __iter__(self):
            yield b"partial"
            raise httpx.ReadTimeout("private-provider-body secret-key")

    def respond(request):
        if phase == "awaiting_headers":
            raise httpx.ReadTimeout("private-provider-body secret-key", request=request)
        return httpx.Response(200, stream=BrokenBody())

    url, records = "https://dashscope.aliyuncs.com/compatible-mode/v1", []
    with httpx.Client(transport=BoundedTransport("embedding", url, records, httpx.MockTransport(respond))) as client:
        with pytest.raises(httpx.ReadTimeout):
            client.post(url + "/embeddings", json={"model": "text-embedding-v4", "input": [VPN_QUERY]})
    assert len(records) == 1
    row = records[0]
    assert row["transport_phase"] == phase
    assert row["status"] == (200 if phase == "reading_body" else None)
    assert row["total_tokens"] is None and row["error_types"] == ["httpx.ReadTimeout"]
    assert "private-provider-body" not in json.dumps(records) and "secret-key" not in json.dumps(records)


@pytest.mark.parametrize("policy", ["v5_3_experimental", "v5_3_1_experimental"])
def test_nonempty_hybrid_over_real_uvicorn_tcp(ask_runtime, monkeypatch, policy):
    """Real TCP/JWT/SQL/Chroma; external models stay mocked, not quality evidence."""
    from auth import get_current_user
    from scripts.smoke_hybrid_runtime import live_http_client

    engine, rerank_calls, state, _ = ask_runtime
    own = add_document(engine, content="VPN guide: reconnect the client.")
    foreign = add_document(engine, tenant="tenant_other", content="Never send foreign body")
    hr = add_document(engine, category="hr", content="Never send other category")
    chroma = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
    collection = chroma.create_collection("tcp_" + uuid4().hex, embedding_function=None)
    for doc in (own, foreign, hr):
        collection.add(ids=[f"doc:{doc.id}:v1:chunk:0"], embeddings=[[1.0, 0.0]],
            documents=["VPN guide: reconnect the client." if doc.id == own.id else "Never send"],
            metadatas=[{"tenant_id": doc.tenant_id, "category": doc.category}])
    monkeypatch.setattr(hybrid, "get_collection", lambda: collection)
    monkeypatch.delitem(main.app.dependency_overrides, get_current_user)  # Exercise actual JWT dependency.
    try:
        with live_http_client(main.app) as api:
            assert api.get("/health").status_code == 200
            assert api.post("/rag/ask", json=ASK_BODY).status_code == 401
            login = api.post("/auth/token", json={"username": "support", "password": "support"})
            assert login.status_code == 200
            api.headers["Authorization"] = "Bearer " + login.json()["access_token"]
            body = ASK_BODY | {"tenant_id": "tenant_other", "refusal_policy": policy}
            answer = api.post("/rag/ask", json=body)
            assert answer.status_code == 200 and answer.json()["retrieval_status"] == "ok"
            assert answer.json()["refusal_policy"] == policy
            assert answer.json()["answer"] == state["answer"]
            assert [s["document_id"] for s in answer.json()["sources"]] == [own.id]
            assert [stage for stage, _ in state["events"]] == ["gate", "generation"]
            assert "Never send" not in json.dumps(state["events"]) + json.dumps(rerank_calls)

            state["gate"]["requirements"][0].update(status="MISSING", evidence=[])
            refused = api.post("/rag/ask", json=body)
            assert refused.status_code == 200
            assert refused.json()["retrieval_status"] == "refused_insufficient_evidence"
            assert [stage for stage, _ in state["events"]] == ["gate", "generation", "gate"]
            state["error_stage"] = "gate"
            failed = api.post("/rag/ask", json=body)
            assert failed.status_code == 503 and "private-provider-body" not in failed.text
            assert [stage for stage, _ in state["events"]] == ["gate", "generation", "gate", "gate"]
            before = (len(rerank_calls), len(state["events"]))
            empty = api.post("/rag/ask", json=body | {"category": "other"})
            assert empty.status_code == 200 and empty.json()["retrieval_status"] == "no_context"
            monkeypatch.setenv("RAG_HYBRID_ASK_ENABLED", "false")
            assert api.post("/rag/ask", json=body).status_code == 503
            assert before == (len(rerank_calls), len(state["events"]))
            logs = api.get("/agent-ops/retrieval-logs").json()
            assert len(logs) == 5 and all(row["tenant_id"] == "tenant_demo" for row in logs)
            assert all(json.loads(row["scores_json"])["refusal_policy"] == policy for row in logs)
            assert api.get("/agent-ops/metrics/retrieval").json()["retrieval_status_counts"] == {
                "ok": 1, "refused_insufficient_evidence": 1, "failed": 2, "no_context": 1}
            assert "private-provider-body" not in json.dumps(logs) and "secret-key" not in json.dumps(logs)
            login = api.post("/auth/token", json={"username": "support_other", "password": "support"})
            assert login.status_code == 200
            api.headers["Authorization"] = "Bearer " + login.json()["access_token"]
            assert api.get("/agent-ops/retrieval-logs").json() == []
    finally:
        chroma.delete_collection(collection.name)


def test_live_http_cli_is_plan_only_without_execute():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-B", "-X", "utf8", "-m", "scripts.smoke_hybrid_runtime", "--ask",
                             "--live-http", "--env-file", "-"], input="not credentials", text=True,
                            encoding="utf-8", capture_output=True, timeout=30)
    assert result.returncode == 0
    plan = json.loads(result.stdout)
    assert plan["mode"] == "PLAN_ONLY" and plan["http_transport"] == "uvicorn_loopback_tcp"
    assert sum(plan["limits"].values()) == 28 and len(plan["documents"]) == 7


def test_live_http_cli_rejects_unselected_stdin_settings_without_leaking_values():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-B", "-X", "utf8", "-m", "scripts.smoke_hybrid_runtime", "--execute",
                             "--ask", "--live-http", "--env-file", "-"],
                            input=json.dumps({"BUSINESS_SETTING": "private-key-must-not-print"}),
                            text=True, encoding="utf-8", capture_output=True, timeout=30)
    assert result.returncode == 1 and "Smoke failed (RuntimeError)" in result.stderr
    assert "private-key-must-not-print" not in result.stdout + result.stderr


@pytest.mark.parametrize("category", ["it", "other"])
def test_ticket_preview_reuses_hybrid_and_preserves_confirm_boundary(runtime, monkeypatch, category):
    import services.agent_ops_service as ops
    import services.ticket_agent_service as agent
    import services.ticket_service as tickets
    from auth import get_current_user
    from models.agent_ops import AgentRun, ApprovalRequest, ToolCall
    from models.ticket import Ticket

    engine, calls = runtime
    monkeypatch.setattr(ops, "engine", engine)
    monkeypatch.setattr(tickets, "engine", engine)
    monkeypatch.setattr(agent, "search_chroma", lambda **_: pytest.fail("No implicit Dense fallback"))
    monkeypatch.setenv("RAG_HYBRID_ASK_ENABLED", "false")
    monkeypatch.delenv("DASHSCOPE_PHASE_B_API_KEY", raising=False)
    monkeypatch.delitem(main.app.dependency_overrides, get_current_user)
    own = add_document(engine, content="VPN needle authorized guide")
    add_document(engine, tenant="tenant_other", content="FORBIDDEN foreign VPN")
    if category == "it":
        add_document(engine, category="hr", content="FORBIDDEN other-category VPN")
    monkeypatch.setattr(hybrid, "get_collection", lambda: SimpleNamespace(
        query=lambda **_: {"ids": [[]], "distances": [[]]}))  # Exercise BM25-only null distance.
    with TestClient(main.app) as api:
        login = api.post("/auth/token", json={"username": "support", "password": "support"})
        assert login.status_code == 200
        api.headers["Authorization"] = "Bearer " + login.json()["access_token"]
        response = api.post("/agent/ticket/preview", json={"message": "VPN 连不上", "category": category,
            "retrieval_mode": "hybrid", "tenant_id": "tenant_other"})
        assert response.status_code == 200, response.text
        preview = response.json()
        assert preview["retrieval_mode"] == "hybrid" and preview["should_create_ticket"] is True
        assert [s["document_id"] for s in preview["sources"]] == [own.id]
        assert preview["sources"][0]["distance"] is None and preview["sources"][0]["rerank_score"] == 1
        assert preview["sources"][0]["dense_rank"] is None and preview["sources"][0]["bm25_rank"] == 1
        assert len(calls) == 1 and calls[0]["documents"] == ["VPN needle authorized guide"]
        with Session(engine) as session:
            run = session.get(AgentRun, preview["agent_run_id"])
            summary = json.loads(run.retrieval_summary_json)
            assert run.tenant_id == "tenant_demo" and run.user_id == "support_demo"
            assert summary["retrieval_mode"] == "hybrid"
            assert summary["top_k"] == 14 and summary["top_distance"] is None
            assert summary["rag_category"] == ("it" if category == "it" else None)
            assert summary["score_type"] == "rerank_relevance" and summary["rerank_scores"] == [1]
            assert session.exec(select(Ticket)).all() == []
            approval = session.get(ApprovalRequest, preview["approval_request_id"])
            assert approval.status == "pending" and json.loads(approval.draft_json) == preview["draft"]
            tools = session.exec(select(ToolCall)).all()
            assert {t.tool_name for t in tools} == {"search_kb", "classify_ticket"}
            search = next(t for t in tools if t.tool_name == "search_kb")
            assert json.loads(search.tool_input_json)["retrieval_mode"] == "hybrid"
            output = json.loads(search.tool_output_json)
            assert output["top_distance"] is None and output["rerank_scores"] == [1]
        confirm = {key: preview[key] for key in ("agent_run_id", "approval_request_id", "draft")}
        foreign = api.post("/auth/token", json={"username": "support_other", "password": "support"})
        assert foreign.status_code == 200
        assert api.post("/agent/ticket/confirm", json=confirm, headers={
            "Authorization": "Bearer " + foreign.json()["access_token"]}).status_code == 404
        assert api.post("/agent/ticket/confirm", json=confirm | {
            "draft": preview["draft"] | {"title": "tampered"}}).status_code == 400
        assert api.post("/agent/ticket/confirm", json=confirm).status_code == 201  # Temporary test DB only.
        assert api.post("/agent/ticket/confirm", json=confirm).status_code == 400
    with Session(engine) as session:
        assert len(session.exec(select(Ticket)).all()) == 1
        assert len(session.exec(select(ToolCall).where(ToolCall.tool_name == "create_ticket")).all()) == 1
    assert len(calls) == 1  # Confirm never reruns retrieval.


@pytest.mark.parametrize("decision", ["confirm", "reject", "cancel"])
def test_ticket_hybrid_document_to_decision_over_real_http(runtime, tmp_path, monkeypatch, decision):
    """Real HTTP/JWT/document indexing/Chroma/AgentOps; model transports stay offline."""
    from auth import get_current_user
    from scripts.smoke_hybrid_runtime import live_http_client
    import routers.documents as document_router
    import services.agent_ops_service as ops
    import services.ticket_agent_service as agent
    import services.ticket_service as tickets

    engine, calls = runtime
    monkeypatch.setattr(ops, "engine", engine)
    monkeypatch.setattr(tickets, "engine", engine)
    monkeypatch.setattr(agent, "search_chroma", lambda **_: pytest.fail("No implicit Dense fallback"))
    monkeypatch.setenv("DOCUMENT_STORAGE_ROOT", str(tmp_path / "uploads"))
    monkeypatch.setenv("RAG_HYBRID_ASK_ENABLED", "false")
    monkeypatch.delitem(main.app.dependency_overrides, get_current_user)
    chroma = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
    collection = chroma.create_collection("ticket_tcp_" + uuid4().hex, embedding_function=None)
    monkeypatch.setattr(hybrid, "get_collection", lambda: collection)
    monkeypatch.setattr(document_router, "index_document_service", lambda **kwargs: documents.index_document(
        **kwargs, embedding_function=hybrid.embed_texts, chroma_collection=collection))
    try:
        with live_http_client(main.app) as api:
            body = {"message": "VPN 连不上", "category": "it", "retrieval_mode": "hybrid",
                    "tenant_id": "tenant_other"}
            assert api.post("/agent/ticket/preview", json=body).status_code == 401
            headers = {}
            for username in ("support", "support_other", "employee"):
                login = api.post("/auth/token", json={"username": username,
                    "password": "employee" if username == "employee" else "support"})
                assert login.status_code == 200
                headers[username] = {"Authorization": "Bearer " + login.json()["access_token"]}
            own = None
            for username, category, text in (("support", "it", "VPN needle authorized guide"),
                    ("support", "hr", "FORBIDDEN other-category VPN"),
                    ("support_other", "it", "FORBIDDEN foreign VPN")):
                upload = api.post("/documents/upload", files={"file": ("vpn.md", text.encode(), "text/markdown")},
                    data={"category": category}, headers=headers[username])
                assert upload.status_code == 201, upload.text
                doc_id = upload.json()["id"]
                indexed = api.post(f"/documents/{doc_id}/index", headers=headers[username])
                assert indexed.status_code == 200 and indexed.json()["status"] == "indexed"
                if username == "support" and category == "it":
                    own = doc_id
            assert collection.count() == 3
            api.headers.update(headers["support"])
            response = api.post("/agent/ticket/preview", json=body)
            assert response.status_code == 200, response.text
            preview = response.json()
            assert preview["should_create_ticket"] and preview["retrieval_mode"] == "hybrid"
            assert [s["document_id"] for s in preview["sources"]] == [own]
            source = preview["sources"][0]
            assert source["dense_rank"] == source["bm25_rank"] == 1
            assert source["rerank_score"] == 1 and source["distance"] is not None
            assert len(calls) == 1 and calls[0]["documents"] == ["VPN needle authorized guide"]
            assert api.get("/tickets").json() == []
            run_id, approval_id = preview["agent_run_id"], preview["approval_request_id"]
            trace_url = f"/agent-ops/runs/{run_id}/trace"
            trace_response = api.get(trace_url)
            assert trace_response.status_code == 200
            trace = trace_response.json()
            assert trace["agent_run"]["tenant_id"] == "tenant_demo"
            summary = json.loads(trace["agent_run"]["retrieval_summary_json"])
            assert summary["retrieval_mode"] == "hybrid" and summary["top_k"] == 14
            assert summary["top_distance"] is None and summary["rerank_scores"] == [1]
            assert {t["tool_name"] for t in trace["tool_calls"]} == {"search_kb", "classify_ticket"}
            search = next(t for t in trace["tool_calls"] if t["tool_name"] == "search_kb")
            output = json.loads(search["tool_output_json"])
            assert output["score_type"] == "rerank_relevance" and output["larger_is_better"] is True
            assert output["top_distance"] is None and output["rerank_scores"] == [1]
            assert trace["approval_requests"][0]["status"] == "pending"
            assert json.loads(trace["approval_requests"][0]["draft_json"]) == preview["draft"]
            assert api.get(trace_url, headers=headers["support_other"]).status_code == 404
            assert api.get(trace_url, headers=headers["employee"]).status_code == 403
            confirm = {key: preview[key] for key in ("agent_run_id", "approval_request_id", "draft")}
            if decision == "confirm":
                result = api.post("/agent/ticket/confirm", json=confirm)
                assert result.status_code == 201, result.text  # Disposable database only.
                ticket = result.json()["ticket"]
                assert ticket["tenant_id"] == "tenant_demo" and ticket["created_by"] == "support_demo"
                assert all(ticket[key] == value for key, value in preview["draft"].items())
                assert api.get(f"/tickets/{ticket['id']}", headers=headers["support_other"]).status_code == 404
                expected_status = "approved"
            else:
                url = f"/agent-ops/approval-requests/{approval_id}/{decision}"
                assert api.post(url, json={"reason": "Fictional test decision"},
                    headers=headers["support_other"]).status_code == 404
                result = api.post(url, json={"reason": "Fictional test decision"})
                assert result.status_code == 200, result.text
                expected_status = "rejected" if decision == "reject" else "cancelled"
                assert result.json()["decision_reason"] == "Fictional test decision"
                assert result.json()["approved_by"] == "support_demo"
                assert api.post(url, json={"reason": "Repeated decision"}).status_code == 400
            assert api.post("/agent/ticket/confirm", json=confirm).status_code == 400
            final = api.get(trace_url).json()
            assert final["approval_requests"][0]["status"] == expected_status
            assert final["approval_requests"][0]["decided_at"] is not None
            assert all(t["status"] == "success" for t in final["tool_calls"])
            assert sum(t["tool_name"] == "create_ticket" for t in final["tool_calls"]) == (decision == "confirm")
            assert len(api.get("/tickets").json()) == (decision == "confirm")
            metrics = api.get("/agent-ops/metrics/summary").json()
            assert metrics["total_agent_runs"] == metrics["total_approval_requests"] == 1
            assert metrics[f"{expected_status}_approval_requests"] == 1
            assert metrics["pending_approval_requests"] == metrics["failed_tool_calls"] == 0
            assert metrics["total_tool_calls"] == (3 if decision == "confirm" else 2)
            assert api.get("/agent-ops/runs", headers=headers["support_other"]).json() == []
            assert api.get("/tickets", headers=headers["support_other"]).json() == []
            assert api.get("/agent-ops/metrics/summary", headers=headers["support_other"]).json()["total_agent_runs"] == 0
            assert len(calls) == 1  # Decisions and read APIs never re-run retrieval.
    finally:
        chroma.delete_collection(collection.name)


@pytest.mark.parametrize("message_length", [600, 2000])
def test_ticket_hybrid_top14_keeps_full_message_within_draft_limit(runtime, monkeypatch, message_length):
    import services.agent_ops_service as ops
    from models.agent_ops import AgentRun, ApprovalRequest, ToolCall
    from models.ticket import Ticket

    engine, calls = runtime
    monkeypatch.setattr(ops, "engine", engine)
    expected_ids = {add_document(engine, content="VPN needle guide").id for _ in range(14)}
    message = "VPN 连不上 " + "请协助排查。" * 334
    message = message[:message_length]
    with TestClient(main.app) as api:
        response = api.post("/agent/ticket/preview", json={"message": message, "category": "it",
            "retrieval_mode": "hybrid"})
    assert response.status_code == 200, response.text
    preview = response.json()
    assert preview["should_create_ticket"] and preview["draft"]["description"] == message
    assert len(preview["draft"]["description"]) == message_length
    assert {s["document_id"] for s in preview["sources"]} == expected_ids
    assert len(calls) == 1 and len(calls[0]["documents"]) == 14
    with Session(engine) as session:
        run = session.get(AgentRun, preview["agent_run_id"])
        assert run.status == "completed"
        assert set(json.loads(run.retrieval_summary_json)["source_document_ids"]) == expected_ids
        approval = session.get(ApprovalRequest, preview["approval_request_id"])
        assert approval.status == "pending" and json.loads(approval.draft_json) == preview["draft"]
        assert all(t.status == "success" for t in session.exec(select(ToolCall)).all())
        assert session.exec(select(Ticket)).all() == []


@pytest.mark.parametrize("failure", ["disabled", "provider"])
def test_ticket_hybrid_failure_is_audited_without_approval_or_fallback(runtime, monkeypatch, failure):
    import services.agent_ops_service as ops
    import services.ticket_agent_service as agent
    from models.agent_ops import AgentRun, ApprovalRequest, ToolCall
    from models.ticket import Ticket

    engine, calls = runtime
    monkeypatch.setattr(ops, "engine", engine)
    monkeypatch.setattr(agent, "search_chroma", lambda **_: pytest.fail("No implicit Dense fallback"))
    add_document(engine)
    if failure == "disabled":
        monkeypatch.setenv("RAG_HYBRID_SEARCH_ENABLED", "false")
    else:
        def fail(request):
            calls.append("one failed provider attempt")
            return httpx.Response(500, json={"error": {"message": "private-provider-body secret-key"}})
        monkeypatch.setattr(hybrid, "get_rerank_client", lambda: OpenAI(
            api_key="offline", base_url="https://rerank.invalid/v1", max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(fail))))
    with TestClient(main.app) as api:
        assert api.post("/agent/ticket/preview", json={"message": "VPN", "retrieval_mode": "unknown"}).status_code == 422
        response = api.post("/agent/ticket/preview", json={"message": "VPN", "retrieval_mode": "hybrid"})
        assert response.status_code == 503 and "secret-key" not in response.text
    with Session(engine) as session:
        run = session.exec(select(AgentRun)).one()
        tool = session.exec(select(ToolCall)).one()
        assert run.status == tool.status == "failed" and tool.tool_name == "search_kb"
        assert json.loads(tool.tool_input_json)["retrieval_mode"] == "hybrid"
        assert "private-provider-body" not in run.result_summary + tool.error_message
        assert "secret-key" not in run.result_summary + tool.error_message
        assert session.exec(select(ApprovalRequest)).all() == session.exec(select(Ticket)).all() == []
    assert len(calls) == (0 if failure == "disabled" else 1)
