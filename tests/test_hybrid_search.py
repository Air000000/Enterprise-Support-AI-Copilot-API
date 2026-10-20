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


def test_hybrid_ask_admits_with_null_distance_and_audits_gate(ask_runtime, monkeypatch):
    engine, _, state, ask = ask_runtime
    good = add_document(engine, content="VPN guidance with necessary conditions.")
    add_document(engine, tenant="foreign", content="Never send foreign body")
    add_document(engine, category="hr", content="Never send other category")
    monkeypatch.setattr(hybrid, "get_collection", lambda: SimpleNamespace(query=lambda **_: {"ids": [[]], "distances": [[]]}))
    with TestClient(main.app) as api:
        response = api.post("/rag/ask", json=ASK_BODY | {"tenant_id": "foreign"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["retrieval_status"] == "ok" and body["top_distance"] is None
        assert body["refusal_policy"] == ask.POLICY and body["retrieval_mode"] == "hybrid"
        assert body["evidence_gate"]["decision"] == "SUFFICIENT"
        assert body["sources"][0]["distance"] is None and body["sources"][0]["document_id"] == good.id
        assert body["sources"][0]["bm25_rank"] == 1 and body["sources"][0]["rerank_score"] == 1
        logs = api.get("/agent-ops/retrieval-logs", params={"endpoint": "ask"}).json()
        audit = json.loads(logs[0]["scores_json"])
        assert audit["refusal_policy"] == ask.POLICY and audit["evidence_gate"]["decision"] == "SUFFICIENT"
        assert audit["score_type"] == "rerank_relevance" and logs[0]["top_distance"] is None
    assert [stage for stage, _ in state["events"]] == ["gate", "generation"]
    classifier, generation = [body for _, body in state["events"]]
    assert classifier["model"] == ask.CLASSIFIER_MODEL and classifier["max_tokens"] == 2048
    assert generation["max_tokens"] == 1024
    assert generation["enable_thinking"] is False
    assert next(timeout for stage, timeout in state["timeouts"] if stage == "generation")["read"] == 30
    from experiments.evals.refusal_question_scoped_coverage import CLASSIFIER_SYSTEM_PROMPT_V5_3_DEV
    assert classifier["messages"][0]["content"] == CLASSIFIER_SYSTEM_PROMPT_V5_3_DEV
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
def test_hybrid_ask_failures_are_not_success_or_normal_refusal(ask_runtime, failure):
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
        response = api.post("/rag/ask", json=ASK_BODY)
        assert response.status_code == 503
        logs = api.get("/agent-ops/retrieval-logs").json()
        assert logs[0]["retrieval_status"] == "failed"
        assert "private-provider-body" not in response.text + logs[0]["error_message"]
        assert "secret-key" not in response.text + logs[0]["error_message"]
        assert json.loads(logs[0]["scores_json"])["refusal_policy"] == "v5_3_experimental"
    assert sum(stage == "gate" for stage, _ in state["events"]) == 1
    assert sum(stage == "generation" for stage, _ in state["events"]) == (failure not in {"bad_gate", "truncated_gate", "gate_provider"})


def test_hybrid_ask_empty_and_generator_refusal_are_distinct(ask_runtime):
    engine, calls, state, ask = ask_runtime
    with TestClient(main.app) as api:
        response = api.post("/rag/ask", json=ASK_BODY)
        assert response.status_code == 200 and response.json()["retrieval_status"] == "no_context"
        assert calls == state["events"] == []
        add_document(engine)
        state["answer"] = ask.REFUSAL
        response = api.post("/rag/ask", json=ASK_BODY)
        assert response.json()["retrieval_status"] == "refused_by_generator"
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


def test_hybrid_ask_flags_policy_and_missing_config_stop_early(ask_runtime, monkeypatch):
    _, calls, state, ask = ask_runtime
    with TestClient(main.app) as api:
        for body in (ASK_BODY | {"max_distance": 0.9}, ASK_BODY | {"top_k": 15}, ASK_BODY | {"question": "  "},
                     ASK_BODY | {"refusal_policy": None}, {"question": "x", "top_k": 11},
                     {"question": "x", "refusal_policy": "v5_3_experimental"}):
            assert api.post("/rag/ask", json=body).status_code == 422
        for flag in ("RAG_HYBRID_ASK_ENABLED", "RAG_HYBRID_SEARCH_ENABLED"):
            monkeypatch.setenv(flag, "false")
            assert api.post("/rag/ask", json=ASK_BODY).status_code == 503
            monkeypatch.setenv(flag, "true")
        monkeypatch.setattr(ask, "get_classifier_client", state["real_classifier_factory"])
        monkeypatch.delenv("DASHSCOPE_PHASE_B_API_KEY", raising=False)
        assert api.post("/rag/ask", json=ASK_BODY).status_code == 503
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


def test_nonempty_hybrid_over_real_uvicorn_tcp(ask_runtime, monkeypatch):
    """Real TCP/JWT/SQL/Chroma; external models stay mocked, not quality evidence."""
    import socket
    from threading import Thread
    import time
    import uvicorn
    from auth import get_current_user

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
    # ponytail: one OS-assigned loopback socket, not a new server/proxy framework.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(main.app, loop="asyncio", http="h11", ws="none",
            lifespan="on", log_config=None, access_log=False))
        worker = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started:
                assert worker.is_alive() and time.monotonic() < deadline, "Uvicorn startup failed"
                time.sleep(.02)
            with httpx.Client(base_url=f"http://127.0.0.1:{listener.getsockname()[1]}",
                              trust_env=False, timeout=10) as api:
                assert api.get("/health").status_code == 200
                assert api.post("/rag/ask", json=ASK_BODY).status_code == 401
                login = api.post("/auth/token", json={"username": "support", "password": "support"})
                assert login.status_code == 200
                api.headers["Authorization"] = "Bearer " + login.json()["access_token"]
                body = ASK_BODY | {"tenant_id": "tenant_other"}
                answer = api.post("/rag/ask", json=body)
                assert answer.status_code == 200 and answer.json()["retrieval_status"] == "ok"
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
                assert api.get("/agent-ops/metrics/retrieval").json()["retrieval_status_counts"] == {
                    "ok": 1, "refused_insufficient_evidence": 1, "failed": 2, "no_context": 1}
                assert "private-provider-body" not in json.dumps(logs) and "secret-key" not in json.dumps(logs)
                login = api.post("/auth/token", json={"username": "support_other", "password": "support"})
                assert login.status_code == 200
                api.headers["Authorization"] = "Bearer " + login.json()["access_token"]
                assert api.get("/agent-ops/retrieval-logs").json() == []
        finally:
            server.should_exit = True
            worker.join(timeout=10)
            chroma.delete_collection(collection.name)
            assert not worker.is_alive(), "Uvicorn did not stop before fixture teardown"
