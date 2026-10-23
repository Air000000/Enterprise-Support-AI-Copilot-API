"""Synthetic only: real TCP/JWT/service/SDK, no dataset reads or external sockets."""

from decimal import Decimal
import json
from types import SimpleNamespace

import httpx
import pytest
from sqlmodel import SQLModel, Session, create_engine

from experiments.evals import runtime_qa as qa, runtime_qa_runner as run
from models.document import Document, DocumentChunk


def gate():
    supported = dict(status="SUPPORTED", evidence=[dict(source_id="Source 1", segment_ids=[1])], reason="reset step")
    return dict(requirements=[dict(requirement="reset", **supported)],
        coverage_checks=dict(requested_outcomes=supported, applicability=supported,
            version_and_time=dict(status="NOT_REQUIRED", evidence=[], reason="no version requested")), unresolved_ambiguities=[])


def providers(seen, failure=None):
    def send(stage, request):
        body = json.loads(request.content)
        seen.append((stage, body))
        if failure == "timeout":
            raise httpx.ReadTimeout("PRIVATE_NEVER_LOG")
        if stage == "embedding":
            value = dict(data=[dict(index=0, embedding=[1.]*1024)], usage=dict(prompt_tokens=5, total_tokens=5))
        elif stage == "rerank":
            value = dict(results=[dict(index=i, relevance_score=1/(i+1)) for i in range(len(body["documents"]))], usage=dict(total_tokens=10))
        else:
            content = json.dumps({} if failure == "schema" else gate()) if stage == "classifier" else "Press reset. [Source 1]"
            value = dict(id="synthetic", model=body["model"], choices=[dict(index=0, finish_reason="stop", message=dict(role="assistant", content=content))],
                         usage=dict(prompt_tokens=20, completion_tokens=10, total_tokens=30))
        return httpx.Response(200, json=value)
    from functools import partial
    return {s: httpx.MockTransport(partial(send, s)) for s in qa.ENDPOINTS}


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    from auth import get_current_user
    from main import app
    from rag_runtime import hybrid_search as hybrid
    from scripts.smoke_hybrid_runtime import live_http_client
    engine = create_engine("sqlite:///"+str(tmp_path / "synthetic.db"))
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(hybrid, "engine", engine)
    monkeypatch.setenv("RAG_HYBRID_SEARCH_ENABLED", "true")
    monkeypatch.setenv("RAG_HYBRID_ASK_ENABLED", "true")
    monkeypatch.setenv("CHAT_MODEL", "qwen3.5-plus")
    hybrid._bm25_index.cache_clear()
    with Session(engine) as db:
        for name, tenant in (("a", run.TENANT), ("b", run.TENANT), ("foreign", "another-tenant")):
            db.add(Document(id=name, tenant_id=tenant, uploaded_by="synthetic", filename=name+".md", file_type="md",
                source_path="synthetic/"+name+".md", checksum="fake", category="other", status="indexed", chunk_count=1))
            db.add(DocumentChunk(document_id=name, tenant_id=tenant, category="other", chunk_index=0,
                content="Press reset." if name == "a" else "Status is ready.", embedding_id=f"doc:{name}:v1:chunk:0"))
        db.commit()
    queries = []
    def query(**kwargs):
        queries.append(kwargs)
        assert run.TENANT in json.dumps(kwargs["where"]) and "another-tenant" not in json.dumps(kwargs)
        ids = ["doc:a:v1:chunk:0", "doc:b:v1:chunk:0"]
        return dict(ids=[ids], distances=[[.1, .2]], documents=[["Press reset.", "Status is ready."]],
            metadatas=[[dict(document_id=name, title=name, source_path="synthetic/"+name+".md", chunk_index=0,
                            tenant_id=run.TENANT, category="other") for name in ("a", "b")]])
    override = app.dependency_overrides.pop(get_current_user)
    try:
        with run.no_provider_fallback(), live_http_client(app) as client:
            assert client.post("/rag/ask", json={"question": "reset?"}).status_code == 401
            yield client, SimpleNamespace(query=query), queries
    finally:
        app.dependency_overrides[get_current_user] = override
        hybrid._bm25_index.cache_clear()
        engine.dispose()


def test_http_pair_and_mid_case_resume_never_requery_or_resend(runtime, tmp_path, monkeypatch):
    client, collection, queries = runtime
    seen, keys = [], {s: "SYNTHETIC_ONLY_KEY" for s in qa.ENDPOINTS}
    row = dict(id="DEV_SYNTH", question="How to reset?", stratum="answerable")
    original = run.keep
    crashed = []
    def crash_after_paid(path, value):
        if path.name == "result.json" and not crashed:
            crashed.append(True)
            raise OSError("synthetic lost HTTP result before checkpoint")
        return original(path, value)
    monkeypatch.setattr(run, "keep", crash_after_paid)
    with pytest.raises(OSError):
        run.http_case(client, row, "H", tmp_path / "H", "OFFLINE_ONLY", run.Book(), keys, collection, delegates=providers(seen))
    assert len(seen) == 4 and len(queries) == 1
    monkeypatch.setattr(run, "keep", original)
    restored = run.http_case(client, row, "H", tmp_path / "H", "OFFLINE_ONLY", run.Book(), keys, collection, delegates=providers(seen))
    assert restored["http_status"] == 200, restored
    assert restored["reconstructed_after_interruption"] and restored["latency_ms"] is None
    assert len(seen) == 4 and len(queries) == 1
    dense = run.http_case(client, row, "D", tmp_path / "D", "OFFLINE_ONLY", run.Book(), keys, collection, delegates=providers(seen))
    assert dense["body"]["answer"] == restored["body"]["answer"]
    assert len(seen) == 6 and len(queries) == 2
    from rag_runtime import query_rag_chroma
    with pytest.raises(RuntimeError):
        query_rag_chroma.get_llm_client()  # A late worker cannot fall back to ordinary env clients.
    assert all("stratum" not in json.dumps(body) and "another-tenant" not in json.dumps(body) for _, body in seen)
    assert [s for s, _ in seen] == ["embedding", "rerank", "classifier", "generation", "embedding", "generation"]
    assert all("SYNTHETIC_ONLY_KEY" not in p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.json"))


@pytest.mark.parametrize("failure", ["timeout", "schema"])
def test_unknown_billing_stops_but_known_classifier_format_failure_is_http_failure(runtime, tmp_path, failure):
    client, collection, _ = runtime
    row = dict(id="DEV_SYNTH", question="How to reset?", stratum="impossible")
    seen, keys = [], {s: "SYNTHETIC_ONLY_KEY" for s in qa.ENDPOINTS}
    folder = tmp_path / "failure"
    if failure == "timeout":
        for _ in range(2):
            with pytest.raises(RuntimeError):
                run.http_case(client, row, "H", folder, "OFFLINE", run.Book(), keys, collection, delegates=providers(seen, failure))
        assert len(seen) == 1 and (folder / "wire/pending.json").exists() and (folder / "halt.json").exists()
    else:
        result = run.http_case(client, row, "H", folder, "OFFLINE", run.Book(), keys, collection, delegates=providers(seen, failure))
        assert result["http_status"] == 503 and len(seen) == 3
        metric = qa.score(result, dict(verdict="AMBIGUOUS"), None)
        assert metric["failed"] and not metric["refused"] and not metric["success"]


def test_budget_reference_and_wire_identity_are_complete(tmp_path):
    assert qa.full_reserve() == dict(service="136.802246080", evidence="96.501760", answer="96.501760")
    assert sum(map(Decimal, qa.full_reserve().values())) == Decimal("329.805766080")
    assert sum(p[4] for p in qa.PROFILES.values()) == 3100
    book = run.Book()
    event = dict(call_id="one", stage="generation", prompt_tokens=65536, completion_tokens=1024,
                 request_sha256="a", response_sha256="b")
    book.add(event)
    book.add(event)
    assert len(book.rows) == 1
    with pytest.raises(RuntimeError):
        book.add(event | {"response_sha256": "changed"})
    with pytest.raises(ValueError):
        qa.charge("generation", 65537, 1)
    book.rows = {str(i): ("answer", 65536, 2048, "a", "b") for i in range(620)}
    with pytest.raises(RuntimeError):
        book.check("answer")
    old = dict(run.wire.STAGES)
    with qa.wire_profile(True):
        assert run.wire.STAGES["classifier"][0] == qa.JUDGE_MODEL
    assert run.wire.STAGES == old
    with pytest.raises(ValueError):
        run.client_factory({}, tmp_path / "blocked", run.Book(), {}, lambda *a: None, None)


def test_frozen_question_loader_and_phase2_authorization_are_explicit(tmp_path, monkeypatch):
    metadata = tmp_path / "synthetic_metadata.json"
    rows = [dict(id=f"TRAIN_SYNTH_{i:03d}", question="train never projected", is_impossible=i >= 450) for i in range(600)]
    rows += [dict(id=f"DEV_SYNTH_{i:03d}", question="Frozen question  \n", is_impossible=i >= 160,
                  answer="PRIVATE_REFERENCE_NOT_FOR_PROVIDER") for i in range(310)]
    run.keep(metadata, rows)
    manifest = tmp_path / "manifest.json"
    run.keep(manifest, dict(generation_dataset=dict(metadata_sha256=run.assets.sha(metadata))))
    monkeypatch.setattr(run.assets, "MANIFEST", manifest)
    questions = run.load_questions(metadata)
    assert len(questions) == 310 and questions[0]["question"] == "Frozen question"
    assert "answer" not in questions[0] and sum(r["stratum"] == "answerable" for r in questions) == 160
    with pytest.raises(ValueError):
        run.load_questions(manifest)
    monkeypatch.setattr(run, "identity", lambda: {"synthetic": True})
    monkeypatch.setattr(run.inputs, "reference_profile", lambda p: {"path": "synthetic"})
    run.keep(tmp_path / "prepared.json", dict(scope="DEV310_HTTP_AND_TWO_BLIND_REVIEW_PHASES", identity=run.identity(), reference_tokenizer={"path": "synthetic"}))
    value = dict(prepared_sha256=run.assets.sha(tmp_path / "prepared.json"), scope="DEV310_HTTP_AND_TWO_BLIND_REVIEW_PHASES",
        max_provider_calls=3100, accounting_limits_cny={k: str(v) for k, v in qa.CAPS.items()}, accept_reference_estimate=True,
        current_prices_reconfirmed=True, allow_resume_completed_receipts_only=True, human_approval_reference="SYNTHETIC_ONLY_NOT_CONSENT")
    path = tmp_path / "approval.json"
    run.keep(path, value)
    assert run.approval(tmp_path, path, run.assets.sha(path))[1] == value["human_approval_reference"]
    old = tmp_path / "old_phase1_approval.json"
    run.keep(old, value | {"scope": "DEV160_RETRIEVAL_ONLY"})
    with pytest.raises(ValueError):
        run.approval(tmp_path, old, run.assets.sha(old))


def test_ambiguous_evidence_cannot_become_primary_success_after_answer_unblinding():
    answer = dict(behavior="ANSWER", supported=True, complete=True, citations_valid=True, reason="synthetic")
    result = dict(http_status=200, body=dict(answer="Press reset. [Source 1]"))
    score = qa.score(result, dict(verdict="AMBIGUOUS"), answer)
    assert score["review_conflict"] and not score["correct"] and not score["success"]
    diagnostic = run.safe_error_chain(ValueError("PRIVATE_DO_NOT_EMIT"))
    assert "PRIVATE_DO_NOT_EMIT" not in json.dumps(diagnostic)


def test_judge_payload_schema_and_completed_receipt_replay(tmp_path):
    packet = dict(question="How to reset?", sources=[dict(source_id="Source 1", content="Press reset.")])
    verdict = dict(verdict="SUFFICIENT", citations=["Source 1"], reason="explicit reset step")
    seen = []
    with pytest.raises(ValueError):
        qa.judge_body("evidence", packet | {"answer": "must remain blinded"})
    with pytest.raises(ValueError):
        qa.validate_verdict("evidence", verdict | {"citations": ["Source 9"]}, packet)
    with pytest.raises(ValueError):
        qa.parse_verdict("evidence", '{"verdict":"SUFFICIENT","verdict":"INSUFFICIENT","citations":[],"reason":"x"}', packet)
    with qa.wire_profile(True):
        body = qa.judge_body("evidence", packet)
        c = run.contract("OFFLINE", {"classifier": qa.BEIJING}, "synthetic:evidence")
        def validate(stage, actual):
            assert actual == body
        def respond(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json=dict(choices=[dict(index=0, finish_reason="stop", message=dict(role="assistant", content=json.dumps(verdict)))],
                usage=dict(prompt_tokens=10, completion_tokens=5, total_tokens=15)))
        for _ in range(2):
            factory, clients = run.client_factory(c, tmp_path / "wire", run.Book(), {"classifier": "OFFLINE_KEY"}, validate,
                None, delegates={"classifier": httpx.MockTransport(respond)}, kind="evidence")
            try:
                result = factory("classifier").chat.completions.create(**{k: v for k, v in body.items() if k != "enable_thinking"}, extra_body={"enable_thinking": False})
                assert qa.validate_verdict("evidence", json.loads(result.choices[0].message.content), packet) == verdict
            finally:
                for cli in clients:
                    cli.close()
    assert len(seen) == 1 and seen[0]["model"] == qa.JUDGE_MODEL


def test_blind_packet_freeze_gates_and_full_report(tmp_path):
    root = tmp_path
    questions = [dict(id=f"DEV_SYNTH_{i:03d}", question=f"Reset device {i}?", stratum="answerable" if i < 160 else "impossible") for i in range(310)]
    run.keep(root / "prepared.json", dict(questions=questions))
    for i, row in enumerate(questions):
        for arm in ("D", "H"):
            folder = root / "service" / f"{i:03d}-{arm}"
            folder.mkdir(parents=True)
            unobserved = i == 0 and arm == "D"
            if not unobserved:
                run.keep(folder / "context.json", dict(question=row["question"], sources=[dict(source_id="Source 1", content="Press reset.")]))
            run.keep(folder / "result.json", dict(http_status=503 if unobserved else 200, body=dict(answer=qa.REFUSAL), latency_ms=10))
    run.keep(root / "service_complete.json", dict(files=run.service_fingerprints(root)))
    with pytest.raises(ValueError):
        run.packets(root, "answer")
    tasks = run.packets(root, "evidence")
    assert len(tasks) == 310  # Exact duplicate D/H contexts share a blinded judgment.
    assert all(set(p) == {"question", "sources"} for p in tasks.values())
    values = {}
    for pid in tasks:
        values[pid] = dict(verdict="INSUFFICIENT", citations=[], reason="missing device applicability")
        folder = root / "evidence" / pid
        folder.mkdir(parents=True)
        run.keep(folder / "result.json", values[pid])
    path = root / "reviewed.json"
    run.keep(path, values)
    run.freeze(root, "evidence", path, "ACTUAL_ASSISTANT_REVIEW_REFERENCE_SYNTHETIC")
    assert run.packets(root, "answer") == {}
    empty = root / "empty.json"
    run.keep(empty, {})
    run.freeze(root, "answer", empty, "SYNTHETIC_ALL_EXACT_REFUSALS")
    report = run.report(root)
    assert report["primary"]["delta"] == 0 and report["primary"]["conclusion"] == "IMPROVEMENT_NOT_ESTABLISHED"
    assert report["arms"]["D"]["all_question_success"] == 309/310
    assert report["arms"]["D"]["failures"] == report["arms"]["D"]["unobserved_contexts"] == 1
    assert report["arms"]["D"]["grounded_complete_answer_rate"] == 0  # Refusing everything cannot win the primary metric.
    assert report["arms"]["D"]["answered_only_correctness"] is None
    assert report["http_latency"]["H"]["n"] == 310
    with pytest.raises(ValueError):
        qa.paired_report([])


def test_fresh_migration_copies_only_sources_and_never_upgrades_archive(tmp_path, monkeypatch):
    source, work = tmp_path / "source", tmp_path / "work"
    source.mkdir()
    work.mkdir()
    engine = create_engine("sqlite:///"+str(source / "runtime.db"))
    SQLModel.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(Document(id="synth", tenant_id=run.TENANT, uploaded_by="synthetic", filename="manual.md", file_type="md",
            source_path="synthetic/manual.md", checksum="fake", category="other", status="indexed"))
        db.commit()
    engine.dispose()
    before = run.assets.sha(source / "runtime.db")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///"+str(work / "runtime.db"))
    run.migrate_copy(source, work)
    with run.sqlite3.connect(work / "runtime.db") as db:
        assert db.execute("SELECT count(*) FROM documents").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM ticket").fetchone()[0] == 0
        assert db.execute("SELECT version_num FROM alembic_version").fetchone()[0]
    assert run.assets.sha(source / "runtime.db") == before
