"""Synthetic phase-1 cases: real SQL/BM25/RRF/SDK, MockTransport only."""

from decimal import Decimal
import json
from types import SimpleNamespace

import httpx
from openai import OpenAIError
import pytest
from sqlmodel import SQLModel, Session, create_engine

from experiments.evals import runtime_retrieval as evaluation
from models.document import Document, DocumentChunk


def plan(question="needle  \n"):
    return evaluation.build_plan([(f"DEV_SYNTH_{i:03d}", question) for i in range(160)], "a" * 64,
                                 "SYNTHETIC_ONLY_NOT_PAID_CONSENT")


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'synthetic.db'}")
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(evaluation.hybrid, "engine", engine)
    evaluation.hybrid._bm25_index.cache_clear()
    with Session(engine) as session:
        for i in range(120):
            doc = Document(id=f"d{i:03d}", tenant_id=evaluation.TENANT, uploaded_by="synthetic",
                           filename=f"manual{i}.md", file_type="md", source_path="synthetic/manual.md",
                           checksum="synthetic", category=evaluation.CATEGORY, status="indexed")
            session.add(doc)
            session.add(DocumentChunk(document_id=doc.id, tenant_id=evaluation.TENANT, category=evaluation.CATEGORY,
                chunk_index=0, content=f"network manual {i:03d}" + (" needle" if i == 119 else ""),
                embedding_id=f"doc:{doc.id}:v1:chunk:0"))
        session.commit()

    def query(**kwargs):
        assert kwargs["n_results"] == 100 and len(kwargs["query_embeddings"][0]) == 1024
        ids = list(evaluation.hybrid.load_visible_chunks(evaluation.TENANT, evaluation.CATEGORY))[:100]
        return {"ids": [ids], "distances": [[.1] * 100]}

    monkeypatch.setattr(evaluation.hybrid, "get_collection", lambda: SimpleNamespace(query=query))
    yield
    evaluation.hybrid._bm25_index.cache_clear()
    engine.dispose()


def providers(seen, failure=None):
    def embedding(request):
        body = json.loads(request.content)
        seen.append(("embedding", body))
        if failure == "timeout":
            raise httpx.ReadTimeout("synthetic")
        return httpx.Response(200, json={"data": [{"index": 0, "object": "embedding", "embedding": [1.] * 1024}],
            "usage": {"prompt_tokens": 8193 if failure == "usage" else 5,
                      "total_tokens": 8193 if failure == "usage" else 5}})

    def rerank(request):
        body = json.loads(request.content)
        seen.append(("rerank", body))
        order = sorted(range(len(body["documents"])), key=lambda i: "needle" not in body["documents"][i])
        rows = [{"index": i, "relevance_score": 1 / (rank + 1)} for rank, i in enumerate(order)]
        if failure == "permutation":
            rows[1]["index"] = rows[0]["index"]
        return httpx.Response(200, json={"results": rows, "usage": {"total_tokens": 100}})
    return {"embedding": httpx.MockTransport(embedding), "rerank": httpx.MockTransport(rerank)}


KEYS = {"embedding": "SYNTHETIC_KEY_ONLY", "rerank": "SYNTHETIC_KEY_ONLY"}


def test_frozen_schedule_and_decimal_reservation():
    value = plan()
    calls = value["contract"]["calls"]
    assert len(calls) == 480 and list(calls.values()).count("embedding") == 160
    assert list(calls)[:6] == ["DEV_SYNTH_000:E", "DEV_SYNTH_000:DR", "DEV_SYNTH_000:H",
                               "DEV_SYNTH_001:E", "DEV_SYNTH_001:H", "DEV_SYNTH_001:DR"]
    assert value["identity"]["questions"][0][1] == "needle"
    events = [{"stage": stage, "prompt_tokens": evaluation.TOKEN_RESERVATIONS[stage]} for stage in calls.values()]
    assert evaluation.accounting(events) == Decimal("29.433088")
    assert value["contract"]["max_total_request_bytes"] == 160 * 32768 + 320 * 524288
    with pytest.raises(RuntimeError, match="budget"):
        evaluation.accounting([{"stage": "rerank", "prompt_tokens": 120000}] * 389, "rerank")
    with pytest.raises(RuntimeError, match="reservation"):
        evaluation.accounting([{"stage": "embedding", "prompt_tokens": 8193}])


@pytest.mark.parametrize("bad", ["count", "duplicate", "label", "blank", "asset"])
def test_plan_has_no_label_or_unfrozen_subset_input(bad):
    questions = [(f"DEV_SYNTH_{i:03d}", "needle") for i in range(160)]
    if bad == "count":
        questions.pop()
    elif bad == "duplicate":
        questions[-1] = questions[0]
    elif bad == "label":
        questions[0] = (*questions[0], "SECRET_QREL_NOT_FOR_PROVIDER")
    elif bad == "blank":
        questions[0] = (questions[0][0], "  ")
    with pytest.raises(ValueError):
        evaluation.build_plan(questions, "bad" if bad == "asset" else "a" * 64, "offline")


def test_observe_full_rankings_runtime_top14_and_restore_without_sends(runtime, tmp_path):
    value, seen, root = plan(), [], tmp_path / "wire"
    delegates = providers(seen)
    first = evaluation.observe_case(value, 0, root, KEYS, delegates)
    assert len(seen) == 3 and [stage for stage, _ in seen] == ["embedding", "rerank", "rerank"]
    assert all(len(rows) == 100 for rows in first["chunk_rankings"].values())
    assert all(len(rows) == 100 for rows in first["document_rankings"].values())
    assert first["top14_chunk_ids"] == first["chunk_rankings"]["H"][:14]
    assert first["chunk_rankings"]["D"] != first["chunk_rankings"]["H"]
    assert first["document_rankings"]["H"][0] == "d119"
    assert evaluation.observe_case(value, 0, root, KEYS, delegates) == first
    assert len(seen) == 3  # Fresh clients read the original response, no provider invocation.
    evaluation.observe_case(value, 1, root, KEYS, delegates)
    assert len(seen) == 6
    events = evaluation.wire._read_jsonl(root / "attempts.jsonl")
    assert [row["call_id"] for row in events] == list(value["contract"]["calls"])[:6]
    for stage, body in seen:
        assert all(cid not in json.dumps(body) for cid, _ in value["identity"]["questions"])
        assert set(body) == ({"model", "input", "encoding_format", "dimensions"} if stage == "embedding" else
                             {"model", "query", "documents", "top_n", "instruct"})
    assert not (root / "pending.json").exists()


@pytest.mark.parametrize("failure", ["timeout", "usage", "permutation"])
def test_failed_or_unknown_case_stops_and_cannot_resume_spending(runtime, tmp_path, failure):
    value, seen, root = plan(), [], tmp_path / "wire"
    delegates = providers(seen, failure)
    for _ in range(2):
        with pytest.raises((OpenAIError, RuntimeError)):
            evaluation.observe_case(value, 0, root, KEYS, delegates)
    assert len(seen) == (2 if failure == "permutation" else 1)
    assert (root / "pending.json").exists()
    assert not evaluation._SLOT.locked() and not (root / "transport.lock").exists()
    if failure != "timeout":
        assert (root / f"responses/{1 if failure == 'permutation' else 0}.json").exists()


@pytest.mark.parametrize("reason", ["skip", "bytes", "replay_changed", "response_changed", "identity"])
def test_no_sends_after_skipped_changed_or_oversized_input(runtime, tmp_path, reason):
    value, seen, root = plan("x" * 33000 if reason == "bytes" else "needle"), [], tmp_path / "wire"
    delegates = providers(seen)
    if reason in {"replay_changed", "response_changed", "identity"}:
        evaluation.observe_case(value, 0, root, KEYS, delegates)
        if reason == "replay_changed":
            value["identity"]["questions"][0][1] = "changed needle"
        elif reason == "response_changed":
            (root / "responses/0.json").write_text('{"wire_utf8":"{}"}', encoding="utf-8")
        else:
            value["contract"]["authorization_reference"] += "changed"
    before = len(seen)
    with pytest.raises((OpenAIError, RuntimeError, KeyError)):
        evaluation.observe_case(value, 1 if reason == "skip" else 0, root, KEYS, delegates)
    assert len(seen) == before


def test_offline_only_no_accidental_real_transport(tmp_path):
    value = plan()
    with httpx.HTTPTransport(retries=0) as real:
        with pytest.raises(ValueError, match="offline MockTransport"):
            evaluation.make_client(value, tmp_path / "wire", "DEV_SYNTH_000:E", "fake", {}, delegate=real)
    assert not (tmp_path / "wire").exists()


def test_runtime_top14_mismatch_blocks_restoration(runtime, tmp_path, monkeypatch):
    value, seen, root = plan(), [], tmp_path / "wire"
    original = evaluation.hybrid.search_hybrid
    monkeypatch.setattr(evaluation.hybrid, "search_hybrid", lambda *a, **kw: original(*a, **kw)[:13])
    delegates = providers(seen)
    with pytest.raises(RuntimeError, match="Top14"):
        evaluation.observe_case(value, 0, root, KEYS, delegates)
    assert len(seen) == 3 and (root / "pending.json").exists()
    with pytest.raises(OpenAIError):
        evaluation.observe_case(value, 0, root, KEYS, delegates)
    assert len(seen) == 3


def test_projection_rejected_before_any_send(tmp_path):
    value, seen, root = plan(), [], tmp_path / "wire"
    body = {"model": "text-embedding-v4", "input": ["needle"], "dimensions": 1024, "encoding_format": "float"}
    with evaluation.make_client(value, root, "DEV_SYNTH_000:E", KEYS["embedding"], body,
                                delegate=providers(seen)["embedding"]) as client:
        with pytest.raises(OpenAIError):
            client.embeddings.create(**(body | {"input": ["unauthorized text"]}))
    assert seen == [] and not (root / "pending.json").exists()


@pytest.mark.parametrize("change", ["source", "stale_lock"])
def test_source_and_stale_lock_fail_closed(runtime, tmp_path, monkeypatch, change):
    value, seen, root = plan(), [], tmp_path / "wire"
    delegates = providers(seen)
    if change == "source":
        monkeypatch.setattr(evaluation, "_SOURCE_FILES", ())
    else:
        evaluation.observe_case(value, 0, root, KEYS, delegates)
        (root / "transport.lock").write_text("", encoding="utf-8")
    before = len(seen)
    with pytest.raises((RuntimeError, OpenAIError)):
        evaluation.observe_case(value, 0, root, KEYS, delegates)
    assert len(seen) == before
    if change == "stale_lock":
        assert (root / "transport.lock").exists()  # Never automatically remove another attempt's lock.
