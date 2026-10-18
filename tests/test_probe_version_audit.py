"""Offline wire-guard checks; mocked classifier output is not quality evidence."""

import json
from pathlib import Path

import httpx
import pytest

from scripts import probe_version_audit as probe

URL = "https://ws-offline.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"


def run_fake(monkeypatch, respond, url=URL):
    monkeypatch.setattr("dotenv.dotenv_values", lambda _: {
        "DASHSCOPE_PHASE_B_API_KEY": "private-test-key", "DASHSCOPE_PHASE_B_BASE_URL": url})

    def transport(**kwargs):
        assert kwargs == {"retries": 0}
        return httpx.MockTransport(respond)

    monkeypatch.setattr(httpx, "HTTPTransport", transport)
    return probe.execute(Path("not-read.env"), probe.plan()["plan_sha256"])


def reply(request, finish="stop"):
    body = json.loads(request.content)
    question = json.loads(body["messages"][1]["content"])["question"]
    case = next(c for c in probe.CASES if c[1] == question)
    status = case[3]
    if case[0] == "fixed_version" and body["messages"][0]["content"] == probe.candidate.prior.CLASSIFIER_SYSTEM_PROMPT_V5_3_DEV:
        status = "NOT_REQUIRED"  # Simulated mismatch stays visible, not a real baseline result.
    ref = [{"source_id": "Source 1", "segment_ids": [1]}]
    facet = {"status": status, "evidence": ref if status == "SUPPORTED" else [], "reason": "Offline fixture."}
    outcome = {"status": "SUPPORTED" if case[4] == "SUFFICIENT" else "MISSING",
               "evidence": ref if case[4] == "SUFFICIENT" else [], "reason": "Offline fixture."}
    raw = json.dumps({"requirements": [{"requirement": "Requested claim", **outcome}],
        "coverage_checks": {"requested_outcomes": outcome, "applicability": outcome, "version_and_time": facet},
        "unresolved_ambiguities": []})
    return httpx.Response(200, json={"id": "offline", "object": "chat.completion", "model": probe.MODEL,
        "choices": [{"index": 0, "finish_reason": finish, "message": {
            "role": "assistant", "content": raw, "reasoning_content": "PRIVATE_REASONING"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}})


def test_plan_is_clean_paired_and_has_no_provider_or_env_reads(monkeypatch):
    monkeypatch.setattr("dotenv.dotenv_values", lambda _: pytest.fail("plan read credentials"))
    planned = probe.planned_calls()
    assert len(planned) == probe.plan()["max_calls"] == 12
    for index in range(0, 12, 2):
        first, second = planned[index:index + 2]
        assert first[0] == second[0] and first[1] != second[1]
        assert first[3]["messages"][1:] == second[3]["messages"][1:]
        assert first[1] == ("v5_3" if index // 2 % 2 == 0 else "v5_3_1")
        for _, _, _, body in (first, second):
            payload = json.loads(body["messages"][1]["content"])
            assert set(payload) == {"question", "sources"} and "LOCAL_" not in json.dumps(payload)
            assert body["enable_thinking"] is False and body["max_tokens"] == 2048


def test_complete_mock_run_records_all_pairs_without_relabeling(monkeypatch):
    sent = []

    def respond(request):
        assert request.extensions["timeout"]["read"] == 30
        sent.append(request)
        return reply(request)

    result = run_fake(monkeypatch, respond)
    assert result["status"] == "COMPLETED_DEVELOPMENT_CHECK_NO_PROMOTION"
    assert len(sent) == len(result["calls"]) == len(result["outcomes"]) == 12
    assert all(row["status"] == 200 and row["transport_phase"] == "complete" for row in result["calls"])
    assert sum(row["total_tokens"] for row in result["calls"]) == 240
    assert [row["facet_matches"] for row in result["outcomes"]].count(False) == 1
    assert all(row["decision_matches"] for row in result["outcomes"])
    assert "private-test-key" not in json.dumps(result) and "PRIVATE_REASONING" not in json.dumps(result)


@pytest.mark.parametrize("failure", ["read", "connect", "http", "json", "truncated", "body"])
def test_provider_or_parse_failure_stops_without_retry_or_error_body(monkeypatch, failure):
    sent = []

    class BrokenBody(httpx.SyncByteStream):
        def __iter__(self):
            yield b"partial"
            raise httpx.ReadTimeout("private-provider-error")

    def respond(request):
        sent.append(request)
        if failure in {"read", "connect"}:
            error = httpx.ReadTimeout if failure == "read" else httpx.ConnectTimeout
            raise error("private-provider-error", request=request)
        if failure == "http":
            return httpx.Response(500, json={"error": "private-provider-error"})
        if failure == "json":
            return httpx.Response(200, content=b"private-provider-error")
        if failure == "body":
            return httpx.Response(200, stream=BrokenBody())
        return reply(request, "length")

    result = run_fake(monkeypatch, respond)
    assert result["status"] == "FAILED_NO_RERUN" and result["error_types"]
    assert len(sent) == len(result["calls"]) == 1 and not result["outcomes"]
    assert "private-provider-error" not in json.dumps(result) and "private-test-key" not in json.dumps(result)
    if failure == "body":
        assert result["calls"][0]["status"] == 200 and result["calls"][0]["transport_phase"] == "reading_body"


@pytest.mark.parametrize("change", ["label", "foreign_source"])
def test_wire_guard_blocks_unapproved_data_before_transport(monkeypatch, change):
    original_send = httpx.Client.send
    sent = []

    def tamper(client, request, **kwargs):
        body = json.loads(request.content)
        if change == "label":
            body["expected_status"] = "SUPPORTED"
        else:
            body["messages"][1]["content"] = "foreign business material"
        changed = httpx.Request(request.method, request.url, json=body, extensions=request.extensions,
            headers={k: v for k, v in request.headers.items() if k.lower() != "content-length"})
        return original_send(client, changed, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", tamper)
    result = run_fake(monkeypatch, lambda request: sent.append(request) or reply(request))
    assert result["status"] == "FAILED_NO_RERUN" and not sent and not result["calls"]


def test_changed_plan_fails_before_credentials_are_read(monkeypatch):
    monkeypatch.setattr("dotenv.dotenv_values", lambda _: pytest.fail("read credentials before hash check"))
    with pytest.raises(RuntimeError, match="Plan changed"):
        probe.execute(Path("not-read.env"), "wrong-hash")


@pytest.mark.parametrize("limit", ["MAX_CALLS", "MAX_BODY_BYTES", "MAX_TOTAL_BYTES"])
def test_wire_budget_stops_before_any_extra_send(monkeypatch, limit):
    monkeypatch.setattr(probe, limit, 1)
    sent = []
    result = run_fake(monkeypatch, lambda request: sent.append(request) or reply(request))
    assert result["status"] == "FAILED_NO_RERUN"
    assert len(sent) == len(result["calls"]) == (1 if limit == "MAX_CALLS" else 0)


@pytest.mark.parametrize("url", ["https://example.com/compatible-mode/v1", "https://user:password@dashscope-intl.aliyuncs.com/compatible-mode/v1"])
def test_endpoint_outside_scope_is_rejected_before_transport(monkeypatch, url):
    with pytest.raises(RuntimeError, match="Singapore"):
        run_fake(monkeypatch, lambda _: pytest.fail("unapproved endpoint called"), url)
