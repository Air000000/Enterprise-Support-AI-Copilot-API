"""Real SDK/HTTPX with synthetic payloads only; no credential or dataset files."""

import json

import httpx
from openai import OpenAIError
import pytest

from experiments.evals import runtime_transport as guard
from experiments.evals.rerankers import qwen3_reranker as rerank
from rag_runtime import build_rag_index as embedding, query_rag_chroma as dense


def contract():
    return {"authorization_reference": "OFFLINE_TEST_ONLY_NOT_PAID_AUTHORIZATION",
            "stages": {stage: {"base_url": "https://model.invalid/v1", "max_request_bytes": 64000} for stage in guard.STAGES},
            "calls": {stage: stage for stage in guard.STAGES} | {"next": "generation"},
            "max_total_request_bytes": 100000}


def reply(stage):
    if stage == "embedding":
        return {"data": [{"index": 0, "object": "embedding", "embedding": [1.0] * 1024}],
                "usage": {"prompt_tokens": 5, "total_tokens": 5}}
    if stage == "rerank":
        return {"results": [{"index": 0, "relevance_score": 1.0}], "usage": {"total_tokens": 5}}
    return {"id": "synthetic-response-id", "choices": [{"index": 0, "finish_reason": "stop", "message": {
        "role": "assistant", "content": "synthetic answer"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}


def chat(client, stage="generation", **extra):
    options = dict(model=guard.STAGES[stage][0], messages=[
        {"role": "system", "content": "synthetic system"}, {"role": "user", "content": "synthetic question"}],
        temperature=0 if stage == "classifier" else 0.2)
    if stage == "classifier":
        options.update(max_tokens=2048, extra_body={"enable_thinking": False}, response_format={"type": "json_object"})
    return client.chat.completions.create(**(options | extra))


def test_all_stages_use_real_sdk_bounded_requests_and_durable_usage(tmp_path, monkeypatch):
    root, seen = tmp_path / "ledger", []
    for stage in guard.STAGES:
        def respond(request):
            body = json.loads(request.content)
            seen.append((stage, body, request.extensions["timeout"]))
            assert (root / "pending.json").exists()  # Reservation precedes send.
            assert len(list((root / "requests").glob("*.json"))) == len(seen)
            return httpx.Response(200, json=reply(stage))
        with guard.make_client(contract(), root, stage, "SECRET_FAKE_KEY_ONLY", delegate=httpx.MockTransport(respond)) as client:
            assert client.max_retries == 0 and client._client.follow_redirects is False
            if stage == "embedding":
                monkeypatch.setattr(embedding, "get_client", lambda: client)
                monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-v4")
                assert len(embedding.embed_texts(["synthetic question"])[0]) == 1024
            elif stage == "rerank":
                assert len(rerank.rerank_candidates("synthetic question", [rerank.RerankCandidate("c", "d", "body")], client=client).results) == 1
            elif stage == "generation":
                monkeypatch.setattr(dense, "get_llm_client", lambda: client)
                monkeypatch.setenv("CHAT_MODEL", "qwen3.5-plus")
                assert dense.generate_answer("synthetic question", "synthetic context") == "synthetic answer"
            else:
                chat(client, stage)
        assert not (root / "pending.json").exists()
    assert len(seen) == 4
    assert seen[0][1]["dimensions"] == 1024
    assert seen[-1][1]["max_tokens"] == 1024 and seen[-1][1]["enable_thinking"] is False
    assert all(max(timeout.values()) <= guard.STAGES[stage][2] for stage, _, timeout in seen)
    events = guard._read_jsonl(root / "attempts.jsonl")
    assert sum(row["total_tokens"] for row in events) == 24
    assert all("SECRET_FAKE_KEY_ONLY" not in path.read_text(encoding="utf-8") for path in root.rglob("*.json*"))


@pytest.mark.parametrize("failure", ["timeout", "redirect", "500", "usage", "finish", "truncated", "tokens", "journal"])
def test_ambiguous_or_incomplete_attempt_blocks_every_later_send(tmp_path, monkeypatch, failure):
    root, sends = tmp_path / "ledger", []
    def respond(request):
        sends.append(1)
        if failure == "timeout":
            raise httpx.ReadTimeout("PRIVATE_PROVIDER_TEXT_MUST_NOT_BE_LOGGED")
        if failure in {"redirect", "500"}:
            return httpx.Response(302 if failure == "redirect" else 500,
                headers={"location": "https://foreign.invalid"}, text="SECRET_FAKE_KEY_ONLY")
        payload = reply("generation")
        if failure == "usage":
            del payload["usage"]
        elif failure in {"finish", "truncated"}:
            payload["choices"][0]["finish_reason"] = None if failure == "finish" else "length"
        elif failure == "tokens":
            payload["usage"] = {"prompt_tokens": 5, "completion_tokens": 1025, "total_tokens": 1030}
        return httpx.Response(200, json=payload)
    if failure == "journal":
        monkeypatch.setattr(guard, "_append_jsonl", lambda *a: (_ for _ in ()).throw(OSError("disk failure")))
    for cid in ("generation", "next"):
        with guard.make_client(contract(), root, cid, "offline", delegate=httpx.MockTransport(respond)) as client:
            with pytest.raises(OpenAIError):
                chat(client)
    assert sends == [1] and (root / "pending.json").exists()
    assert not (root / "transport.lock").exists()
    if failure in {"usage", "finish", "truncated", "tokens", "journal"}:
        assert (root / "responses/0.json").exists()  # Preserve invalid draft for audit, not paid repair.
    assert all("SECRET_FAKE_KEY_ONLY" not in path.read_text(encoding="utf-8") for path in root.rglob("*.json*"))


@pytest.mark.parametrize("problem", ["model", "timeout", "bytes", "unexpected_field", "endpoint", "output", "thinking"])
def test_unapproved_request_rejected_before_reservation_or_send(tmp_path, problem):
    value, root, sends = contract(), tmp_path / "ledger", []
    if problem == "bytes":
        value["max_total_request_bytes"] = 1
    with guard.make_client(value, root, "generation", "offline", delegate=httpx.MockTransport(lambda r: sends.append(r))) as client:
        options = {}
        if problem == "model":
            options["model"] = "unapproved"
        elif problem == "timeout":
            options["timeout"] = None
        elif problem == "unexpected_field":
            options["extra_body"] = {"enable_thinking": False, "labels": "not allowed"}
        elif problem == "endpoint":
            client = client.with_options(base_url="https://foreign.invalid/v1")
        elif problem == "output":
            options["max_tokens"] = 2048
        elif problem == "thinking":
            options["extra_body"] = {"enable_thinking": True}
        with pytest.raises(OpenAIError):
            chat(client, **options)
    assert sends == [] and not (root / "pending.json").exists()


@pytest.mark.parametrize("change", ["duplicate", "response", "contract", "orphan", "profile"])
def test_restart_rejects_duplicate_or_changed_checkpoint(tmp_path, monkeypatch, change):
    root, sends = tmp_path / "ledger", []
    def respond(request):
        sends.append(1)
        return httpx.Response(200, json=reply("generation"))
    with guard.make_client(contract(), root, "generation", "offline", delegate=httpx.MockTransport(respond)) as client:
        chat(client)
    if change == "response":
        (root / "responses/0.json").write_text('{"wire_utf8":"{}"}', encoding="utf-8")
    elif change == "orphan":
        (root / "responses/1.json").write_text("{}", encoding="utf-8")
    value = contract()
    if change in {"contract", "profile"}:
        if change == "profile":
            monkeypatch.setitem(guard.STAGES, "generation", ("qwen3.5-plus", "/chat/completions", 31, 1024))
        else:
            value["max_total_request_bytes"] += 1
        with pytest.raises(RuntimeError):
            guard.make_client(value, root, "next", "offline", delegate=httpx.MockTransport(respond))
    else:
        with guard.make_client(value, root, "generation" if change == "duplicate" else "next", "offline", delegate=httpx.MockTransport(respond)) as client:
            with pytest.raises(OpenAIError):
                chat(client)
    assert sends == [1]


@pytest.mark.parametrize("invalid", ["credentials", "delegate"])
def test_no_implicit_credentials_or_retrying_transport_injection(tmp_path, invalid):
    root = tmp_path / "ledger"
    with httpx.HTTPTransport(retries=1, trust_env=False) as retrying:
        with pytest.raises(ValueError):
            guard.make_client(contract(), root, "generation", None if invalid == "credentials" else "offline",
                              delegate=retrying if invalid == "delegate" else httpx.MockTransport(lambda r: None))
    assert not root.exists()


@pytest.mark.parametrize("location", ["request", "response"])
def test_active_credential_never_stored_as_payload_or_response(tmp_path, location):
    root, sends, key = tmp_path / "ledger", [], "SECRET_FAKE_KEY_ONLY"
    def respond(request):
        sends.append(1)
        value = reply("generation")
        value["choices"][0]["message"]["content"] = key
        return httpx.Response(200, json=value)
    with guard.make_client(contract(), root, "generation", key, delegate=httpx.MockTransport(respond)) as client:
        options = {"messages": [{"role": "system", "content": "synthetic"}, {"role": "user", "content": key}]} if location == "request" else {}
        with pytest.raises(OpenAIError):
            chat(client, **options)
    assert len(sends) == (0 if location == "request" else 1)
    assert all(key not in path.read_text(encoding="utf-8") for path in root.rglob("*.json*"))
