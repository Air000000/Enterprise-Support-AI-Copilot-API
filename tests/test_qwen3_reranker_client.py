import importlib

import pytest


def _reranker_module():
    return importlib.import_module("experiments.evals.rerankers.qwen3_reranker")


def _require(module, name: str):
    value = getattr(module, name, None)
    if value is None:
        pytest.fail(f"{module.__name__}.{name} is not implemented yet")
    return value


class FakeRerankClient:
    def __init__(self):
        self.calls = []

    def post(self, path, *, body, cast_to):
        self.calls.append({"path": path, "body": body, "cast_to": cast_to})
        return {
            "id": "request-dedicated",
            "results": [{"index": 0, "relevance_score": 0.9}],
            "usage": {"total_tokens": 10},
        }


def test_get_rerank_client_uses_only_dedicated_rerank_environment(monkeypatch):
    reranker = _reranker_module()
    getter = _require(reranker, "get_rerank_client")

    monkeypatch.setenv("DASHSCOPE_RERANK_API_KEY", "rerank-key")
    monkeypatch.setenv(
        "DASHSCOPE_RERANK_BASE_URL",
        "https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-api/v1",
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "chat-key")
    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    observed = {}

    class FakeOpenAI:
        def __init__(self, *, api_key, base_url, max_retries):
            observed["api_key"] = api_key
            observed["base_url"] = base_url
            observed["max_retries"] = max_retries

    monkeypatch.setattr(reranker, "OpenAI", FakeOpenAI, raising=False)

    getter()

    assert observed == {
        "api_key": "rerank-key",
        "base_url": (
            "https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-api/v1"
        ),
        "max_retries": 0,
    }


def test_get_rerank_client_rejects_missing_dedicated_config_even_if_chat_config_exists(
    monkeypatch,
):
    reranker = _reranker_module()
    getter = _require(reranker, "get_rerank_client")

    monkeypatch.delenv("DASHSCOPE_RERANK_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_RERANK_BASE_URL", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "chat-key")
    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    with pytest.raises(RuntimeError, match="DASHSCOPE_RERANK_API_KEY"):
        getter()


def test_rerank_candidates_defaults_to_dedicated_rerank_client(monkeypatch):
    reranker = _reranker_module()
    client = FakeRerankClient()

    monkeypatch.setattr(
        reranker,
        "get_rerank_client",
        lambda: client,
        raising=False,
    )

    def forbidden_chat_client():
        raise AssertionError("reranker must not use the shared chat/embedding client")

    monkeypatch.setattr(reranker, "get_llm_client", forbidden_chat_client)

    result = reranker.rerank_candidates(
        "technical support question",
        [reranker.RerankCandidate("c0", "d0", "evidence")],
    )

    assert result.request_id == "request-dedicated"
    assert len(client.calls) == 1
