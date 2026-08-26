import importlib

import pytest


def _reranker_module():
    try:
        return importlib.import_module("experiments.evals.rerankers.qwen3_reranker")
    except ModuleNotFoundError:
        pytest.fail("experiments.evals.rerankers.qwen3_reranker is not implemented yet")


class FakeRerankClient:
    def __init__(self, response=None, *, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls = []

    def post(self, path, *, body, cast_to):
        self.calls.append({"path": path, "body": body, "cast_to": cast_to})
        if self.error is not None:
            raise self.error
        return self.response


def test_qwen3_reranker_reorders_candidates_and_preserves_identity():
    reranker = _reranker_module()
    candidates = [
        reranker.RerankCandidate("c0", "d0", "zero"),
        reranker.RerankCandidate("c1", "d1", "one"),
        reranker.RerankCandidate("c2", "d2", "two"),
    ]
    client = FakeRerankClient(
        {
            "id": "request-123",
            "results": [
                {"index": 2, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.70},
                {"index": 1, "relevance_score": 0.10},
            ],
            "usage": {"total_tokens": 321},
        }
    )

    result = reranker.rerank_candidates(
        "technical support question \n",
        candidates,
        client=client,
    )

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["path"] == "/reranks"
    assert call["cast_to"] is object
    assert call["body"] == {
        "model": "qwen3-rerank",
        "query": "technical support question",
        "documents": ["zero", "one", "two"],
        "top_n": 3,
        "instruct": reranker.DEFAULT_RERANK_INSTRUCTION,
    }

    assert [item.chunk_id for item in result.results] == ["c2", "c0", "c1"]
    assert [item.document_id for item in result.results] == ["d2", "d0", "d1"]
    assert [item.original_index for item in result.results] == [2, 0, 1]
    assert [item.relevance_score for item in result.results] == [0.95, 0.70, 0.10]
    assert result.request_id == "request-123"
    assert result.total_tokens == 321


def test_qwen3_reranker_rejects_incomplete_provider_permutation():
    reranker = _reranker_module()
    candidates = [
        reranker.RerankCandidate("c0", "d0", "zero"),
        reranker.RerankCandidate("c1", "d1", "one"),
        reranker.RerankCandidate("c2", "d2", "two"),
    ]
    client = FakeRerankClient(
        {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.8},
            ]
        }
    )

    with pytest.raises(reranker.RerankProviderError, match="complete permutation"):
        reranker.rerank_candidates("question", candidates, client=client)


def test_qwen3_reranker_surfaces_provider_failure_without_dense_fallback():
    reranker = _reranker_module()
    candidates = [reranker.RerankCandidate("c0", "d0", "zero")]
    client = FakeRerankClient(error=RuntimeError("provider unavailable"))

    with pytest.raises(reranker.RerankProviderError, match="provider unavailable"):
        reranker.rerank_candidates("question", candidates, client=client)
