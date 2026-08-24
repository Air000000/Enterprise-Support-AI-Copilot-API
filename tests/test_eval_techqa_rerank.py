import importlib
from types import SimpleNamespace

import pytest


def _rerank_eval_module():
    try:
        return importlib.import_module("experiments.evals.eval_techqa_rerank")
    except ModuleNotFoundError:
        pytest.fail("experiments.evals.eval_techqa_rerank is not implemented yet")


class FakeCollection:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def get(self, *, ids, include):
        self.calls.append({"ids": list(ids), "include": list(include)})
        selected = [self.rows[chunk_id] for chunk_id in reversed(ids) if chunk_id in self.rows]
        return {
            "ids": [row["chunk_id"] for row in selected],
            "documents": [row["content"] for row in selected],
            "metadatas": [
                {"document_id": row["document_id"]}
                for row in selected
            ],
        }


def test_rerank_frozen_e0_record_normalizes_query_rehydrates_order_then_collapses_docs():
    module = _rerank_eval_module()
    collection = FakeCollection(
        {
            "c0": {"chunk_id": "c0", "document_id": "doc_a", "content": "alpha zero"},
            "c1": {"chunk_id": "c1", "document_id": "doc_a", "content": "alpha one"},
            "c2": {"chunk_id": "c2", "document_id": "doc_gold", "content": "gold evidence"},
        }
    )
    record = {
        "question_id": "TRAIN_Q001",
        "question": "technical support question\n",
        "relevant_document_ids": ["doc_gold"],
        "raw_chunk_ids": ["c0", "c1", "c2"],
        "raw_document_ids": ["doc_a", "doc_a", "doc_gold"],
    }
    observed = {}

    def fake_reranker(query, candidates):
        observed["query"] = query
        observed["candidate_ids"] = [candidate.chunk_id for candidate in candidates]
        observed["candidate_docs"] = [candidate.document_id for candidate in candidates]
        observed["candidate_contents"] = [candidate.content for candidate in candidates]
        return SimpleNamespace(
            results=(
                SimpleNamespace(
                    chunk_id="c2",
                    document_id="doc_gold",
                    content="gold evidence",
                    original_index=2,
                    relevance_score=0.98,
                ),
                SimpleNamespace(
                    chunk_id="c1",
                    document_id="doc_a",
                    content="alpha one",
                    original_index=1,
                    relevance_score=0.70,
                ),
                SimpleNamespace(
                    chunk_id="c0",
                    document_id="doc_a",
                    content="alpha zero",
                    original_index=0,
                    relevance_score=0.60,
                ),
            ),
            request_id="req-1",
            total_tokens=99,
        )

    result = module.rerank_frozen_e0_record(
        record,
        collection=collection,
        reranker=fake_reranker,
        clock=iter([10.000, 10.025]).__next__,
    )

    assert collection.calls == [
        {
            "ids": ["c0", "c1", "c2"],
            "include": ["documents", "metadatas"],
        }
    ]
    assert observed == {
        "query": "technical support question",
        "candidate_ids": ["c0", "c1", "c2"],
        "candidate_docs": ["doc_a", "doc_a", "doc_gold"],
        "candidate_contents": ["alpha zero", "alpha one", "gold evidence"],
    }
    assert result.question_id == "TRAIN_Q001"
    assert result.relevant_document_ids == ("doc_gold",)
    assert result.dense_chunk_ids == ("c0", "c1", "c2")
    assert result.reranked_chunk_ids == ("c2", "c1", "c0")
    assert result.reranked_document_ids == ("doc_gold", "doc_a", "doc_a")
    assert result.document_ranking == ("doc_gold", "doc_a")
    assert result.rerank_latency_ms == pytest.approx(25.0)
    assert result.request_id == "req-1"
    assert result.total_tokens == 99


def test_rerank_frozen_e0_record_rejects_missing_chroma_candidate():
    module = _rerank_eval_module()
    collection = FakeCollection(
        {
            "c0": {"chunk_id": "c0", "document_id": "doc_a", "content": "alpha zero"},
            "c2": {"chunk_id": "c2", "document_id": "doc_gold", "content": "gold evidence"},
        }
    )
    record = {
        "question_id": "TRAIN_Q001",
        "question": "technical support question",
        "relevant_document_ids": ["doc_gold"],
        "raw_chunk_ids": ["c0", "c1", "c2"],
        "raw_document_ids": ["doc_a", "doc_a", "doc_gold"],
    }

    with pytest.raises(module.FrozenCandidatePoolError, match="missing.*c1"):
        module.rerank_frozen_e0_record(
            record,
            collection=collection,
            reranker=lambda query, candidates: pytest.fail("reranker must not be called"),
        )
