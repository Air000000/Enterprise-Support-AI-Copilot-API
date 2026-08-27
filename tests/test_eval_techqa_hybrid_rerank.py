from importlib import import_module

import pytest

from experiments.evals.rerankers.qwen3_reranker import RerankedCandidate


def _load_module():
    try:
        return import_module("experiments.evals.eval_techqa_hybrid_rerank")
    except ModuleNotFoundError:
        pytest.fail("R4 C1 hybrid rerank evaluator is not implemented yet")


def test_candidate_snapshot_contract_preserves_cross_source_rrf_credit():
    module = _load_module()

    dense_ids = [f"d{i}" for i in range(99)]
    bm25_ids = [f"b{i}" for i in range(99)]

    # The shared chunk is rank 50 in both sources.
    # Its two RRF contributions must beat a rank-1 chunk
    # that appears in only one source.
    dense_ids.insert(49, "shared")
    bm25_ids.insert(49, "shared")

    all_ids = set(dense_ids) | set(bm25_ids)

    chunks_by_id = {
        chunk_id: module.TechQAChunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            chunk_index=0,
            content=f"content for {chunk_id}",
        )
        for chunk_id in all_ids
    }

    record = {
        "question_id": "TRAIN_Q1",
        "question": "shared technical query",
        "relevant_document_ids": ["doc-shared"],
        "raw_chunk_ids": dense_ids,
    }

    bm25_chunks = [
        chunks_by_id[chunk_id]
        for chunk_id in bm25_ids
    ]

    result = module.build_hybrid_snapshot_record(
        record,
        chunks_by_id=chunks_by_id,
        bm25_searcher=lambda query, top_k: bm25_chunks[:top_k],
    )

    assert len(result.dense_chunk_ids) == 100
    assert len(set(result.dense_chunk_ids)) == 100

    assert len(result.bm25_chunk_ids) == 100
    assert len(set(result.bm25_chunk_ids)) == 100

    fused_ids = [
        candidate.chunk_id
        for candidate in result.fused_candidates
    ]

    assert len(fused_ids) == 100
    assert len(set(fused_ids)) == 100

    # This is the key C1 contract:
    # cross-source overlap must retain both RRF rank contributions.
    assert fused_ids[0] == "shared"


def test_one_query_orchestration_preserves_frozen_candidates_and_provider_identity():
    module = _load_module()

    candidates = tuple(
        module.HybridCandidate(
            chunk_id=f"c{i}",
            document_id=(
                "shared-doc"
                if i < 2
                else f"doc{i}"
            ),
            content=f"content {i}",
        )
        for i in range(100)
    )

    record = module.HybridSnapshotRecord(
        question_id="TRAIN_Q1",
        question="technical query",
        relevant_document_ids=("shared-doc",),
        dense_chunk_ids=tuple(
            f"d{i}"
            for i in range(100)
        ),
        bm25_chunk_ids=tuple(
            f"b{i}"
            for i in range(100)
        ),
        fused_candidates=candidates,
    )

    class FakeResult:
        def __init__(self):
            self.results = tuple(
                RerankedCandidate(
                    chunk_id=candidates[index].chunk_id,
                    document_id=candidates[index].document_id,
                    content=candidates[index].content,
                    original_index=index,
                    relevance_score=float(100 - index),
                )
                for index in reversed(range(100))
            )
            self.request_id = "req-c1"
            self.total_tokens = 12345

    seen = {}

    def fake_reranker(query, rerank_candidates):
        seen["query"] = query
        seen["chunk_ids"] = [
            item.chunk_id
            for item in rerank_candidates
        ]
        return FakeResult()

    clock = iter([1.0, 1.25]).__next__

    result = module.rerank_snapshot_record(
        record,
        reranker=fake_reranker,
        clock=clock,
    )

    assert seen["query"] == "technical query"

    assert seen["chunk_ids"] == [
        f"c{i}"
        for i in range(100)
    ]

    assert len(result.reranked_chunk_ids) == 100

    assert result.document_ranking[0] == "doc99"

    # c0/c1 both map to shared-doc, so first-occurrence
    # document collapse yields 99 unique documents.
    assert len(result.document_ranking) == 99

    assert result.request_id == "req-c1"
    assert result.total_tokens == 12345
    assert result.rerank_latency_ms == 250.0
