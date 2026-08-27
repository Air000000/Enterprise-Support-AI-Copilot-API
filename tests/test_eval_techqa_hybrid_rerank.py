from importlib import import_module

import pytest


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
