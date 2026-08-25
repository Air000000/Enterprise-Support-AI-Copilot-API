from __future__ import annotations

from importlib import import_module

import pytest


def _load_module():
    try:
        return import_module("experiments.evals.rerank_residual_analysis")
    except ModuleNotFoundError:
        pytest.fail("rerank_residual_analysis module is not implemented yet")


def test_residual_buckets_separate_candidate_miss_from_rerank_residual() -> None:
    module = _load_module()
    e0 = [
        {
            "question_id": "Q1",
            "question": "q1",
            "relevant_document_ids": ["g1"],
            "raw_chunk_ids": ["c1", "c1b"],
            "raw_document_ids": ["g1", "g1"],
        },
        {
            "question_id": "Q2",
            "question": "q2",
            "relevant_document_ids": ["g2"],
            "raw_chunk_ids": ["c2", "c3", "c4"],
            "raw_document_ids": ["g2", "x", "g2"],
        },
        {
            "question_id": "Q3",
            "question": "q3",
            "relevant_document_ids": ["g3"],
            "raw_chunk_ids": ["c5", "c6"],
            "raw_document_ids": ["x", "y"],
        },
    ]
    e1 = [
        {
            "question_id": "Q1",
            "dense_chunk_ids": ["c1", "c1b"],
            "document_ranking": ["g1"],
        },
        {
            "question_id": "Q2",
            "dense_chunk_ids": ["c2", "c3", "c4"],
            "document_ranking": ["x"],
        },
        {
            "question_id": "Q3",
            "dense_chunk_ids": ["c5", "c6"],
            "document_ranking": ["x", "y"],
        },
    ]

    records = module.build_residual_records(e0, e1)

    assert [record.residual_bucket for record in records] == [
        "resolved_top20",
        "rerank_residual_top20",
        "dense_candidate_miss_top100",
    ]
    assert records[0].dense_candidate_document_ids == ("g1",)
    assert records[1].dense_candidate_document_ids == ("g2", "x")


def test_residual_analysis_rejects_dense_candidate_identity_drift() -> None:
    module = _load_module()
    e0 = [
        {
            "question_id": "Q1",
            "question": "q1",
            "relevant_document_ids": ["g1"],
            "raw_chunk_ids": ["c1", "c2"],
            "raw_document_ids": ["g1", "x"],
        }
    ]
    e1 = [
        {
            "question_id": "Q1",
            "dense_chunk_ids": ["c1", "DIFFERENT"],
            "document_ranking": ["g1"],
        }
    ]

    with pytest.raises(RuntimeError, match="dense candidate chunk identity mismatch"):
        module.build_residual_records(e0, e1)


def test_residual_analysis_rejects_question_id_set_drift() -> None:
    module = _load_module()
    e0 = [
        {
            "question_id": "Q1",
            "question": "q1",
            "relevant_document_ids": ["g1"],
            "raw_chunk_ids": ["c1"],
            "raw_document_ids": ["g1"],
        }
    ]
    e1 = [
        {
            "question_id": "Q2",
            "dense_chunk_ids": ["c1"],
            "document_ranking": ["g1"],
        }
    ]

    with pytest.raises(RuntimeError, match="question_id mismatch"):
        module.build_residual_records(e0, e1)
