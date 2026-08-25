from __future__ import annotations

import hashlib
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


def _residual_record(module, question_id: str, bucket: str):
    return module.RerankResidualRecord(
        question_id=question_id,
        question=f"question {question_id}",
        relevant_document_ids=(f"gold-{question_id}",),
        dense_candidate_document_ids=(f"dense-{question_id}", "shared"),
        e1_document_ranking=(f"e1-{question_id}", "shared"),
        residual_bucket=bucket,
    )


def test_candidate_miss_review_sample_is_hash_stable_and_bucket_scoped() -> None:
    module = _load_module()
    records = [
        _residual_record(module, "TRAIN_Q3", "dense_candidate_miss_top100"),
        _residual_record(module, "TRAIN_Q1", "resolved_top20"),
        _residual_record(module, "TRAIN_Q2", "dense_candidate_miss_top100"),
        _residual_record(module, "TRAIN_Q4", "dense_candidate_miss_top100"),
    ]
    expected_ids = sorted(
        ["TRAIN_Q2", "TRAIN_Q3", "TRAIN_Q4"],
        key=lambda question_id: hashlib.sha256(question_id.encode("utf-8")).hexdigest(),
    )[:2]

    forward = module.select_candidate_miss_review_sample(records, sample_size=2)
    reversed_input = module.select_candidate_miss_review_sample(
        list(reversed(records)),
        sample_size=2,
    )

    assert [record.question_id for record in forward] == expected_ids
    assert [record.question_id for record in reversed_input] == expected_ids
    assert all(
        record.residual_bucket == "dense_candidate_miss_top100" for record in forward
    )


def test_residual_summary_partitions_all_records() -> None:
    module = _load_module()
    records = [
        _residual_record(module, "TRAIN_Q1", "resolved_top20"),
        _residual_record(module, "TRAIN_Q2", "resolved_top20"),
        _residual_record(module, "TRAIN_Q3", "rerank_residual_top20"),
        _residual_record(module, "TRAIN_Q4", "dense_candidate_miss_top100"),
    ]

    summary = module.summarize_residual_records(records)

    assert summary == {
        "query_count": 4,
        "bucket_counts": {
            "dense_candidate_miss_top100": 1,
            "rerank_residual_top20": 1,
            "resolved_top20": 2,
        },
        "bucket_rates": {
            "dense_candidate_miss_top100": 0.25,
            "rerank_residual_top20": 0.25,
            "resolved_top20": 0.5,
        },
        "dense_candidate_miss_count": 1,
    }


def test_build_review_rows_adds_frozen_document_excerpts_and_blank_labels() -> None:
    module = _load_module()
    record = module.RerankResidualRecord(
        question_id="TRAIN_Q9",
        question="why does code X fail?",
        relevant_document_ids=("gold",),
        dense_candidate_document_ids=("dense-1", "dense-2"),
        e1_document_ranking=("e1-1", "dense-2"),
        residual_bucket="dense_candidate_miss_top100",
    )
    documents_by_id = {
        "gold": "gold evidence text",
        "dense-1": "dense candidate one",
        "dense-2": "dense candidate two",
        "e1-1": "reranked candidate one",
    }

    rows = module.build_candidate_miss_review_rows(
        [record],
        documents_by_id=documents_by_id,
        excerpt_chars=80,
    )

    assert rows == [
        {
            "question_id": "TRAIN_Q9",
            "question": "why does code X fail?",
            "gold_documents": [
                {"document_id": "gold", "text_excerpt": "gold evidence text"}
            ],
            "dense_top5": [
                {"document_id": "dense-1", "text_excerpt": "dense candidate one"},
                {"document_id": "dense-2", "text_excerpt": "dense candidate two"},
            ],
            "e1_top5": [
                {"document_id": "e1-1", "text_excerpt": "reranked candidate one"},
                {"document_id": "dense-2", "text_excerpt": "dense candidate two"},
            ],
            "manual_label": "",
            "notes": "",
        }
    ]
