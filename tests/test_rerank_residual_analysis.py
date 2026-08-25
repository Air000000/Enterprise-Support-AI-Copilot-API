from __future__ import annotations

import hashlib
import json
from importlib import import_module
from pathlib import Path

import pytest

from experiments.evals.adapters.techqa import TechQADocument


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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _formal_rows() -> tuple[list[dict], list[dict]]:
    e0_rows = [
        {
            "question_id": "TRAIN_Q1",
            "question": "resolved question",
            "relevant_document_ids": ["g1"],
            "raw_chunk_ids": ["c1"],
            "raw_document_ids": ["g1"],
        },
        {
            "question_id": "TRAIN_Q2",
            "question": "rerank residual question",
            "relevant_document_ids": ["g2"],
            "raw_chunk_ids": ["c2", "c3"],
            "raw_document_ids": ["g2", "x"],
        },
        {
            "question_id": "TRAIN_Q3",
            "question": "dense miss question",
            "relevant_document_ids": ["g3"],
            "raw_chunk_ids": ["c4", "c5"],
            "raw_document_ids": ["x", "y"],
        },
    ]
    e1_rows = [
        {
            "question_id": "TRAIN_Q1",
            "dense_chunk_ids": ["c1"],
            "document_ranking": ["g1"],
        },
        {
            "question_id": "TRAIN_Q2",
            "dense_chunk_ids": ["c2", "c3"],
            "document_ranking": ["x"],
        },
        {
            "question_id": "TRAIN_Q3",
            "dense_chunk_ids": ["c4", "c5"],
            "document_ranking": ["x", "y"],
        },
    ]
    return e0_rows, e1_rows


def test_materialize_residual_analysis_writes_train_summary_and_review(tmp_path: Path) -> None:
    module = _load_module()
    e0_rows, e1_rows = _formal_rows()
    e0_path = tmp_path / "e0" / "train_results.jsonl"
    e1_path = tmp_path / "e1" / "train_results.jsonl"
    report_dir = tmp_path / "reports" / "e1_rerank"
    _write_jsonl(e0_path, e0_rows)
    _write_jsonl(e1_path, e1_rows)

    documents = [
        TechQADocument("g1", "resolved gold"),
        TechQADocument("g2", "rerank residual gold"),
        TechQADocument("g3", "dense miss gold"),
        TechQADocument("x", "candidate x"),
        TechQADocument("y", "candidate y"),
    ]

    summary = module.materialize_rerank_residual_analysis(
        e0_results_path=e0_path,
        e1_results_path=e1_path,
        report_dir=report_dir,
        expected_count=3,
        review_sample_size=2,
        document_loader=lambda: documents,
    )

    assert summary["query_count"] == 3
    assert summary["bucket_counts"] == {
        "dense_candidate_miss_top100": 1,
        "rerank_residual_top20": 1,
        "resolved_top20": 1,
    }
    persisted_summary = json.loads(
        (report_dir / "train_residual_summary.json").read_text(encoding="utf-8")
    )
    assert persisted_summary == summary

    review_rows = [
        json.loads(line)
        for line in (report_dir / "train_residual_review.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [row["question_id"] for row in review_rows] == ["TRAIN_Q3"]
    assert review_rows[0]["gold_documents"] == [
        {"document_id": "g3", "text_excerpt": "dense miss gold"}
    ]
    assert review_rows[0]["manual_label"] == ""
    assert review_rows[0]["notes"] == ""


def test_materialize_residual_analysis_checks_count_before_loading_corpus(
    tmp_path: Path,
) -> None:
    module = _load_module()
    e0_rows, e1_rows = _formal_rows()
    e0_path = tmp_path / "e0.jsonl"
    e1_path = tmp_path / "e1.jsonl"
    _write_jsonl(e0_path, e0_rows[:2])
    _write_jsonl(e1_path, e1_rows[:2])

    def fail_if_called():
        raise AssertionError("document loader must not run after count mismatch")

    with pytest.raises(RuntimeError, match="TRAIN row count mismatch"):
        module.materialize_rerank_residual_analysis(
            e0_results_path=e0_path,
            e1_results_path=e1_path,
            report_dir=tmp_path / "out",
            expected_count=3,
            document_loader=fail_if_called,
        )


def test_review_summary_reports_sample_only_counts_and_rates() -> None:
    module = _load_module()
    rows = [
        {"question_id": "TRAIN_Q1", "manual_label": "lexical_candidate"},
        {"question_id": "TRAIN_Q2", "manual_label": "semantic_or_indirect_miss"},
        {"question_id": "TRAIN_Q3", "manual_label": "qrel_or_query_ambiguity"},
        {"question_id": "TRAIN_Q4", "manual_label": "qrel_or_query_ambiguity"},
    ]

    summary = module.summarize_review_rows(rows)

    assert summary == {
        "reviewed_count": 4,
        "label_counts": {
            "lexical_candidate": 1,
            "semantic_or_indirect_miss": 1,
            "qrel_or_query_ambiguity": 2,
        },
        "label_rates": {
            "lexical_candidate": 0.25,
            "semantic_or_indirect_miss": 0.25,
            "qrel_or_query_ambiguity": 0.5,
        },
        "population_rate_claim_allowed": False,
    }


def test_review_summary_rejects_blank_label() -> None:
    module = _load_module()

    with pytest.raises(RuntimeError, match="review label missing"):
        module.summarize_review_rows(
            [{"question_id": "TRAIN_Q1", "manual_label": ""}]
        )


def test_review_summary_rejects_unknown_label() -> None:
    module = _load_module()

    with pytest.raises(RuntimeError, match="unknown review label"):
        module.summarize_review_rows(
            [{"question_id": "TRAIN_Q1", "manual_label": "other"}]
        )


def test_build_r3_gate_freezes_numeric_admission_thresholds() -> None:
    module = _load_module()

    assert module.build_r3_gate(40) == {
        "split": "train",
        "query_count": 450,
        "dense_candidate_chunk_k": 100,
        "bm25_candidate_document_k": 100,
        "hybrid_candidate_document_k": 100,
        "rrf_k": 60,
        "dense_candidate_miss_count": 40,
        "required_recovered_dense_misses": 6,
        "required_net_gain_cases": 5,
        "required_net_gain_pp": 1.1111111111111112,
        "admission_logic": (
            "recovered_dense_misses >= required_recovered_dense_misses AND "
            "hybrid_hit100 - dense_hit100 >= required_net_gain_cases"
        ),
    }
