from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pytest

from experiments.evals.adapters.techqa import TechQADocument


def _load_module():
    return import_module("experiments.evals.rerank_residual_analysis")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _synthetic_train_rows() -> tuple[list[dict], list[dict]]:
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


def test_prepare_cli_defaults_are_frozen_train_only() -> None:
    module = _load_module()

    assert module.DEFAULT_E0_TRAIN_RESULTS_PATH == Path(
        "experiments/evals/reports/e0_dense/train_results.jsonl"
    )
    assert module.DEFAULT_E1_TRAIN_RESULTS_PATH == Path(
        "experiments/evals/reports/e1_rerank/train_results.jsonl"
    )
    assert module.DEFAULT_RERANK_REPORT_DIR == Path(
        "experiments/evals/reports/e1_rerank"
    )
    assert module.DEFAULT_EXPECTED_TRAIN_COUNT == 450
    assert module.DEFAULT_REVIEW_SAMPLE_SIZE == 30


def test_prepare_cli_materializes_train_and_reports_zero_provider_calls(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    e0_rows, e1_rows = _synthetic_train_rows()
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

    module.main(
        [
            "prepare",
            "--e0-results",
            str(e0_path),
            "--e1-results",
            str(e1_path),
            "--report-dir",
            str(report_dir),
            "--expected-count",
            "3",
            "--review-sample-size",
            "2",
        ],
        document_loader=lambda: documents,
    )

    output = capsys.readouterr().out
    assert "R2 TRAIN residual evidence materialized." in output
    assert "query_count = 3" in output
    assert "dense_candidate_miss_top100 = 1" in output
    assert "provider_calls = 0" in output
    assert (report_dir / "train_residual_summary.json").exists()
    assert (report_dir / "train_residual_review.jsonl").exists()
