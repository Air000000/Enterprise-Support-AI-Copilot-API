import json
from dataclasses import asdict

import pytest

from experiments.evals.eval_techqa_generation import TechQAGenerationEvalResult
from experiments.evals.eval_techqa_retrieval import TechQARetrievalResult
import experiments.evals.failure_analysis as analysis


def _retrieval(question_id: str, *, ranking: tuple[str, ...]) -> TechQARetrievalResult:
    return TechQARetrievalResult(
        question_id=question_id,
        question=f"question {question_id}",
        relevant_document_ids=("gold",),
        raw_chunk_ids=("c1", "c2", "c3"),
        raw_document_ids=("d1", "d2", "d3"),
        document_ranking=ranking,
        latency_ms=10.0,
    )


def _generation(
    question_id: str,
    *,
    answerable: bool = True,
    retrieved_document_ids: tuple[str, ...] = ("gold", "d2", "d3"),
    correctness_score: float | None = 0.9,
) -> TechQAGenerationEvalResult:
    return TechQAGenerationEvalResult(
        question_id=question_id,
        question=f"question {question_id}",
        gold_answer="gold answer" if answerable else "-",
        answerable=answerable,
        retrieved_chunk_ids=("c1", "c2", "c3"),
        retrieved_document_ids=retrieved_document_ids,
        retrieval_context=("ctx1", "ctx2", "ctx3"),
        generated_answer="generated answer",
        retrieval_status="ok",
        top_distance=0.2,
        abstained=False,
        hallucinated=not answerable,
        correctness_score=correctness_score if answerable else None,
        correctness_reason="reason" if answerable else None,
        faithfulness_score=0.9 if answerable else None,
        faithfulness_reason="reason" if answerable else None,
        e2e_latency_ms=100.0,
    )


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(asdict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_failure_analysis_materializes_detail_and_summary_from_frozen_artifacts(tmp_path):
    retrieval_path = tmp_path / "train_results.jsonl"
    generation_path = tmp_path / "train_generation_results.jsonl"
    report_dir = tmp_path / "reports"

    retrieval_rows = [
        _retrieval("TRAIN_Q000", ranking=("d1", "d2", "d3")),
        _retrieval("TRAIN_Q001", ranking=("d1", "d2", "d3", "gold")),
        _retrieval("TRAIN_Q002", ranking=("gold", "d2", "d3")),
    ]
    generation_rows = [
        _generation("TRAIN_Q000", retrieved_document_ids=("d1", "d2", "d3")),
        _generation("TRAIN_Q001", retrieved_document_ids=("d1", "d2", "d3")),
        _generation("TRAIN_Q002", retrieved_document_ids=("gold", "d2", "d3")),
        _generation("TRAIN_Q999", answerable=False),
    ]
    _write_jsonl(retrieval_path, retrieval_rows)
    _write_jsonl(generation_path, generation_rows)

    summary = analysis.materialize_generation_failure_analysis(
        retrieval_results_path=retrieval_path,
        generation_results_path=generation_path,
        report_dir=report_dir,
    )

    detail_path = report_dir / "train_failure_analysis.jsonl"
    summary_path = report_dir / "train_failure_analysis_summary.json"
    assert detail_path.exists()
    assert summary_path.exists()

    details = [
        json.loads(line)
        for line in detail_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["question_id"] for row in details] == [
        "TRAIN_Q000",
        "TRAIN_Q001",
        "TRAIN_Q002",
    ]

    assert summary["answerable_count"] == 3
    assert summary["bucket_counts"] == {
        "retrieval_miss_top20": 1,
        "retrieval_rank_4_5": 1,
        "gold_in_context_high_correctness": 1,
    }
    assert summary["bucket_rates"] == {
        "retrieval_miss_top20": pytest.approx(1 / 3),
        "retrieval_rank_4_5": pytest.approx(1 / 3),
        "gold_in_context_high_correctness": pytest.approx(1 / 3),
    }
    assert summary["gold_in_generation_context_count"] == 1
    assert summary["gold_not_in_generation_context_count"] == 2
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary

    with pytest.raises(FileExistsError):
        analysis.materialize_generation_failure_analysis(
            retrieval_results_path=retrieval_path,
            generation_results_path=generation_path,
            report_dir=report_dir,
        )
