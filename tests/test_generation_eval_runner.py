import json

import pytest

from experiments.evals.adapters.techqa import TechQAGenerationCase
from experiments.evals.eval_techqa_generation import (
    TechQAGenerationEvalResult,
    load_generation_checkpoint,
    run_resumable_generation_eval,
    write_generation_reports,
)


def _case(question_id: str, *, answerable: bool = True) -> TechQAGenerationCase:
    return TechQAGenerationCase(
        question_id=question_id,
        question=f"question {question_id}",
        gold_answer="gold answer" if answerable else "",
        answerable=answerable,
        split="train",
    )


def _result(
    question_id: str,
    *,
    answerable: bool = True,
    latency_ms: float = 100.0,
) -> TechQAGenerationEvalResult:
    return TechQAGenerationEvalResult(
        question_id=question_id,
        question=f"question {question_id}",
        gold_answer="gold answer" if answerable else "",
        answerable=answerable,
        retrieved_chunk_ids=("doc_chunk_0",),
        retrieved_document_ids=("doc",),
        retrieval_context=("retrieved evidence",),
        generated_answer=(
            "generated answer"
            if answerable
            else "我在已提供资料中没有找到足够依据。"
        ),
        retrieval_status="ok" if answerable else "refused_low_relevance",
        top_distance=0.2 if answerable else 1.1,
        abstained=not answerable,
        hallucinated=False,
        correctness_score=0.8 if answerable else None,
        correctness_reason="correct" if answerable else None,
        faithfulness_score=0.9 if answerable else None,
        faithfulness_reason="grounded" if answerable else None,
        e2e_latency_ms=latency_ms,
    )


def test_resumable_runner_skips_checkpointed_cases_and_appends_only_missing(tmp_path):
    checkpoint_path = tmp_path / "train_checkpoint.jsonl"
    existing = _result("TRAIN_Q001", latency_ms=100.0)
    checkpoint_path.write_text(
        json.dumps(existing.__dict__, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    evaluated: list[str] = []

    def fake_evaluator(case: TechQAGenerationCase) -> TechQAGenerationEvalResult:
        evaluated.append(case.question_id)
        return _result(case.question_id, latency_ms=200.0)

    summary = run_resumable_generation_eval(
        [_case("TRAIN_Q001"), _case("TRAIN_Q002")],
        evaluator=fake_evaluator,
        checkpoint_path=checkpoint_path,
        split="train",
    )

    assert evaluated == ["TRAIN_Q002"]
    assert summary.query_count == 2
    assert [result.question_id for result in summary.results] == [
        "TRAIN_Q001",
        "TRAIN_Q002",
    ]

    persisted = load_generation_checkpoint(checkpoint_path)
    assert [result.question_id for result in persisted] == [
        "TRAIN_Q001",
        "TRAIN_Q002",
    ]


def test_resumable_runner_persists_each_result_before_later_failure(tmp_path):
    checkpoint_path = tmp_path / "train_checkpoint.jsonl"
    calls: list[str] = []

    def flaky_evaluator(case: TechQAGenerationCase) -> TechQAGenerationEvalResult:
        calls.append(case.question_id)
        if case.question_id == "TRAIN_Q002":
            raise RuntimeError("simulated provider failure")
        return _result(case.question_id)

    with pytest.raises(RuntimeError, match="simulated provider failure"):
        run_resumable_generation_eval(
            [_case("TRAIN_Q001"), _case("TRAIN_Q002")],
            evaluator=flaky_evaluator,
            checkpoint_path=checkpoint_path,
            split="train",
        )

    assert calls == ["TRAIN_Q001", "TRAIN_Q002"]
    persisted = load_generation_checkpoint(checkpoint_path)
    assert [result.question_id for result in persisted] == ["TRAIN_Q001"]


def test_completed_checkpoint_rebuilds_summary_and_reports_without_evaluator(tmp_path):
    checkpoint_path = tmp_path / "train_checkpoint.jsonl"
    results = [
        _result("TRAIN_Q001", latency_ms=100.0),
        _result("TRAIN_I001", answerable=False, latency_ms=300.0),
    ]
    checkpoint_path.write_text(
        "".join(
            json.dumps(result.__dict__, ensure_ascii=False) + "\n"
            for result in results
        ),
        encoding="utf-8",
    )

    def forbidden_evaluator(case: TechQAGenerationCase) -> TechQAGenerationEvalResult:
        raise AssertionError("completed checkpoint must not call evaluator")

    summary = run_resumable_generation_eval(
        [_case("TRAIN_Q001"), _case("TRAIN_I001", answerable=False)],
        evaluator=forbidden_evaluator,
        checkpoint_path=checkpoint_path,
        split="train",
    )

    assert summary.query_count == 2
    assert summary.answerable_count == 1
    assert summary.impossible_count == 1
    assert summary.correctness_mean == pytest.approx(0.8)
    assert summary.faithfulness_mean == pytest.approx(0.9)
    assert summary.abstention_accuracy == pytest.approx(1.0)
    assert summary.hallucination_rate == pytest.approx(0.0)
    assert summary.e2e_latency_p50_ms == pytest.approx(200.0)
    assert summary.e2e_latency_p95_ms == pytest.approx(290.0)

    report_dir = tmp_path / "reports"
    write_generation_reports(summary, report_dir=report_dir)

    metrics = json.loads(
        (report_dir / "train_generation_metrics.json").read_text(encoding="utf-8")
    )
    result_lines = (report_dir / "train_generation_results.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()

    assert metrics["query_count"] == 2
    assert metrics["correctness_mean"] == pytest.approx(0.8)
    assert metrics["faithfulness_mean"] == pytest.approx(0.9)
    assert len(result_lines) == 2
