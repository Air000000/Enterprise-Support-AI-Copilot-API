import json

import pytest

import experiments.evals.eval_techqa_generation as generation_eval
from experiments.evals.adapters.techqa import TechQAGenerationCase
from experiments.evals.eval_techqa_generation import (
    TechQAGenerationEvalResult,
    load_generation_checkpoint,
    run_resumable_generation_eval,
    write_generation_reports,
)


def _candidate_document_selector():
    selector = getattr(
        generation_eval,
        "select_candidate_document_ids",
        None,
    )
    assert selector is not None, "select_candidate_document_ids is not implemented"
    return selector


def _document_local_pool_builder():
    builder = getattr(
        generation_eval,
        "build_document_local_candidate_pool",
        None,
    )
    assert builder is not None, "build_document_local_candidate_pool is not implemented"
    return builder


def _retrieved_chunk(
    chunk_id: str,
    document_id: str,
    chunk_index: int,
    distance: float,
) -> generation_eval.G0RetrievedChunk:
    return generation_eval.G0RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        content=f"content {chunk_id}",
        distance=distance,
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


def test_candidate_document_selection_deduplicates_repeated_document_slots():
    selector = _candidate_document_selector()
    ranked = (
        _retrieved_chunk("A_chunk_3", "A", 3, 0.10),
        _retrieved_chunk("A_chunk_5", "A", 5, 0.12),
        _retrieved_chunk("B_chunk_2", "B", 2, 0.15),
        _retrieved_chunk("A_chunk_1", "A", 1, 0.18),
        _retrieved_chunk("C_chunk_4", "C", 4, 0.20),
        _retrieved_chunk("D_chunk_2", "D", 2, 0.25),
    )

    assert selector(ranked, limit=3) == ("A", "B", "C")


@pytest.mark.parametrize("limit", [0, -1])
def test_candidate_document_selection_requires_positive_limit(limit):
    selector = _candidate_document_selector()

    with pytest.raises(ValueError):
        selector((), limit=limit)


def test_document_local_candidate_pool_preserves_document_and_loader_order():
    builder = _document_local_pool_builder()
    chunks_by_document = {
        "B": (
            _retrieved_chunk("B_chunk_4", "B", 4, 0.20),
            _retrieved_chunk("B_chunk_1", "B", 1, 0.22),
            _retrieved_chunk("B_chunk_3", "B", 3, 0.25),
        ),
        "A": (
            _retrieved_chunk("A_chunk_5", "A", 5, 0.10),
            _retrieved_chunk("A_chunk_2", "A", 2, 0.12),
        ),
    }

    pool = builder(
        ("B", "A"),
        load_document_chunks=chunks_by_document.__getitem__,
        max_chunks=5,
    )

    assert tuple(chunk.chunk_id for chunk in pool) == (
        "B_chunk_4",
        "B_chunk_1",
        "B_chunk_3",
        "A_chunk_5",
        "A_chunk_2",
    )


def test_document_local_candidate_pool_deduplicates_chunk_ids_globally():
    builder = _document_local_pool_builder()
    chunks_by_document = {
        "A": (
            _retrieved_chunk("A_chunk_1", "A", 1, 0.10),
            _retrieved_chunk("A_chunk_1", "A", 1, 0.10),
            _retrieved_chunk("A_chunk_2", "A", 2, 0.12),
        ),
        "B": (_retrieved_chunk("B_chunk_1", "B", 1, 0.20),),
    }

    pool = builder(
        ("A", "B"),
        load_document_chunks=chunks_by_document.__getitem__,
        max_chunks=4,
    )

    assert tuple(chunk.chunk_id for chunk in pool) == (
        "A_chunk_1",
        "A_chunk_2",
        "B_chunk_1",
    )


def test_document_local_candidate_pool_stops_at_total_budget():
    builder = _document_local_pool_builder()
    calls: list[str] = []
    chunks_by_document = {
        "A": (
            _retrieved_chunk("A_chunk_1", "A", 1, 0.10),
            _retrieved_chunk("A_chunk_2", "A", 2, 0.12),
        ),
        "B": (
            _retrieved_chunk("B_chunk_1", "B", 1, 0.20),
            _retrieved_chunk("B_chunk_2", "B", 2, 0.25),
        ),
        "C": (_retrieved_chunk("C_chunk_1", "C", 1, 0.30),),
    }

    def load_document_chunks(document_id: str):
        calls.append(document_id)
        return chunks_by_document[document_id]

    pool = builder(
        ("A", "B", "C"),
        load_document_chunks=load_document_chunks,
        max_chunks=3,
    )

    assert tuple(chunk.chunk_id for chunk in pool) == (
        "A_chunk_1",
        "A_chunk_2",
        "B_chunk_1",
    )
    assert calls == ["A", "B"]


def test_document_local_candidate_pool_rejects_wrong_document_membership():
    builder = _document_local_pool_builder()
    chunks_by_document = {
        "A": (_retrieved_chunk("shared_chunk", "A", 1, 0.10),),
        "B": (_retrieved_chunk("shared_chunk", "A", 1, 0.10),),
    }

    with pytest.raises(RuntimeError, match="document_id"):
        builder(
            ("A", "B"),
            load_document_chunks=chunks_by_document.__getitem__,
            max_chunks=2,
        )


@pytest.mark.parametrize("max_chunks", [0, -1])
def test_document_local_candidate_pool_requires_positive_budget(max_chunks):
    builder = _document_local_pool_builder()

    with pytest.raises(ValueError):
        builder(
            (),
            load_document_chunks=lambda _: (),
            max_chunks=max_chunks,
        )


def test_default_single_case_evaluator_uses_frozen_g0_retriever(monkeypatch):
    from types import SimpleNamespace

    import experiments.evals.eval_techqa_generation as generation_eval

    case = _case("TRAIN_Q001")
    expected_result = _result("TRAIN_Q001")

    observed: dict[str, object] = {}

    def fake_evaluate(cases, **kwargs):
        observed["cases"] = list(cases)
        observed["split"] = kwargs.get("split")
        observed["retriever"] = kwargs.get("retriever")

        return SimpleNamespace(
            results=(expected_result,)
        )

    monkeypatch.setattr(
        generation_eval,
        "evaluate_techqa_generation_cases",
        fake_evaluate,
    )

    actual = generation_eval._evaluate_single_generation_case(
        case
    )

    assert actual is expected_result
    assert observed["cases"] == [case]
    assert observed["split"] == "train"

    assert (
        observed["retriever"]
        is generation_eval.retrieve_g0_e1_context
    )


def test_g0_train_pilot_selection_is_deterministic_and_balanced():
    import hashlib

    from experiments.evals import eval_techqa_generation as generation_eval

    train_answerable = [
        TechQAGenerationCase(
            question_id=f"TRAIN_Q{index:03d}",
            question=f"answerable question {index}",
            gold_answer=f"answer {index}",
            answerable=True,
            split="train",
        )
        for index in range(12)
    ]

    train_impossible = [
        TechQAGenerationCase(
            question_id=f"TRAIN_I{index:03d}",
            question=f"impossible question {index}",
            gold_answer="",
            answerable=False,
            split="train",
        )
        for index in range(12)
    ]

    dev_noise = [
        TechQAGenerationCase(
            question_id=f"DEV_Q{index:03d}",
            question=f"dev question {index}",
            gold_answer=f"dev answer {index}",
            answerable=True,
            split="dev",
        )
        for index in range(4)
    ]

    all_cases = (
        train_answerable
        + train_impossible
        + dev_noise
    )

    selected = generation_eval.select_g0_train_pilot_cases(
        all_cases
    )

    selected_from_reversed = (
        generation_eval.select_g0_train_pilot_cases(
            list(reversed(all_cases))
        )
    )

    assert generation_eval.DEFAULT_G0_PILOT_SIZE == 12
    assert generation_eval.DEFAULT_G0_PILOT_PER_CLASS == 6
    assert (
        generation_eval.DEFAULT_G0_PILOT_SEED
        == "techqa-g0-generation-pilot-v1"
    )

    assert len(selected) == 12

    assert sum(
        case.answerable
        for case in selected
    ) == 6

    assert sum(
        not case.answerable
        for case in selected
    ) == 6

    assert all(
        case.split == "train"
        for case in selected
    )

    selected_ids = [
        case.question_id
        for case in selected
    ]

    reversed_ids = [
        case.question_id
        for case in selected_from_reversed
    ]

    assert selected_ids == reversed_ids

    def stable_key(case):
        return hashlib.sha256(
            (
                f"{generation_eval.DEFAULT_G0_PILOT_SEED}:"
                f"{case.question_id}"
            ).encode("utf-8")
        ).hexdigest()

    expected_answerable = sorted(
        train_answerable,
        key=stable_key,
    )[:6]

    expected_impossible = sorted(
        train_impossible,
        key=stable_key,
    )[:6]

    expected_ids = {
        case.question_id
        for case in (
            expected_answerable
            + expected_impossible
        )
    }

    assert set(selected_ids) == expected_ids
