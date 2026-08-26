import importlib

import pytest

from experiments.evals.eval_techqa_generation import TechQAGenerationEvalResult
from experiments.evals.eval_techqa_retrieval import TechQARetrievalResult


def _analysis_module():
    try:
        return importlib.import_module("experiments.evals.failure_analysis")
    except ModuleNotFoundError:
        pytest.fail("experiments.evals.failure_analysis is not implemented yet")


def _retrieval(
    question_id: str,
    *,
    question: str | None = None,
    gold_document_id: str = "gold",
    document_ranking: tuple[str, ...] = ("gold", "d2", "d3"),
) -> TechQARetrievalResult:
    return TechQARetrievalResult(
        question_id=question_id,
        question=question or f"question {question_id}",
        relevant_document_ids=(gold_document_id,),
        raw_chunk_ids=("c1", "c2", "c3"),
        raw_document_ids=("d1", "d2", "d3"),
        document_ranking=document_ranking,
        latency_ms=10.0,
    )


def _generation(
    question_id: str,
    *,
    question: str | None = None,
    retrieved_document_ids: tuple[str, ...] = ("gold", "d2", "d3"),
    retrieval_status: str = "ok",
    abstained: bool = False,
    correctness_score: float = 0.9,
) -> TechQAGenerationEvalResult:
    return TechQAGenerationEvalResult(
        question_id=question_id,
        question=question or f"question {question_id}",
        gold_answer="gold answer",
        answerable=True,
        retrieved_chunk_ids=("c1", "c2", "c3"),
        retrieved_document_ids=retrieved_document_ids,
        retrieval_context=("ctx1", "ctx2", "ctx3"),
        generated_answer=(
            "我在已提供资料中没有找到足够依据。" if abstained else "generated answer"
        ),
        retrieval_status=retrieval_status,
        top_distance=0.2,
        abstained=abstained,
        hallucinated=False,
        correctness_score=correctness_score,
        correctness_reason="correctness reason",
        faithfulness_score=0.9,
        faithfulness_reason="faithfulness reason",
        e2e_latency_ms=100.0,
    )


def test_failure_analysis_classifies_observable_pipeline_buckets():
    analysis = _analysis_module()

    retrieval_results = [
        _retrieval("TRAIN_Q000", document_ranking=("d1", "d2", "d3")),
        _retrieval(
            "TRAIN_Q001",
            document_ranking=("d1", "d2", "d3", "d4", "d5", "gold"),
        ),
        _retrieval(
            "TRAIN_Q002",
            document_ranking=("d1", "d2", "d3", "gold", "d5"),
        ),
        _retrieval(
            "TRAIN_Q003",
            document_ranking=("d1", "gold", "d3", "d4"),
        ),
        _retrieval("TRAIN_Q004"),
        _retrieval("TRAIN_Q005"),
        _retrieval("TRAIN_Q006"),
        _retrieval("TRAIN_Q007"),
        _retrieval("TRAIN_Q008"),
    ]
    generation_results = [
        _generation("TRAIN_Q000", retrieved_document_ids=("d1", "d2", "d3")),
        _generation("TRAIN_Q001", retrieved_document_ids=("d1", "d2", "d3")),
        _generation("TRAIN_Q002", retrieved_document_ids=("d1", "d2", "d3")),
        _generation("TRAIN_Q003", retrieved_document_ids=("d1", "d1", "d1")),
        _generation(
            "TRAIN_Q004",
            retrieval_status="refused_low_relevance",
            abstained=True,
            correctness_score=0.0,
        ),
        _generation("TRAIN_Q005", abstained=True, correctness_score=0.0),
        _generation("TRAIN_Q006", correctness_score=0.4),
        _generation("TRAIN_Q007", correctness_score=0.6),
        _generation("TRAIN_Q008", correctness_score=0.8),
    ]

    records = analysis.build_generation_failure_analysis(
        reversed(retrieval_results),
        reversed(generation_results),
    )

    assert [record.question_id for record in records] == [
        f"TRAIN_Q00{index}" for index in range(9)
    ]
    assert [record.failure_bucket for record in records] == [
        "retrieval_miss_top20",
        "retrieval_rank_6_20",
        "retrieval_rank_4_5",
        "top3_chunk_admission_gap",
        "gold_in_context_threshold_refusal",
        "gold_in_context_model_abstain",
        "gold_in_context_low_correctness",
        "gold_in_context_mid_correctness",
        "gold_in_context_high_correctness",
    ]

    assert records[0].gold_document_rank is None
    assert records[1].gold_document_rank == 6
    assert records[2].gold_document_rank == 4
    assert records[3].gold_document_rank == 2
    assert records[6].gold_in_generation_context is True


def test_failure_analysis_filters_impossible_generation_rows_and_requires_exact_join():
    analysis = _analysis_module()

    impossible = _generation("TRAIN_Q999")
    impossible = TechQAGenerationEvalResult(
        **{**impossible.__dict__, "answerable": False}
    )

    records = analysis.build_generation_failure_analysis(
        [_retrieval("TRAIN_Q000")],
        [_generation("TRAIN_Q000"), impossible],
    )
    assert [record.question_id for record in records] == ["TRAIN_Q000"]

    with pytest.raises(ValueError, match="ID mismatch"):
        analysis.build_generation_failure_analysis(
            [_retrieval("TRAIN_Q000")],
            [_generation("TRAIN_Q001")],
        )

    with pytest.raises(ValueError, match="question mismatch"):
        analysis.build_generation_failure_analysis(
            [_retrieval("TRAIN_Q000", question="retrieval question")],
            [_generation("TRAIN_Q000", question="generation question")],
        )


def test_failure_analysis_allows_trailing_whitespace_but_rejects_content_mismatch():
    analysis = _analysis_module()

    records = analysis.build_generation_failure_analysis(
        [_retrieval("TRAIN_Q000", question="same question")],
        [_generation("TRAIN_Q000", question="same question \n")],
    )
    assert [record.question_id for record in records] == ["TRAIN_Q000"]

    with pytest.raises(ValueError, match="question mismatch"):
        analysis.build_generation_failure_analysis(
            [_retrieval("TRAIN_Q000", question="same question")],
            [_generation("TRAIN_Q000", question="different question \n")],
        )


def test_failure_analysis_rejects_duplicate_ids_and_missing_correctness():
    analysis = _analysis_module()

    duplicate_retrieval = _retrieval("TRAIN_Q000")
    with pytest.raises(ValueError, match="duplicate retrieval"):
        analysis.build_generation_failure_analysis(
            [duplicate_retrieval, duplicate_retrieval],
            [_generation("TRAIN_Q000")],
        )

    missing_score = _generation("TRAIN_Q000")
    missing_score = TechQAGenerationEvalResult(
        **{**missing_score.__dict__, "correctness_score": None}
    )
    with pytest.raises(ValueError, match="correctness_score"):
        analysis.build_generation_failure_analysis(
            [_retrieval("TRAIN_Q000")],
            [missing_score],
        )
