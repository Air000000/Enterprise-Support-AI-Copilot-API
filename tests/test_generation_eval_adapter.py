from types import SimpleNamespace

import pytest

from experiments.evals.adapters.techqa import TechQAGenerationCase
from experiments.evals.eval_techqa_generation import (
    DEFAULT_GENERATION_TOP_K,
    DEFAULT_REFUSAL_ANSWER,
    DEFAULT_REFUSAL_MAX_DISTANCE,
    GenerationJudgeResult,
    build_techqa_retrieved_context,
    evaluate_techqa_generation_cases,
    load_frozen_techqa_generation_cases,
)


def test_load_frozen_generation_cases_preserves_original_train_dev_split():
    observed: list[tuple[str, str, str]] = []
    rows = []

    for index in range(450):
        rows.append(
            {
                "id": f"TRAIN_A{index:03d}",
                "question": f"answerable train {index}",
                "answer": f"answer {index}",
                "is_impossible": False,
                "contexts": ["gold context must not be used for retrieval"],
            }
        )
    for index in range(150):
        rows.append(
            {
                "id": f"TRAIN_I{index:03d}",
                "question": f"impossible train {index}",
                "answer": "",
                "is_impossible": True,
                "contexts": [],
            }
        )
    for index in range(160):
        rows.append(
            {
                "id": f"DEV_A{index:03d}",
                "question": f"answerable dev {index}",
                "answer": f"answer {index}",
                "is_impossible": False,
                "contexts": ["gold context"],
            }
        )
    for index in range(150):
        rows.append(
            {
                "id": f"DEV_I{index:03d}",
                "question": f"impossible dev {index}",
                "answer": "",
                "is_impossible": True,
                "contexts": [],
            }
        )

    def fake_loader(path, *, split, revision):
        observed.append((path, split, revision))
        return rows

    cases = load_frozen_techqa_generation_cases(dataset_loader=fake_loader)

    assert observed == [
        (
            "nvidia/TechQA-RAG-Eval",
            "train",
            "0b5bbc84b7f07d6d09d063130e90b716d8d4a32a",
        )
    ]
    assert len(cases) == 910
    assert sum(case.split == "train" for case in cases) == 600
    assert sum(case.split == "train" and case.answerable for case in cases) == 450
    assert sum(case.split == "train" and not case.answerable for case in cases) == 150
    assert sum(case.split == "dev" for case in cases) == 310


def test_context_is_built_only_from_actual_retrieved_chunks():
    results = [
        SimpleNamespace(
            chunk_id="doc_a_chunk_0",
            document_id="doc_a",
            chunk_index=0,
            content="actual retrieved evidence A",
            distance=0.12,
        ),
        SimpleNamespace(
            chunk_id="doc_b_chunk_2",
            document_id="doc_b",
            chunk_index=2,
            content="actual retrieved evidence B",
            distance=0.20,
        ),
    ]

    context = build_techqa_retrieved_context(results)

    assert "actual retrieved evidence A" in context
    assert "actual retrieved evidence B" in context
    assert "doc_a_chunk_0" in context
    assert "doc_b_chunk_2" in context
    assert "gold" not in context.lower()


def test_answerable_case_uses_top3_actual_context_and_judge_after_system_latency():
    case = TechQAGenerationCase(
        question_id="TRAIN_Q001",
        question="How do I fix the service?",
        gold_answer="Restart the service.",
        answerable=True,
        split="train",
    )
    search_calls: list[tuple[str, int]] = []
    generation_calls: list[tuple[str, str]] = []
    judge_calls: list[dict[str, object]] = []

    retrieved = [
        SimpleNamespace(
            chunk_id="doc_gold_chunk_0",
            document_id="doc_gold",
            chunk_index=0,
            content="Restart the service to recover it.",
            distance=0.25,
        ),
        SimpleNamespace(
            chunk_id="doc_other_chunk_0",
            document_id="doc_other",
            chunk_index=0,
            content="Other troubleshooting information.",
            distance=0.40,
        ),
    ]

    def fake_searcher(query: str, *, top_k: int):
        search_calls.append((query, top_k))
        return retrieved

    def fake_generator(question: str, context: str) -> str:
        generation_calls.append((question, context))
        return "Restart the service."

    def fake_judge(**kwargs):
        judge_calls.append(kwargs)
        return GenerationJudgeResult(
            correctness_score=1.0,
            correctness_reason="correct",
            faithfulness_score=1.0,
            faithfulness_reason="grounded",
        )

    clock = iter([0.000, 0.250]).__next__
    summary = evaluate_techqa_generation_cases(
        [case],
        searcher=fake_searcher,
        generator=fake_generator,
        judge=fake_judge,
        split="train",
        clock=clock,
    )

    assert DEFAULT_GENERATION_TOP_K == 3
    assert DEFAULT_REFUSAL_MAX_DISTANCE == 0.9
    assert search_calls == [(case.question, 3)]
    assert len(generation_calls) == 1
    assert "Restart the service to recover it." in generation_calls[0][1]
    assert len(judge_calls) == 1
    assert judge_calls[0]["retrieval_context"] == [
        "Restart the service to recover it.",
        "Other troubleshooting information.",
    ]

    result = summary.results[0]
    assert result.generated_answer == "Restart the service."
    assert result.retrieval_status == "ok"
    assert result.abstained is False
    assert result.correctness_score == 1.0
    assert result.faithfulness_score == 1.0
    assert result.e2e_latency_ms == pytest.approx(250.0)


def test_impossible_cases_use_deterministic_abstention_without_llm_judge():
    cases = [
        TechQAGenerationCase(
            question_id="TRAIN_I001",
            question="unsupported one",
            gold_answer="",
            answerable=False,
            split="train",
        ),
        TechQAGenerationCase(
            question_id="TRAIN_I002",
            question="unsupported two",
            gold_answer="",
            answerable=False,
            split="train",
        ),
    ]
    rankings = {
        "unsupported one": [
            SimpleNamespace(
                chunk_id="a0",
                document_id="doc_a",
                chunk_index=0,
                content="unrelated evidence",
                distance=1.10,
            )
        ],
        "unsupported two": [
            SimpleNamespace(
                chunk_id="b0",
                document_id="doc_b",
                chunk_index=0,
                content="plausible but wrong evidence",
                distance=0.40,
            )
        ],
    }
    generated: list[str] = []

    def fake_searcher(query: str, *, top_k: int):
        return rankings[query]

    def fake_generator(question: str, context: str) -> str:
        generated.append(question)
        return "This is an unsupported answer."

    def forbidden_judge(**kwargs):
        raise AssertionError("Impossible cases must not call the LLM judge")

    clock = iter([0.000, 0.010, 0.010, 0.030]).__next__
    summary = evaluate_techqa_generation_cases(
        cases,
        searcher=fake_searcher,
        generator=fake_generator,
        judge=forbidden_judge,
        split="train",
        clock=clock,
    )

    first, second = summary.results
    assert first.generated_answer == DEFAULT_REFUSAL_ANSWER
    assert first.retrieval_status == "refused_low_relevance"
    assert first.abstained is True
    assert first.hallucinated is False

    assert generated == ["unsupported two"]
    assert second.abstained is False
    assert second.hallucinated is True

    assert summary.answerable_count == 0
    assert summary.impossible_count == 2
    assert summary.abstention_accuracy == pytest.approx(0.5)
    assert summary.hallucination_rate == pytest.approx(0.5)
    assert summary.e2e_latency_p50_ms == pytest.approx(15.0)
    assert summary.e2e_latency_p95_ms == pytest.approx(19.5)


def test_g0_retrieval_uses_dense100_rerank_and_returns_top3_context():
    from experiments.evals import eval_techqa_generation as generation_eval

    question = "How do I fix the service?   "

    dense_results = [
        SimpleNamespace(
            chunk_id=f"chunk_{index:03d}",
            document_id=f"doc_{index:03d}",
            chunk_index=index,
            content=f"candidate evidence {index}",
            distance=0.01 + index * 0.001,
        )
        for index in range(100)
    ]

    dense_calls: list[tuple[str, int]] = []
    rerank_calls: list[tuple[str, list[object]]] = []

    def fake_dense_searcher(query: str, *, top_k: int):
        dense_calls.append((query, top_k))
        return dense_results

    def fake_reranker(query: str, candidates):
        rerank_calls.append((query, list(candidates)))

        return SimpleNamespace(
            results=tuple(reversed(candidates)),
            request_id="g0-test-request",
            total_tokens=123,
        )

    outcome = generation_eval.retrieve_g0_e1_context(
        question,
        dense_searcher=fake_dense_searcher,
        reranker=fake_reranker,
    )

    assert generation_eval.DEFAULT_GENERATION_CANDIDATE_K == 100
    assert generation_eval.DEFAULT_GENERATION_TOP_K == 3

    assert dense_calls == [
        (question, 100)
    ]

    assert len(rerank_calls) == 1
    rerank_query, rerank_candidates = rerank_calls[0]

    assert rerank_query == question.rstrip()
    assert len(rerank_candidates) == 100

    assert outcome.dense_top_distance == pytest.approx(0.01)

    assert [
        result.chunk_id
        for result in outcome.results
    ] == [
        "chunk_099",
        "chunk_098",
        "chunk_097",
    ]

    assert [
        result.content
        for result in outcome.results
    ] == [
        "candidate evidence 99",
        "candidate evidence 98",
        "candidate evidence 97",
    ]


def test_generation_evaluator_consumes_g0_retrieval_outcome():
    from experiments.evals import eval_techqa_generation as generation_eval

    case = TechQAGenerationCase(
        question_id="TRAIN_Q001",
        question="How do I fix the service?",
        gold_answer="Restart the service.",
        answerable=True,
        split="train",
    )

    retriever_calls: list[str] = []
    generation_calls: list[tuple[str, str]] = []
    judge_calls: list[dict[str, object]] = []

    def fake_retriever(question: str):
        retriever_calls.append(question)

        return generation_eval.G0RetrievalOutcome(
            results=(
                generation_eval.G0RetrievedChunk(
                    chunk_id="reranked_1",
                    document_id="doc_1",
                    chunk_index=1,
                    content="Restart the service.",
                    distance=1.20,
                ),
                generation_eval.G0RetrievedChunk(
                    chunk_id="reranked_2",
                    document_id="doc_2",
                    chunk_index=2,
                    content="Secondary evidence.",
                    distance=1.10,
                ),
                generation_eval.G0RetrievedChunk(
                    chunk_id="reranked_3",
                    document_id="doc_3",
                    chunk_index=3,
                    content="Additional evidence.",
                    distance=1.00,
                ),
            ),
            dense_top_distance=0.25,
        )

    def fake_generator(
        question: str,
        context: str,
    ) -> str:
        generation_calls.append(
            (question, context)
        )
        return "Restart the service."

    def fake_judge(**kwargs):
        judge_calls.append(kwargs)

        return GenerationJudgeResult(
            correctness_score=1.0,
            correctness_reason="correct",
            faithfulness_score=1.0,
            faithfulness_reason="grounded",
        )

    clock = iter(
        [0.000, 0.100]
    ).__next__

    summary = generation_eval.evaluate_techqa_generation_cases(
        [case],
        retriever=fake_retriever,
        generator=fake_generator,
        judge=fake_judge,
        split="train",
        clock=clock,
    )

    assert retriever_calls == [
        case.question
    ]

    assert len(generation_calls) == 1
    assert len(judge_calls) == 1

    result = summary.results[0]

    assert result.retrieved_chunk_ids == (
        "reranked_1",
        "reranked_2",
        "reranked_3",
    )

    assert result.retrieved_document_ids == (
        "doc_1",
        "doc_2",
        "doc_3",
    )

    # Refusal semantics remain tied to the
    # original Dense Top1 distance.
    #
    # The reranked first chunk has distance 1.20,
    # which would be rejected by the old direct
    # Top1 interpretation. Dense Top1 was 0.25,
    # so G0 should proceed with generation.
    assert result.top_distance == pytest.approx(
        0.25
    )

    assert result.retrieval_status == "ok"
    assert result.abstained is False

    assert (
        result.generated_answer
        == "Restart the service."
    )

    assert result.retrieval_context == (
        "Restart the service.",
        "Secondary evidence.",
        "Additional evidence.",
    )

    assert result.correctness_score == 1.0
    assert result.faithfulness_score == 1.0
    assert result.e2e_latency_ms == pytest.approx(
        100.0
    )
