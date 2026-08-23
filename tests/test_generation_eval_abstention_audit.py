import importlib
import json

import pytest

import experiments.evals.eval_techqa_generation as generation_eval


MANUAL_CATEGORIES = {
    "corpus_supported_impossible",
    "semantic_abstention",
    "true_unsafe_answer",
    "correct_abstain",
}


def _result(
    question_id: str,
    *,
    answerable: bool,
    generated_answer: str,
    abstained: bool,
) -> generation_eval.TechQAGenerationEvalResult:
    return generation_eval.TechQAGenerationEvalResult(
        question_id=question_id,
        question=f"question {question_id}",
        gold_answer="gold" if answerable else "-",
        answerable=answerable,
        retrieved_chunk_ids=("doc_chunk_0",),
        retrieved_document_ids=("doc",),
        retrieval_context=("retrieved evidence",),
        generated_answer=generated_answer,
        retrieval_status="ok",
        top_distance=0.2,
        abstained=abstained,
        hallucinated=(not answerable) and not abstained,
        correctness_score=0.8 if answerable else None,
        correctness_reason="reason" if answerable else None,
        faithfulness_score=1.0 if answerable else None,
        faithfulness_reason="reason" if answerable else None,
        e2e_latency_ms=100.0,
    )


def _audit_module():
    try:
        return importlib.import_module("experiments.evals.abstention_audit")
    except ModuleNotFoundError:
        pytest.fail("experiments.evals.abstention_audit is not implemented yet")


def test_abstention_audit_filters_impossible_cases_and_assigns_only_deterministic_signals():
    audit = _audit_module()
    results = [
        _result(
            "TRAIN_Q003",
            answerable=False,
            generated_answer="A confident answer from context.",
            abstained=False,
        ),
        _result(
            "TRAIN_Q001",
            answerable=False,
            generated_answer=generation_eval.DEFAULT_REFUSAL_ANSWER,
            abstained=True,
        ),
        _result(
            "TRAIN_Q002",
            answerable=False,
            generated_answer=(
                generation_eval.DEFAULT_REFUSAL_ANSWER
                + "\n\nHere is some general context-supported information."
            ),
            abstained=False,
        ),
        _result(
            "TRAIN_Q000",
            answerable=True,
            generated_answer="answerable response",
            abstained=False,
        ),
    ]

    records = audit.build_abstention_audit_records(reversed(results))

    assert [record.question_id for record in records] == [
        "TRAIN_Q001",
        "TRAIN_Q002",
        "TRAIN_Q003",
    ]
    assert [record.automatic_signal for record in records] == [
        "exact_refusal",
        "semantic_refusal_candidate",
        "non_refusal_candidate",
    ]

    for record in records:
        assert record.answerable is False
        assert record.question
        assert record.retrieval_context == ("retrieved evidence",)
        assert record.generated_answer
        assert record.manual_category == ""
        assert record.manual_notes == ""

    assert set(audit.MANUAL_ABSTENTION_AUDIT_CATEGORIES) == MANUAL_CATEGORIES


def test_abstention_audit_writer_serializes_once_without_overwrite(tmp_path):
    audit = _audit_module()
    records = audit.build_abstention_audit_records(
        [
            _result(
                "TRAIN_Q001",
                answerable=False,
                generated_answer=generation_eval.DEFAULT_REFUSAL_ANSWER,
                abstained=True,
            ),
            _result(
                "TRAIN_Q002",
                answerable=False,
                generated_answer="unsupported answer",
                abstained=False,
            ),
        ]
    )
    output_path = tmp_path / "abstention_audit.jsonl"

    audit.write_abstention_audit_template(records, output_path=output_path)

    payloads = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(payloads) == 2
    assert payloads[0]["question_id"] == "TRAIN_Q001"
    assert payloads[0]["automatic_signal"] == "exact_refusal"
    assert payloads[0]["retrieval_context"] == ["retrieved evidence"]
    assert payloads[0]["manual_category"] == ""
    assert payloads[0]["manual_notes"] == ""

    original = output_path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        audit.write_abstention_audit_template(records, output_path=output_path)

    assert output_path.read_text(encoding="utf-8") == original
