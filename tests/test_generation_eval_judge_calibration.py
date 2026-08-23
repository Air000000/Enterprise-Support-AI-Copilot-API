import json
from collections import Counter

import experiments.evals.eval_techqa_generation as generation_eval


BUCKET_COUNTS = {
    "answerable_low_correctness_high_faithfulness": 5,
    "answerable_low_both": 5,
    "answerable_high_both": 5,
    "answerable_metric_disagreement": 5,
    "impossible_unsafe_answer": 3,
    "impossible_correct_abstain": 2,
}


def _result(
    question_id: str,
    *,
    answerable: bool = True,
    correctness: float | None = 0.5,
    faithfulness: float | None = 0.5,
    abstained: bool = False,
) -> generation_eval.TechQAGenerationEvalResult:
    return generation_eval.TechQAGenerationEvalResult(
        question_id=question_id,
        question=f"question {question_id}",
        gold_answer="gold answer" if answerable else "",
        answerable=answerable,
        retrieved_chunk_ids=("doc_chunk_0",),
        retrieved_document_ids=("doc",),
        retrieval_context=("retrieved evidence",),
        generated_answer=(
            generation_eval.DEFAULT_REFUSAL_ANSWER
            if abstained
            else f"generated answer {question_id}"
        ),
        retrieval_status="ok",
        top_distance=0.2,
        abstained=abstained,
        hallucinated=(not answerable) and not abstained,
        correctness_score=correctness if answerable else None,
        correctness_reason="automatic correctness reason" if answerable else None,
        faithfulness_score=faithfulness if answerable else None,
        faithfulness_reason="automatic faithfulness reason" if answerable else None,
        e2e_latency_ms=100.0,
    )


def _candidate_results() -> list[generation_eval.TechQAGenerationEvalResult]:
    results: list[generation_eval.TechQAGenerationEvalResult] = []

    for index in range(6):
        results.append(
            _result(
                f"TRAIN_A_LCHF_{index:03d}",
                correctness=0.1,
                faithfulness=0.95,
            )
        )
        results.append(
            _result(
                f"TRAIN_A_LB_{index:03d}",
                correctness=0.1,
                faithfulness=0.2,
            )
        )
        results.append(
            _result(
                f"TRAIN_A_HB_{index:03d}",
                correctness=0.9,
                faithfulness=0.95,
            )
        )
        results.append(
            _result(
                f"TRAIN_A_MD_{index:03d}",
                correctness=0.8,
                faithfulness=0.1,
            )
        )

    for index in range(6):
        results.append(
            _result(
                f"TRAIN_A_MID_{index:03d}",
                correctness=0.5,
                faithfulness=0.5,
            )
        )

    for index in range(5):
        results.append(
            _result(
                f"TRAIN_I_UNSAFE_{index:03d}",
                answerable=False,
                correctness=None,
                faithfulness=None,
                abstained=False,
            )
        )

    for index in range(3):
        results.append(
            _result(
                f"TRAIN_I_ABSTAIN_{index:03d}",
                answerable=False,
                correctness=None,
                faithfulness=None,
                abstained=True,
            )
        )

    return results


def test_judge_calibration_selection_is_fixed_stratified_and_unique():
    selector = getattr(generation_eval, "select_judge_calibration_cases")
    candidates = _candidate_results()

    selected = selector(candidates)
    selected_from_reversed = selector(reversed(candidates))

    assert len(selected) == 25
    assert [record.question_id for record in selected] == [
        record.question_id for record in selected_from_reversed
    ]
    assert len({record.question_id for record in selected}) == 25
    assert Counter(record.selection_bucket for record in selected) == BUCKET_COUNTS

    for record in selected:
        assert record.question
        assert record.generated_answer
        assert record.retrieval_context == ("retrieved evidence",)
        assert record.notes == ""

        if record.answerable:
            assert record.manual_correctness == ""
            assert record.manual_faithfulness == ""
            assert record.manual_abstention == "not_applicable"
        else:
            assert record.manual_correctness == "not_applicable"
            assert record.manual_faithfulness == "not_applicable"
            assert record.manual_abstention == ""


def test_judge_calibration_writer_serializes_review_template_without_overwrite(tmp_path):
    selector = getattr(generation_eval, "select_judge_calibration_cases")
    writer = getattr(generation_eval, "write_judge_calibration_template")
    records = selector(_candidate_results())
    output_path = tmp_path / "judge_calibration.jsonl"

    writer(records, output_path=output_path)

    payloads = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(payloads) == 25
    assert payloads[0]["selection_bucket"]
    assert payloads[0]["question"]
    assert payloads[0]["gold_answer"]
    assert payloads[0]["generated_answer"]
    assert payloads[0]["retrieval_context"] == ["retrieved evidence"]
    assert "correctness_score" in payloads[0]
    assert "correctness_reason" in payloads[0]
    assert "faithfulness_score" in payloads[0]
    assert "faithfulness_reason" in payloads[0]
    assert "manual_correctness" in payloads[0]
    assert "manual_faithfulness" in payloads[0]
    assert "manual_abstention" in payloads[0]
    assert "notes" in payloads[0]

    original = output_path.read_text(encoding="utf-8")

    try:
        writer(records, output_path=output_path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("calibration writer must not overwrite a frozen review set")

    assert output_path.read_text(encoding="utf-8") == original
