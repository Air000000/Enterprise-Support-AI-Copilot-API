import json
from pathlib import Path
from types import SimpleNamespace

from experiments.evals import refusal_phase_b_runner as shared
from experiments.evals import refusal_v2_ai_proxy_runner as runner
from experiments.evals.refusal_v2_ai_proxy import (
    CLASSIFIER_SYSTEM_PROMPT_V2,
    build_classifier_messages_v2,
)


def _prediction(
    question_id: str,
    decision: str,
) -> shared.PhaseBPrediction:
    return shared.PhaseBPrediction(
        question_id=question_id,
        decision=decision,
        reason="test",
        supporting_source_ids=("Source 1",) if decision == "SUFFICIENT" else (),
        model="qwen3.5-plus-2026-04-20",
        request_id="request",
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        latency_ms=1000.0,
        estimated_cost_cny=0.001,
    )


def _contract() -> dict:
    return json.loads(
        Path(runner.DEFAULT_CONTRACT_PATH).read_text(encoding="utf-8")
    )


def _evaluation_rows(
    *,
    sufficient_errors: int,
    insufficient_errors: int,
) -> tuple[list[shared.PhaseBPrediction], list[dict[str, str]]]:
    predictions = []
    targets = []
    for index in range(20):
        question_id = f"TRAIN_S{index:03d}"
        decision = "INSUFFICIENT" if index < sufficient_errors else "SUFFICIENT"
        predictions.append(_prediction(question_id, decision))
        targets.append(
            {"question_id": question_id, "target_class": "SUFFICIENT"}
        )
    for index in range(20):
        question_id = f"TRAIN_I{index:03d}"
        decision = "SUFFICIENT" if index < insufficient_errors else "INSUFFICIENT"
        predictions.append(_prediction(question_id, decision))
        targets.append(
            {"question_id": question_id, "target_class": "INSUFFICIENT"}
        )
    return predictions, targets


def test_v2_contract_and_proxy_gate_pass() -> None:
    contract = _contract()
    runner.validate_contract(contract)
    predictions, targets = _evaluation_rows(
        sufficient_errors=3,
        insufficient_errors=5,
    )

    summary = runner.evaluate_predictions(
        predictions,
        targets,
        contract=contract,
    )

    assert summary.balanced_accuracy == 0.8
    assert summary.sufficient_recall == 0.85
    assert summary.insufficient_recall == 0.75
    assert summary.decision == "ADMIT_TO_INDEPENDENT_HUMAN_CONFIRMATION"


def test_v2_proxy_gate_rejects_low_insufficient_recall() -> None:
    predictions, targets = _evaluation_rows(
        sufficient_errors=0,
        insufficient_errors=7,
    )

    summary = runner.evaluate_predictions(
        predictions,
        targets,
        contract=_contract(),
    )

    assert summary.insufficient_recall == 0.65
    assert summary.decision == "REJECT_EVIDENCE_SUFFICIENCY_V2_AI_PROXY"


def test_shared_paid_call_uses_frozen_v2_prompt_without_question_id() -> None:
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="request",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "decision": "INSUFFICIENT",
                                    "reason": "Missing material support.",
                                    "supporting_source_ids": [],
                                }
                            )
                        )
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=10,
                    total_tokens=110,
                ),
            )

    row = {
        "question_id": "TRAIN_SECRET_ID",
        "question": "Question",
        "sources": [
            {"source_id": f"Source {rank}", "content": f"Content {rank}"}
            for rank in range(1, 15)
        ],
    }
    shared.classify_one(
        row,
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        ),
        model="qwen3.5-plus-2026-04-20",
        input_rate_cny_per_1m=2.936,
        output_rate_cny_per_1m=17.614,
        messages_builder=build_classifier_messages_v2,
        clock=lambda: 1.0,
    )

    assert captured["messages"][0]["content"] == CLASSIFIER_SYSTEM_PROMPT_V2
    assert "TRAIN_SECRET_ID" not in json.dumps(captured["messages"])
