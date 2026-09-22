import json
from types import SimpleNamespace

import pytest

from experiments.evals import refusal_phase_b_runner as runner


def _contract() -> dict:
    return {
        "run": "refusal_evidence_sufficiency_phase_b_v1",
        "status": "PREREGISTERED_NOT_RUN",
        "population": {
            "total_cases": 54,
            "gated_cases": 51,
            "sufficient_proxy": 35,
            "insufficient_proxy": 16,
            "ambiguous_multi_chunk": 3,
        },
        "frozen_inputs": {
            "classifier_inputs_sha256": runner.EXPECTED_INPUTS_SHA256,
            "targets_sha256": runner.EXPECTED_TARGETS_SHA256,
            "context_policy": "flat_rerank_top14_v1",
            "top_k": 14,
        },
        "classifier": {
            "model": "qwen3.5-plus-2026-04-20",
            "temperature": 0.0,
            "enable_thinking": False,
            "response_format": "json_object",
            "max_output_tokens": 256,
            "max_calls": 54,
        },
        "pricing_snapshot": {
            "hard_cost_cap_cny": 3.0,
        },
        "gate": {
            "balanced_accuracy_min": 0.80,
            "sufficient_recall_min": 0.85,
            "insufficient_recall_min": 0.70,
            "sufficient_to_insufficient_max": 5,
        },
        "controls": {
            "dev_artifact_opened": False,
            "retrieval_calls": 0,
            "rerank_calls": 0,
            "generation_calls": 0,
            "judge_calls": 0,
        },
    }


def _input_row(question_id: str = "TRAIN_Q001") -> dict:
    return {
        "question_id": question_id,
        "question": "How do I fix the service?",
        "sources": [
            {
                "source_id": f"Source {index}",
                "content": f"context {index}",
            }
            for index in range(1, 15)
        ],
    }


def _prediction(
    question_id: str,
    decision: str,
) -> runner.PhaseBPrediction:
    return runner.PhaseBPrediction(
        question_id=question_id,
        decision=decision,
        reason="supported",
        supporting_source_ids=("Source 1",),
        model="qwen3.5-plus-2026-04-20",
        request_id="req",
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        latency_ms=100.0,
        estimated_cost_cny=0.001,
    )


def test_validate_contract_accepts_frozen_contract():
    runner.validate_contract(_contract())


def test_validate_contract_rejects_changed_model():
    contract = _contract()
    contract["classifier"]["model"] = "qwen3.5-plus"

    with pytest.raises(RuntimeError, match="model"):
        runner.validate_contract(contract)


def test_parse_classifier_json_requires_source_for_sufficient():
    with pytest.raises(RuntimeError, match="at least one source"):
        runner._parse_classifier_json(
            json.dumps(
                {
                    "decision": "SUFFICIENT",
                    "reason": "enough",
                    "supporting_source_ids": [],
                }
            ),
            allowed_source_ids={"Source 1"},
        )


def test_parse_classifier_json_rejects_unknown_source():
    with pytest.raises(RuntimeError, match="outside the frozen context"):
        runner._parse_classifier_json(
            json.dumps(
                {
                    "decision": "SUFFICIENT",
                    "reason": "enough",
                    "supporting_source_ids": ["Source 15"],
                }
            ),
            allowed_source_ids={"Source 1"},
        )


def test_classify_one_does_not_send_question_id():
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="req-1",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "decision": "SUFFICIENT",
                                    "reason": "Source 1 directly supports the answer.",
                                    "supporting_source_ids": ["Source 1"],
                                }
                            )
                        )
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=20,
                    total_tokens=120,
                ),
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FakeCompletions()
        )
    )

    prediction = runner.classify_one(
        _input_row(),
        client=client,
        model="qwen3.5-plus-2026-04-20",
        input_rate_cny_per_1m=2.936,
        output_rate_cny_per_1m=17.614,
        clock=lambda: 1.0,
    )

    serialized = json.dumps(captured["messages"])
    assert "TRAIN_Q001" not in serialized
    assert prediction.decision == "SUFFICIENT"
    assert prediction.supporting_source_ids == ("Source 1",)
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 256
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["extra_body"] == {"enable_thinking": False}


def test_evaluate_predictions_passes_preregistered_gate():
    predictions = []
    targets = []

    for index in range(35):
        question_id = f"TRAIN_S{index:03d}"
        decision = "INSUFFICIENT" if index < 3 else "SUFFICIENT"
        predictions.append(_prediction(question_id, decision))
        targets.append(
            {
                "question_id": question_id,
                "target_class": "SUFFICIENT_PROXY",
                "binary_gate_eligible": True,
            }
        )

    for index in range(16):
        question_id = f"TRAIN_I{index:03d}"
        decision = "SUFFICIENT" if index < 3 else "INSUFFICIENT"
        predictions.append(_prediction(question_id, decision))
        targets.append(
            {
                "question_id": question_id,
                "target_class": "INSUFFICIENT_PROXY",
                "binary_gate_eligible": True,
            }
        )

    for index in range(3):
        question_id = f"TRAIN_A{index:03d}"
        predictions.append(_prediction(question_id, "SUFFICIENT"))
        targets.append(
            {
                "question_id": question_id,
                "target_class": "AMBIGUOUS_MULTI_CHUNK",
                "binary_gate_eligible": False,
            }
        )

    summary = runner.evaluate_predictions(
        predictions,
        targets,
        contract=_contract(),
    )

    assert summary.gated_predictions == 51
    assert summary.ambiguous_predictions == 3
    assert summary.sufficient_to_insufficient == 3
    assert summary.insufficient_to_sufficient == 3
    assert summary.gate_balanced_accuracy_pass is True
    assert summary.gate_sufficient_recall_pass is True
    assert summary.gate_insufficient_recall_pass is True
    assert summary.gate_over_refusal_pass is True
    assert summary.decision == "ADMIT_TRAIN_ABSTENTION_PROBE"


def test_evaluate_predictions_rejects_excessive_over_refusal():
    predictions = []
    targets = []

    for index in range(35):
        question_id = f"TRAIN_S{index:03d}"
        decision = "INSUFFICIENT" if index < 6 else "SUFFICIENT"
        predictions.append(_prediction(question_id, decision))
        targets.append(
            {
                "question_id": question_id,
                "target_class": "SUFFICIENT_PROXY",
                "binary_gate_eligible": True,
            }
        )

    for index in range(16):
        question_id = f"TRAIN_I{index:03d}"
        predictions.append(_prediction(question_id, "INSUFFICIENT"))
        targets.append(
            {
                "question_id": question_id,
                "target_class": "INSUFFICIENT_PROXY",
                "binary_gate_eligible": True,
            }
        )

    for index in range(3):
        question_id = f"TRAIN_A{index:03d}"
        predictions.append(_prediction(question_id, "INSUFFICIENT"))
        targets.append(
            {
                "question_id": question_id,
                "target_class": "AMBIGUOUS_MULTI_CHUNK",
                "binary_gate_eligible": False,
            }
        )

    summary = runner.evaluate_predictions(
        predictions,
        targets,
        contract=_contract(),
    )

    assert summary.sufficient_to_insufficient == 6
    assert summary.gate_over_refusal_pass is False
    assert summary.decision == "REJECT_EVIDENCE_SUFFICIENCY_V1"


def test_resolve_classifier_auth_prefers_phase_b_key(monkeypatch):
    monkeypatch.setattr(runner, "load_dotenv", lambda: None)
    monkeypatch.setenv("DASHSCOPE_PHASE_B_API_KEY", "phase-b-key")
    monkeypatch.setenv(
        "DASHSCOPE_PHASE_B_BASE_URL",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    monkeypatch.setenv("DASHSCOPE_RERANK_API_KEY", "rerank-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "generic-key")

    api_key, base_url, key_source = runner.resolve_classifier_auth()

    assert api_key == "phase-b-key"
    assert (
        base_url
        == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    assert key_source == "DASHSCOPE_PHASE_B_API_KEY"


def test_resolve_classifier_auth_falls_back_to_singapore_rerank_key(
    monkeypatch,
):
    monkeypatch.setattr(runner, "load_dotenv", lambda: None)
    monkeypatch.delenv("DASHSCOPE_PHASE_B_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_PHASE_B_BASE_URL", raising=False)
    monkeypatch.setenv("DASHSCOPE_RERANK_API_KEY", "rerank-key")
    monkeypatch.setenv(
        "DASHSCOPE_RERANK_BASE_URL",
        "https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-api/v1",
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "generic-key")

    api_key, base_url, key_source = runner.resolve_classifier_auth()

    assert api_key == "rerank-key"
    assert (
        base_url
        == "https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-api/v1"
    )
    assert key_source == "DASHSCOPE_RERANK_API_KEY"


def test_resolve_classifier_auth_rejects_generic_cross_region_key(
    monkeypatch,
):
    monkeypatch.setattr(runner, "load_dotenv", lambda: None)
    monkeypatch.delenv("DASHSCOPE_PHASE_B_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_PHASE_B_BASE_URL", raising=False)
    monkeypatch.delenv("DASHSCOPE_RERANK_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_RERANK_BASE_URL", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "generic-key")

    with pytest.raises(
        RuntimeError,
        match="must not be reused against a Singapore endpoint",
    ):
        runner.resolve_classifier_auth()
