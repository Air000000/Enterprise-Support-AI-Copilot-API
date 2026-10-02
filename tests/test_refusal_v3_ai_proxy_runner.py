import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.evals import refusal_v3_ai_proxy_runner as runner


def _contract():
    return runner.shared._read_json(runner.DEFAULT_CONTRACT_PATH)


def _inputs(tmp_path, monkeypatch):
    path = tmp_path / "inputs.jsonl"
    rows = [
        {
            "question_id": f"TRAIN_Q{index:03d}",
            "question": f"Question {index}",
            "sources": [
                {"source_id": f"Source {rank}", "content": f"Evidence {rank}"}
                for rank in range(1, 15)
            ],
        }
        for index in range(50)
    ]
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")
    monkeypatch.setattr(runner, "EXPECTED_INPUTS_SHA256", runner.shared._sha256(path))
    return path, rows


def test_v3_contract_rejects_tuning():
    contract = _contract()
    runner.validate_contract(contract)
    contract["gate"]["sufficient_recall_min"] = 0.8
    with pytest.raises(RuntimeError, match="contract changed"):
        runner.validate_contract(contract)


def test_paid_resume_is_blind_and_counts_failure(tmp_path, monkeypatch):
    inputs_path, rows = _inputs(tmp_path, monkeypatch)
    targets_path = tmp_path / "targets.jsonl"
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            assert "TRAIN_" not in json.dumps(kwargs["messages"])
            assert kwargs["max_tokens"] == 512
            return SimpleNamespace(
                id=f"request-{len(calls)}",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                "truncated"
                                if len(calls) == 3
                                else json.dumps(
                                    {
                                        "decision": "INSUFFICIENT",
                                        "reason": "Evidence is incomplete.",
                                        "supporting_source_ids": [],
                                    }
                                )
                            )
                        )
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=100, completion_tokens=10, total_tokens=110
                ),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    read_jsonl = runner.shared._read_jsonl

    def blind_read(path):
        assert Path(path) != targets_path, "paid loop opened labels"
        return read_jsonl(path)

    monkeypatch.setattr(runner.shared, "_read_jsonl", blind_read)
    with pytest.raises(RuntimeError, match="invalid JSON"):
        runner.run_paid(inputs_path=inputs_path, output_dir=tmp_path, client=client)
    assert len(runner.shared.load_predictions(tmp_path / "predictions.jsonl")) == 2
    with pytest.raises(RuntimeError, match="all 50 predictions"):
        runner.evaluate_checkpoint(
            inputs_path=inputs_path, targets_path=targets_path, output_dir=tmp_path
        )
    runner.run_paid(inputs_path=inputs_path, output_dir=tmp_path, client=client)
    assert len(calls) == 51
    runner.run_paid(inputs_path=inputs_path, output_dir=tmp_path, client=client)
    assert len(calls) == 51
    assert calls[0]["messages"] != calls[1]["messages"]
    assert calls[2]["messages"] == calls[3]["messages"]
    assert all(call["messages"] != calls[0]["messages"] for call in calls[2:])

    monkeypatch.setattr(runner.shared, "_read_jsonl", read_jsonl)
    with targets_path.open("w", encoding="utf-8", newline="\n") as file:
        for index, row in enumerate(rows):
            file.write(
                json.dumps(
                    {
                        "question_id": row["question_id"],
                        "target_class": "SUFFICIENT" if index < 25 else "INSUFFICIENT",
                    }
                )
                + "\n"
            )
    monkeypatch.setattr(
        runner, "EXPECTED_TARGETS_SHA256", runner.shared._sha256(targets_path)
    )
    summary = runner.evaluate_checkpoint(
        inputs_path=inputs_path, targets_path=targets_path, output_dir=tmp_path
    )
    assert summary.total_predictions == 50
    assert summary.provider_calls == 51
    assert summary.prompt_tokens == 5100
    assert summary.total_tokens == 5610
    assert summary.estimated_cost_cny == pytest.approx(
        51 * (100 * 2.936 + 10 * 17.614) / 1_000_000
    )
    assert summary.balanced_accuracy == summary.accuracy == 0.5
    assert summary.decision == "REJECT_EVIDENCE_SUFFICIENCY_V3_AI_PROXY"


def test_budget_and_unknown_usage_stop_before_provider(tmp_path, monkeypatch):
    inputs_path, rows = _inputs(tmp_path, monkeypatch)
    contract = _contract()
    cost = (170_000 * 2.936 + 10 * 17.614) / 1_000_000
    prediction = runner.shared.PhaseBPrediction(
        question_id=rows[0]["question_id"],
        decision="INSUFFICIENT",
        reason="Missing support",
        supporting_source_ids=(),
        model=contract["classifier"]["model"],
        request_id="req",
        prompt_tokens=170_000,
        completion_tokens=10,
        total_tokens=170_010,
        latency_ms=1,
        estimated_cost_cny=cost,
    )
    for row in rows[:3]:
        runner.shared._append_jsonl(
            tmp_path / "predictions.jsonl",
            asdict(runner.replace(prediction, question_id=row["question_id"])),
        )
    monkeypatch.setattr(
        runner.shared,
        "get_classifier_client",
        lambda: pytest.fail("provider must not be contacted"),
    )
    with pytest.raises(RuntimeError, match="insufficient accounting budget"):
        runner.run_paid(inputs_path=inputs_path, output_dir=tmp_path)
    runner.shared._append_jsonl(
        tmp_path / "failed_attempts.jsonl",
        {
            "question_id": rows[1]["question_id"],
            "model": contract["classifier"]["model"],
            "stage": "provider",
            "estimated_cost_cny": 0,
            "usage_available": False,
        },
    )
    with pytest.raises(RuntimeError, match="unknown provider usage"):
        runner.run_paid(inputs_path=inputs_path, output_dir=tmp_path)


def test_v3_gate_retains_false_refusal_ceiling():
    predictions, targets = [], []
    for index in range(50):
        target = "SUFFICIENT" if index < 25 else "INSUFFICIENT"
        decision = "INSUFFICIENT" if index < 3 else target
        predictions.append(
            runner.shared.PhaseBPrediction(
                question_id=f"TRAIN_Q{index:03d}",
                decision=decision,
                reason="test",
                supporting_source_ids=("Source 1",) if decision == "SUFFICIENT" else (),
                model="qwen3.5-plus-2026-04-20",
                request_id="req",
                prompt_tokens=100,
                completion_tokens=10,
                total_tokens=110,
                latency_ms=1,
                estimated_cost_cny=0.001,
            )
        )
        targets.append({"question_id": f"TRAIN_Q{index:03d}", "target_class": target})
    kwargs = {
        "contract": _contract(),
        "contract_validator": runner.validate_contract,
        "expected_cases": 50,
    }
    passed = runner.v2.evaluate_predictions(predictions, targets, **kwargs)
    assert passed.sufficient_recall == 0.88
    assert passed.decision == "ADMIT_TO_INDEPENDENT_HUMAN_CONFIRMATION"
    predictions[3] = runner.replace(
        predictions[3], decision="INSUFFICIENT", supporting_source_ids=()
    )
    rejected = runner.v2.evaluate_predictions(predictions, targets, **kwargs)
    assert rejected.sufficient_recall == 0.84
    assert rejected.decision == "REJECT_EVIDENCE_SUFFICIENCY_V3_AI_PROXY"


def test_v3_diagnostic_audit_keeps_frozen_result_and_targets():
    report_dir = runner.DEFAULT_CONTRACT_PATH.parent
    audit = runner.shared._read_json(report_dir / "v3_disagreement_audit.json")
    result = runner.shared._read_json(report_dir / "v3_ai_proxy_result_freeze.json")
    annotation = runner.shared._read_json(
        report_dir / "v3_confirmation_annotation_freeze.json"
    )
    targets = {
        row["question_id"]: row["target_class"]
        for row in runner.shared._read_jsonl(
            report_dir / "v3_confirmation_targets.jsonl"
        )
    }
    cases = audit["cases"]
    assert audit["status"] == "POST_HOC_DIAGNOSTIC_NOT_RESCORING"
    assert audit["unchanged_decision"] == result["evaluation"]["decision"]
    assert (
        audit["frozen_artifacts"]["annotated_packet_sha256"]
        == annotation["canonical_annotated_packet_sha256"]
    )
    for key in ("inputs_sha256", "targets_sha256", "predictions_sha256"):
        assert audit["frozen_artifacts"][key] == result["artifacts"][key]
    assert len(cases) == len({case["question_id"] for case in cases}) == 12
    assert (
        Counter(case["primary_attribution"] for case in cases)
        == audit["attribution_counts"]
    )
    for case in cases:
        assert case["frozen_target"] == targets[case["question_id"]]
        assert {case["frozen_target"], case["prediction"]} == {
            "SUFFICIENT",
            "INSUFFICIENT",
        }
        assert case["finding"] and case["confidence"] in {"high", "medium"}
        assert case["evidence"]
        assert all(
            evidence["anchor"]
            and evidence["source_id"] in {f"Source {rank}" for rank in range(1, 15)}
            for evidence in case["evidence"]
        )
    counts = Counter(case["frozen_target"] for case in cases)
    assert (
        counts["SUFFICIENT"] == result["evaluation"]["sufficient_false_negative"] == 6
    )
    assert (
        counts["INSUFFICIENT"]
        == result["evaluation"]["insufficient_false_positive"]
        == 6
    )
    assert (
        audit["controls"]["provider_calls"]
        == audit["controls"]["estimated_cost_cny"]
        == 0
    )
    assert all(
        value is False
        for key, value in audit["controls"].items()
        if key not in {"provider_calls", "estimated_cost_cny"}
    )
