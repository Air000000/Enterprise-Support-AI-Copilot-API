import json
from types import SimpleNamespace

import pytest

from experiments.evals import refusal_v5_2_control_runner as run


def given():
    rows = [{"question_id": f"CONTROL_{i}", "question": "Question", "sources": [
        {"source_id": "Source 1", "content": "A supported fact.\n"}]} for i in range(2)]
    contract = {"run": "synthetic-runner-test", "classifier": {"model": "qwen3.5-plus-2026-04-20"},
                "max_calls_total": 4, "controls": {"failed_pair_is_terminal": True, "max_failed_attempts_per_arm": 2},
                "arms": {arm: {"max_calls": 2, "max_output_tokens": 2048,
                              "request_input_byte_bounds": [10000, 10000]} for arm in run.BUILDERS},
                "pricing_snapshot": {"input_cny_per_1m_tokens": 3, "output_cny_per_1m_tokens": 18,
                                     "hard_cost_cap_cny": 3}}
    return contract, rows


class Client:
    def __init__(self, bad=False, unknown=False):
        self.calls = 0
        self.bad, self.unknown = bad, unknown
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls += 1
        assert kwargs["temperature"] == 0 and kwargs["max_tokens"] == 2048
        assert kwargs["extra_body"] == {"enable_thinking": False}
        assert "CONTROL_" not in kwargs["messages"][1]["content"]
        support = {"status": "SUPPORTED", "evidence": [{"source_id": "Source 1", "segment_ids": [1]}], "reason": "Visible support"}
        value = {"requirements": [{"requirement": "Answer", **support}], "unresolved_ambiguities": []}
        if kwargs["messages"][0]["content"] == run.coverage.CLASSIFIER_SYSTEM_PROMPT_V5_2_DEV:
            value["coverage_checks"] = {"requested_outcomes": support, "applicability": support,
                                       "version_and_time": {"status": "NOT_REQUIRED", "evidence": [], "reason": "No date"}}
        return SimpleNamespace(id="request", usage=None if self.unknown else SimpleNamespace(
            prompt_tokens=100, completion_tokens=100, total_tokens=200), choices=[SimpleNamespace(
            message=SimpleNamespace(content="{}" if self.bad else json.dumps(value)), finish_reason="stop")])


def test_resume_does_not_repeat_successes_and_schema_failures_are_terminal(tmp_path):
    contract, rows = given()
    client = Client()
    def loader():
        return contract, rows
    assert len(run.run_paid(root=tmp_path / "run", client=client, max_new_calls=1, probe_loader=loader)) == 1
    assert len(run.run_paid(root=tmp_path / "run", client=client, probe_loader=loader)) == 4
    run.run_paid(root=tmp_path / "run", client=client, probe_loader=loader)
    assert client.calls == 4
    bad = Client(bad=True)
    events = run.run_paid(root=tmp_path / "bad", client=bad, probe_loader=loader)
    assert len(events) == 4 and all(e["result"] is None and e["error_stage"] == "schema" for e in events)
    run.run_paid(root=tmp_path / "bad", client=bad, probe_loader=loader)
    assert bad.calls == 4  # No schema retries; failures are not refusal predictions.


def test_unknown_usage_or_unanswered_request_blocks_resume(tmp_path):
    contract, rows = given()
    def loader():
        return contract, rows
    client = Client(unknown=True)
    with pytest.raises(RuntimeError, match="unknown usage"):
        run.run_paid(root=tmp_path / "run", client=client, probe_loader=loader)
    with pytest.raises(RuntimeError, match="unknown usage"):
        run.run_paid(root=tmp_path / "run", client=client, probe_loader=loader)
    assert client.calls == 1
    root = tmp_path / "pending"
    good = Client()
    run.run_paid(root=root, client=good, max_new_calls=0, probe_loader=loader)
    run.paired.write_json(root / "pending_request.json", {"call_index": 0, "arm": "v5_1", "question_id": rows[0]["question_id"]})
    with pytest.raises(RuntimeError, match="unknown charge"):
        run.run_paid(root=root, client=good, probe_loader=loader)
    assert good.calls == 0


def test_saved_raw_response_replays_without_provider_call(tmp_path):
    contract, rows = given()
    root = tmp_path / "run"
    client = Client()
    def loader():
        return contract, rows
    run.run_paid(root=root, client=client, max_new_calls=0, probe_loader=loader)
    marker = {"call_index": 0, "arm": "v5_1", "question_id": rows[0]["question_id"]}
    run.paired.write_json(root / "pending_request.json", marker)
    raw = run.request_record(rows[0], "v5_1", 0, contract, client)
    run.paired.write_json(tmp_path / "run_raw_response.json", raw)
    assert len(run.run_paid(root=root, client=client, probe_loader=loader)) == 4
    assert client.calls == 4  # One original request plus three remaining, not five.
    assert not (tmp_path / "run_raw_response.json").exists()


def test_spend_reservation_blocks_before_request(tmp_path):
    contract, rows = given()
    contract["pricing_snapshot"]["hard_cost_cap_cny"] = 0
    client = Client()
    with pytest.raises(RuntimeError, match="spend cap"):
        run.run_paid(root=tmp_path / "run", client=client, probe_loader=lambda: (contract, rows))
    assert client.calls == 0


def test_paid_loop_never_opens_case_targets(tmp_path, monkeypatch):
    contract, rows = given()
    def forbidden():
        raise AssertionError("case targets opened during paid loop")
    monkeypatch.setattr(run.controls, "load_cases", forbidden)
    assert len(run.run_paid(root=tmp_path / "run", client=Client(), probe_loader=lambda: (contract, rows))) == 4


def test_frozen_control_results_keep_failures_in_denominators_and_no_promotion():
    value = run.paired.shared._read_json(run.REPORT / "v5_2_control_result.json")
    assert value["cases"] == 14 and value["provider_calls"] == 28
    for arm in value["arms"].values():
        assert sum(arm["confusion"].values()) + arm["failures"] == arm["case_denominator"] == 14
    assert value["arms"]["v5_1"]["valid_correct"] == 14
    assert value["arms"]["v5_2"]["valid_correct"] == 12
    assert value["arms"]["v5_2"]["failures"] == 2
    assert value["estimated_cost_cny"] == pytest.approx(sum(a["estimated_cost_cny"] for a in value["arms"].values()))
    assert len(value["families"]) == 7
    assert not value["natural_train_pooled"] and not value["dev_opened"] and not value["promotion_allowed"]
