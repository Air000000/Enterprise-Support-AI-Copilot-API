import json
from types import SimpleNamespace

import pytest

from experiments.evals import refusal_v5_2_1_runner as run


def given():
    rows = [{"question_id": f"CONTROL_{i}", "question": "Describe a fact", "sources": [
        {"source_id": "Source 1", "content": "A supported fact.\n"}]} for i in range(2)]
    contract = {"run": "synthetic-regression", "classifier": {"model": "qwen3.5-plus-2026-04-20"},
        "max_calls_total": 4, "controls": {"failed_pair_is_terminal": True, "max_failed_attempts_per_arm": 2},
        "arms": {arm: {"max_calls": 2, "max_output_tokens": 2048, "request_input_byte_bounds": [10000, 10000]}
                 for arm in run.prep.BUILDERS}, "pricing_snapshot": run.prep.PRICING}
    return contract, rows


class Client:
    def __init__(self, bad=False, unknown=False):
        self.calls, self.bad, self.unknown = 0, bad, unknown
        self.prompts = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls += 1
        self.prompts.append(kwargs["messages"][0]["content"])
        assert kwargs["max_tokens"] == 2048 and kwargs["temperature"] == 0
        assert kwargs["extra_body"] == {"enable_thinking": False}
        assert "CONTROL_" not in kwargs["messages"][1]["content"]
        support = {"status": "SUPPORTED", "evidence": [{"source_id": "Source 1", "segment_ids": [1]}], "reason": "Support"}
        value = {"requirements": [{"requirement": "Describe fact", **support}], "unresolved_ambiguities": [],
            "coverage_checks": {"requested_outcomes": support, "applicability": support,
                "version_and_time": {"status": "NOT_REQUIRED", "evidence": [], "reason": "No time claim"}}}
        return SimpleNamespace(id="fake", usage=None if self.unknown else SimpleNamespace(
            prompt_tokens=100, completion_tokens=100, total_tokens=200), choices=[SimpleNamespace(
            message=SimpleNamespace(content="{}" if self.bad else json.dumps(value)), finish_reason="stop")])


def test_isolated_engine_resume_raw_recovery_and_no_schema_retries(tmp_path):
    contract, rows = given()
    def loader():
        return contract, rows
    root, client = tmp_path / "run", Client()
    assert set(run.prior.BUILDERS) == {"v5_1", "v5_2"}  # Shared old module was not monkeypatched.
    run.run_paid(root=root, client=client, max_new_calls=1, probe_loader=loader)
    marker = {"call_index": 1, "arm": "v5_2_1", "question_id": rows[0]["question_id"]}
    run.paired.write_json(root / "pending_request.json", marker)
    raw = run._engine.request_record(rows[0], "v5_2_1", 1, contract, client)
    run.paired.write_json(tmp_path / "run_raw_response.json", raw)
    assert len(run.run_paid(root=root, client=client, probe_loader=loader)) == 4
    run.run_paid(root=root, client=client, probe_loader=loader)
    assert client.calls == 4 and len(set(client.prompts)) == 2
    assert not (tmp_path / "run_raw_response.json").exists()
    bad = Client(bad=True)
    events = run.run_paid(root=tmp_path / "bad", client=bad, probe_loader=loader)
    assert len(events) == 4 and all(e["result"] is None for e in events)
    run.run_paid(root=tmp_path / "bad", client=bad, probe_loader=loader)
    assert bad.calls == 4


def test_unknown_usage_pending_lock_and_budget_block_new_calls(tmp_path):
    contract, rows = given()
    def loader():
        return contract, rows
    unknown = Client(unknown=True)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="unknown usage"):
            run.run_paid(root=tmp_path / "unknown", client=unknown, probe_loader=loader)
    assert unknown.calls == 1
    good, root = Client(), tmp_path / "pending"
    run.run_paid(root=root, client=good, max_new_calls=0, probe_loader=loader)
    run.paired.write_json(root / "pending_request.json", {"call_index": 0, "arm": "v5_2", "question_id": rows[0]["question_id"]})
    with pytest.raises(RuntimeError, match="unknown charge"):
        run.run_paid(root=root, client=good, probe_loader=loader)
    (root / "runner.lock").touch()
    with pytest.raises(FileExistsError):
        run.run_paid(root=root, client=good, probe_loader=loader)
    contract["pricing_snapshot"] = {**contract["pricing_snapshot"], "hard_cost_cap_cny": 0}
    with pytest.raises(RuntimeError, match="spend cap"):
        run.run_paid(root=tmp_path / "budget", client=good, probe_loader=loader)
    assert good.calls == 0


def test_paid_loop_does_not_read_targets_and_invalid_new_call_cap_fails(tmp_path, monkeypatch):
    contract, rows = given()
    def forbidden():
        raise AssertionError("targets opened during paid loop")
    monkeypatch.setattr(run.prior.controls, "load_cases", forbidden)
    assert len(run.run_paid(root=tmp_path / "run", client=Client(), probe_loader=lambda: (contract, rows))) == 4
    for cap in (-1, True, 29):
        with pytest.raises(ValueError):
            run.run_paid(root=tmp_path / "invalid", max_new_calls=cap)


def test_real_probe_verifies_full_contract_before_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "INPUTS", tmp_path / "inputs.jsonl")
    monkeypatch.setattr(run, "CONTRACT", tmp_path / "contract.json")
    contract = run.prepare_contract()
    def forbidden():
        raise AssertionError("probe read targets")
    monkeypatch.setattr(run.prior.controls, "load_cases", forbidden)
    actual, rows = run.load_probe()
    assert actual == contract and len(rows) == 14
    assert actual["preparation_canonical_sha256"] == run.PREPARATION_SHA256
    contract["max_calls_total"] = 29
    run.CONTRACT.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(RuntimeError, match="binding changed"):
        run.load_probe()


@pytest.mark.parametrize("bad_arm,new_error", [("v5_2", False), ("v5_2_1", False), ("v5_2", True)])
def test_evaluation_keeps_null_failures_and_never_promotes(tmp_path, monkeypatch, bad_arm, new_error):
    manifest = run.paired.shared._read_json(run.prep.REPORT)
    contract = run.execution_contract(manifest, "synthetic-only")
    rows = run.prior.controls.classifier_rows(run.prior.controls.load_cases())
    def loader():
        return contract, rows
    monkeypatch.setattr(run, "load_probe", loader)
    monkeypatch.setattr(run, "RESULT", tmp_path / "result.json")
    bad_prompt = run.prep.BUILDERS[bad_arm](run.paired.shared._classifier_input_from_row(rows[0]))[0]["content"]
    candidate_prompt = run.prep.candidate.CLASSIFIER_SYSTEM_PROMPT_V5_2_1_DEV
    class OneFailure(Client):
        failed = False
        semantic_failed = False
        def create(self, **kwargs):
            value = super().create(**kwargs)
            if kwargs["messages"][0]["content"] == bad_prompt and not self.failed:
                value.choices[0].message.content = "{}"
                self.failed = True
            if new_error and kwargs["messages"][0]["content"] == candidate_prompt and not self.semantic_failed:
                payload = json.loads(value.choices[0].message.content)
                payload["requirements"][0].update(status="MISSING", evidence=[], reason="Synthetic false refusal")
                value.choices[0].message.content = json.dumps(payload)
                self.semantic_failed = True
            return value
    root = tmp_path / "run"
    client = OneFailure()
    run.run_paid(root=root, client=client, max_new_calls=1, probe_loader=loader)
    with pytest.raises(RuntimeError, match="all 28 terminal"):
        run.evaluate(root=root)
    run.run_paid(root=root, client=client, probe_loader=loader)
    result = run.evaluate(root=root)
    assert result["provider_calls"] == client.calls == 28
    assert result["transitions"][0][bad_arm] is None
    assert result["common_valid"]["cases"] == 13
    for arm in result["arms"].values():
        assert sum(arm["confusion"].values()) + arm["failures"] == 14
    assert result["arms"][bad_arm]["failures"] == 1
    assert not result["promotion_allowed"] and not result["dev_opened"]
    assert not result["natural_train_pooled"]
    assert bool(result["new_valid_semantic_errors"]) == new_error
    expected = "FORMAT_FIX_DEVELOPMENT_SIGNAL" if bad_arm == "v5_2" and not new_error else "NO_DEMONSTRATED_FORMAT_FIX_GAIN"
    assert result["status"] == expected


def test_frozen_actual_regression_is_equal_not_an_improvement_or_promotion():
    value = run.paired.shared._read_json(run.RESULT)
    assert value["status"] == "NO_DEMONSTRATED_FORMAT_FIX_GAIN"
    assert value["cases"] == 14 and value["provider_calls"] == 28
    assert value["common_valid"] == {"cases": 14, "v5_2": 14, "v5_2_1": 14}
    assert value["new_valid_semantic_errors"] == []
    for arm in value["arms"].values():
        assert arm["valid_correct"] == 14 and arm["failures"] == 0
        assert sum(arm["confusion"].values()) == arm["case_denominator"] == 14
        assert arm["confusion"]["sufficient_as_sufficient"] == 7
        assert arm["confusion"]["insufficient_as_insufficient"] == 7
    assert all(t["v5_2"] == t["v5_2_1"] == t["expected"] for t in value["transitions"])
    assert value["estimated_cost_cny"] == pytest.approx(sum(a["estimated_cost_cny"] for a in value["arms"].values()))
    assert value["estimated_cost_cny"] < 3
    assert not value["promotion_allowed"] and not value["natural_train_pooled"] and not value["dev_opened"]
    manifest = run.paired.shared._read_json(run.prep.REPORT)
    contract = run.paired.shared._read_json(run.CONTRACT)
    assert contract == run.execution_contract(manifest, contract["inputs_sha256"])
    assert run.canonical(contract) == value["contract_canonical_sha256"]
