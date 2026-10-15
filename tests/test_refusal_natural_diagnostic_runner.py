import json
from types import SimpleNamespace

import pytest

from experiments.evals import refusal_natural_diagnostic_runner as run


def given():
    rows = [{"question_id": f"TRAIN_Q{i:03}", "question": "Describe a fact", "sources": [
        {"source_id": "Source 1", "content": "A supported fact.\n"}]} for i in range(2)]
    contract = {"run": "synthetic-natural", "classifier": {"model": "qwen3.5-plus-2026-04-20"},
        "max_calls_total": 4, "controls": {"failed_pair_is_terminal": True, "max_failed_attempts_per_arm": 2},
        "arms": {arm: {"max_calls": 2, "max_output_tokens": 2048, "request_input_byte_bounds": [10000, 10000]}
                 for arm in run.prep.BUILDERS}, "pricing_snapshot": {"input_cny_per_1m_tokens": 3,
                    "output_cny_per_1m_tokens": 18, "hard_cost_cap_cny": 8}}
    return contract, rows


class Client:
    def __init__(self, bad=False, unknown=False):
        self.calls, self.bad, self.unknown = 0, bad, unknown
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls += 1
        assert kwargs["temperature"] == 0 and kwargs["max_tokens"] == 2048
        assert kwargs["extra_body"] == {"enable_thinking": False}
        payload = json.loads(kwargs["messages"][1]["content"])
        assert set(payload) == {"question", "sources"} and "TRAIN_Q" not in kwargs["messages"][1]["content"]
        support = {"status": "SUPPORTED", "evidence": [{"source_id": "Source 1", "segment_ids": [1]}], "reason": "Visible support"}
        value = {"requirements": [{"requirement": "Describe fact", **support}], "unresolved_ambiguities": []}
        if kwargs["messages"][0]["content"] == run.prep.reuse.candidate.CLASSIFIER_SYSTEM_PROMPT_V5_2_1_DEV:
            value["coverage_checks"] = {"requested_outcomes": support, "applicability": support,
                "version_and_time": {"status": "NOT_REQUIRED", "evidence": [], "reason": "No time claim"}}
        return SimpleNamespace(id="fake", usage=None if self.unknown else SimpleNamespace(
            prompt_tokens=100, completion_tokens=100, total_tokens=200), choices=[SimpleNamespace(
            message=SimpleNamespace(content="{}" if self.bad else json.dumps(value)), finish_reason="stop")])


def test_resume_raw_recovery_schema_failures_and_target_isolation(tmp_path, monkeypatch):
    contract, rows = given()
    def loader():
        return contract, rows
    original_read = run.paired.shared._read_jsonl
    def guarded(path):
        if str(path) == str(run.prep.TARGETS):
            raise AssertionError("paid loop read targets")
        return original_read(path)
    monkeypatch.setattr(run.paired.shared, "_read_jsonl", guarded)
    root, client = tmp_path / "run", Client()
    run.run_paid(root=root, client=client, max_new_calls=1, probe_loader=loader)
    marker = {"call_index": 1, "arm": "v5_2_1", "question_id": rows[0]["question_id"]}
    run.paired.write_json(root / "pending_request.json", marker)
    raw = run._engine.request_record(rows[0], "v5_2_1", 1, contract, client)
    run.paired.write_json(tmp_path / "run_raw_response.json", raw)
    assert len(run.run_paid(root=root, client=client, probe_loader=loader)) == 4
    run.run_paid(root=root, client=client, probe_loader=loader)
    assert client.calls == 4
    assert set(run.prior.BUILDERS) == {"v5_1", "v5_2"}
    bad = Client(bad=True)
    events = run.run_paid(root=tmp_path / "bad", client=bad, probe_loader=loader)
    assert all(e["result"] is None for e in events)
    run.run_paid(root=tmp_path / "bad", client=bad, probe_loader=loader)
    assert bad.calls == 4


def test_unknown_usage_pending_lock_budget_and_call_limits_block(tmp_path):
    contract, rows = given()
    def loader():
        return contract, rows
    unknown = Client(unknown=True)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="unknown usage"):
            run.run_paid(root=tmp_path / "unknown", client=unknown, probe_loader=loader)
    assert unknown.calls == 1
    client, root = Client(), tmp_path / "pending"
    run.run_paid(root=root, client=client, max_new_calls=0, probe_loader=loader)
    run.paired.write_json(root / "pending_request.json", {"call_index": 0, "arm": "v5_1", "question_id": rows[0]["question_id"]})
    with pytest.raises(RuntimeError, match="unknown charge"):
        run.run_paid(root=root, client=client, probe_loader=loader)
    (root / "runner.lock").touch()
    with pytest.raises(FileExistsError):
        run.run_paid(root=root, client=client, probe_loader=loader)
    contract["pricing_snapshot"]["hard_cost_cap_cny"] = 0
    with pytest.raises(RuntimeError, match="spend cap"):
        run.run_paid(root=tmp_path / "budget", client=client, probe_loader=loader)
    for cap in (True, -1, 61):
        with pytest.raises(ValueError):
            run.run_paid(root=root, max_new_calls=cap)
    assert client.calls == 0


def test_ambiguous_and_null_outcomes_never_become_correct_refusals():
    rows = [{"target_class": "SUFFICIENT", "v5_1": "SUFFICIENT", "v5_2_1": "INSUFFICIENT"},
            {"target_class": "INSUFFICIENT", "v5_1": "SUFFICIENT", "v5_2_1": None},
            {"target_class": "AMBIGUOUS", "v5_1": "INSUFFICIENT", "v5_2_1": "SUFFICIENT"},
            {"target_class": "AMBIGUOUS", "v5_1": None, "v5_2_1": None}]
    metrics = run.population_metrics(rows)
    assert metrics["cases"] == 4 and metrics["binary_cases"] == metrics["ambiguous_cases"] == 2
    candidate = metrics["arms"]["v5_2_1"]
    assert candidate["confusion"]["insufficient_as_insufficient"] == 0
    assert candidate["binary_failures"] == 1 and candidate["all_case_failures"] == 2
    assert candidate["binary_agreement_possible_bounds"] == [0, .5]
    assert candidate["admission_coverage"] == .25
    assert candidate["admitted_binary_proxy_agreement"] is None
    assert candidate["ambiguous_outcomes"] == {"SUFFICIENT": 1, "INSUFFICIENT": 0, "FAILURE": 1}
    empty = run.population_metrics([])["arms"]["v5_1"]
    assert empty["binary_agreement_including_failures"] is None and empty["insufficient_proxy_recall"] is None


def test_evaluator_waits_for_all_outcomes_and_keeps_details_private(tmp_path, monkeypatch):
    manifest = run.paired.shared._read_json(run.prep.REPORT)
    contract = run.execution_contract(manifest)
    _, short = given()
    rows = [{**short[0], "question_id": qid} for qid in contract["case_order"]]
    labels = [{"question_id": row["question_id"], "target_class": "SUFFICIENT" if i < 21 else "INSUFFICIENT" if i < 25 else "AMBIGUOUS"}
              for i, row in enumerate(rows)]
    targets = tmp_path / "targets.jsonl"
    run.prep.natural.prior._write_jsonl(targets, labels)
    contract["targets_sha256"] = run.paired.shared._sha256(targets)
    def loader():
        return contract, rows
    monkeypatch.setattr(run, "load_probe", loader)
    monkeypatch.setattr(run.prep, "TARGETS", targets)
    monkeypatch.setattr(run, "RESULT", tmp_path / "public.json")
    root, client = tmp_path / "run", Client()
    run.run_paid(root=root, client=client, max_new_calls=1, probe_loader=loader)
    with pytest.raises(RuntimeError, match="all 60 terminal"):
        run.evaluate(root=root)
    run.run_paid(root=root, client=client, probe_loader=loader)
    result = run.evaluate(root=root)
    assert result["provider_calls"] == client.calls == 60
    assert result["primary"]["binary_cases"] == 25 and result["common_valid_binary"]["cases"] == 25
    assert sum(result["all_outcome_transitions"].values()) == 30
    assert all(value["cases"] == 29 for value in result["family_sensitivity"].values())
    assert "private_transitions" not in result and "target_class" not in json.dumps(result)
    assert len(run.paired.shared._read_json(root / "summary.json")["private_transitions"]) == 30
    assert result["formal_pass_gate"] is None and not result["promotion_allowed"]
    with pytest.raises(RuntimeError, match="overwrite"):
        run.evaluate(root=root)


def test_public_execution_contract_is_pinned_and_tampering_fails_before_inputs(tmp_path, monkeypatch):
    manifest = run.paired.shared._read_json(run.prep.REPORT)
    contract = run.paired.shared._read_json(run.CONTRACT)
    assert contract == run.execution_contract(manifest)
    assert contract["max_calls_total"] == 60 and contract["pricing_snapshot"]["hard_cost_cap_cny"] == 8
    assert all(a["max_calls"] == 30 and a["max_output_tokens"] == 2048 for a in contract["arms"].values())
    contract["max_calls_total"] = 61
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(run, "CONTRACT", path)
    monkeypatch.setattr(run.prep, "INPUTS", tmp_path / "absent.jsonl")
    with pytest.raises(RuntimeError, match="binding changed"):
        run.load_probe()


def test_frozen_actual_result_is_proxy_diagnosis_not_acceptance():
    result = run.paired.shared._read_json(run.RESULT)
    assert result["status"] == "EXPLORATORY_DIAGNOSTIC_COMPLETE_NO_ACCEPTANCE"
    assert result["provider_calls"] == 60 and result["estimated_cost_cny"] == pytest.approx(1.663062)
    primary = result["primary"]
    assert primary["cases"] == 30 and primary["binary_cases"] == 25 and primary["ambiguous_cases"] == 5
    assert result["common_valid_binary"]["cases"] == 25
    assert result["paired_binary_agreement_transitions"] == {"both_agree": 17, "baseline_only": 2, "candidate_only": 4, "neither_agrees": 2}
    for arm, correct, refused in (("v5_1", 19, 5), ("v5_2_1", 21, 3)):
        values = primary["arms"][arm]
        assert values["all_case_failures"] == values["binary_failures"] == 0
        assert values["valid_correct_binary"] == correct
        assert values["confusion"]["sufficient_as_insufficient"] == refused
        assert values["confusion"]["insufficient_as_sufficient"] == 1
        assert sum(values["confusion"].values()) == 25
        assert sum(values["ambiguous_outcomes"].values()) == 5
    assert sum(result["all_outcome_transitions"].values()) == 30
    for subset in result["family_sensitivity"].values():
        assert subset["cases"] == 29 and subset["binary_cases"] == 24
        assert subset["arms"]["v5_2_1"]["valid_correct_binary"] - subset["arms"]["v5_1"]["valid_correct_binary"] == 1
    assert result["formal_pass_gate"] is None
    assert not result["original_class_support_rule_met"] and not result["original_formal_rule_changed"]
    assert not result["dev_opened"] and not result["promotion_allowed"]
    assert "private_transitions" not in result and "target_class" not in json.dumps(result)
    contract = run.paired.shared._read_json(run.CONTRACT)
    assert run.canonical(contract) == result["contract_canonical_sha256"]
