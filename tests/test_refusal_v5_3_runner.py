import json

import pytest

from experiments.evals import refusal_v5_3_runner as run
from tests.test_refusal_v5_2_1_runner import Client


def given():
    rows = [{"question_id": f"TRAIN_Q{i:03}", "question": "Describe a fact", "sources": [
        {"source_id": "Source 1", "content": "A supported fact.\n"}]} for i in range(2)]
    contract = {"run": "synthetic-v5-3", "classifier": {"model": "qwen3.5-plus-2026-04-20"},
        "max_calls_total": 4, "controls": {"failed_pair_is_terminal": True, "max_failed_attempts_per_arm": 2},
        "arms": {arm: {"max_calls": 2, "max_output_tokens": 2048, "request_input_byte_bounds": [10000, 10000]}
                 for arm in run.prep.BUILDERS}, "pricing_snapshot": {"input_cny_per_1m_tokens": 3,
                    "output_cny_per_1m_tokens": 18, "hard_cost_cap_cny": 8}}
    return contract, rows


def test_resume_saved_raw_no_repeat_no_schema_retry_and_no_targets(tmp_path, monkeypatch):
    contract, rows = given()
    def loader():
        return contract, rows
    original = run.paired.shared._read_jsonl
    def guarded(path):
        assert str(path) != str(run.prep.prior.TARGETS), "paid loop read targets"
        return original(path)
    monkeypatch.setattr(run.paired.shared, "_read_jsonl", guarded)
    root, client = tmp_path / "run", Client()
    run.run_paid(root=root, client=client, max_new_calls=1, probe_loader=loader)
    marker = {"call_index": 1, "arm": "v5_3", "question_id": rows[0]["question_id"]}
    run.paired.write_json(root / "pending_request.json", marker)
    raw = run._reuse._engine.request_record(rows[0], "v5_3", 1, contract, client)
    run.paired.write_json(tmp_path / "run_raw_response.json", raw)
    events = run.run_paid(root=root, client=client, probe_loader=loader)
    run.run_paid(root=root, client=client, probe_loader=loader)
    assert client.calls == 4 and all(e["result"]["decision"] == "SUFFICIENT" for e in events)
    assert len(set(client.prompts)) == 2
    assert set(run.previous.prep.BUILDERS) == {"v5_1", "v5_2_1"}
    assert set(run.previous._engine.BUILDERS) == {"v5_1", "v5_2_1"}
    assert not (tmp_path / "run_raw_response.json").exists()
    bad = Client(bad=True)
    events = run.run_paid(root=tmp_path / "bad", client=bad, probe_loader=loader)
    assert all(e["result"] is None for e in events)
    run.run_paid(root=tmp_path / "bad", client=bad, probe_loader=loader)
    assert bad.calls == 4


def test_unknown_usage_pending_lock_budget_and_caps_stop_without_new_calls(tmp_path):
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
    run.paired.write_json(root / "pending_request.json", {"call_index": 0, "arm": "v5_2_1", "question_id": rows[0]["question_id"]})
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


def test_evaluation_uses_new_arm_names_and_preserves_privacy_nulls_and_denominators(tmp_path, monkeypatch):
    manifest = run.paired.shared._read_json(run.prep.REPORT)
    contract = run.execution_contract(manifest)
    _, short = given()
    rows = [{**short[0], "question_id": qid} for qid in contract["case_order"]]
    labels = [{"question_id": row["question_id"], "target_class": "SUFFICIENT" if i < 21 else "INSUFFICIENT" if i < 25 else "AMBIGUOUS"}
              for i, row in enumerate(rows)]
    targets = tmp_path / "targets.jsonl"
    run.prep.prior.natural.prior._write_jsonl(targets, labels)
    contract["targets_sha256"] = run.paired.shared._sha256(targets)
    def loader():
        return contract, rows
    monkeypatch.setattr(run, "load_probe", loader)
    monkeypatch.setattr(run.prep.prior, "TARGETS", targets)
    monkeypatch.setattr(run, "RESULT", tmp_path / "public.json")
    root, client = tmp_path / "run", Client()
    run.run_paid(root=root, client=client, max_new_calls=1, probe_loader=loader)
    with pytest.raises(RuntimeError, match="all 60 terminal"):
        run.evaluate(root=root)
    run.run_paid(root=root, client=client, probe_loader=loader)
    result = run.evaluate(root=root)
    assert result["provider_calls"] == client.calls == 60
    assert set(result["primary"]["arms"]) == {"v5_2_1", "v5_3"}
    assert result["primary"]["binary_cases"] == result["common_valid_binary"]["cases"] == 25
    assert result["paired_binary_agreement_transitions"] == {"both_agree": 21, "baseline_only": 0, "candidate_only": 0, "neither_agrees": 4}
    assert sum(result["all_outcome_transitions"].values()) == 30
    assert all(value["cases"] == 29 for value in result["family_sensitivity"].values())
    assert "private_transitions" not in result and "target_class" not in json.dumps(result)
    assert len(run.paired.shared._read_json(root / "summary.json")["private_transitions"]) == 30
    assert result["formal_pass_gate"] is None and not result["promotion_allowed"]
    with pytest.raises(RuntimeError, match="overwrite"):
        run.evaluate(root=root)
    metrics = run.population_metrics([
        {"target_class": "SUFFICIENT", "v5_2_1": "SUFFICIENT", "v5_3": "INSUFFICIENT"},
        {"target_class": "INSUFFICIENT", "v5_2_1": "SUFFICIENT", "v5_3": None},
        {"target_class": "AMBIGUOUS", "v5_2_1": "INSUFFICIENT", "v5_3": "SUFFICIENT"}])
    arm = metrics["arms"]["v5_3"]
    assert arm["binary_agreement_possible_bounds"] == [0, .5] and arm["binary_failures"] == 1
    assert arm["admitted_binary_proxy_agreement"] is None and arm["ambiguous_outcomes"]["SUFFICIENT"] == 1
    assert run.population_metrics([])["arms"]["v5_3"]["binary_agreement_including_failures"] is None


def test_tampered_contract_or_protocol_fails_before_private_inputs(tmp_path, monkeypatch):
    manifest = run.paired.shared._read_json(run.prep.REPORT)
    contract = run.execution_contract(manifest)
    assert contract["max_calls_total"] == 60 and contract["pricing_snapshot"]["hard_cost_cap_cny"] == 8
    assert all(a["max_calls"] == 30 and a["max_output_tokens"] == 2048 for a in contract["arms"].values())
    contract["max_calls_total"] = 61
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(run._reuse, "CONTRACT", path)
    monkeypatch.setattr(run.paired.v2, "load_classifier_inputs", lambda *a, **kw: pytest.fail("private inputs read"))
    with pytest.raises(RuntimeError, match="binding changed"):
        run.load_probe()
    protocol = tmp_path / "protocol.md"
    protocol.write_text("changed", encoding="utf-8")
    monkeypatch.setattr(run.prep, "PROTOCOL", protocol)
    with pytest.raises(RuntimeError, match="protocol changed"):
        run.load_probe()


def test_public_authorization_and_source_bindings_are_exact():
    manifest = run.paired.shared._read_json(run.prep.REPORT)
    contract = run.paired.shared._read_json(run.CONTRACT)
    assert contract == run.execution_contract(manifest)
    assert run.canonical(manifest) == run.PREPARATION_SHA256
    assert "explicitly confirmed PR77" in contract["authorization"]
    assert contract["controls"]["sdk_retries"] == 0
    assert not contract["controls"]["promotion_allowed"] and not contract["controls"]["dev_opened"]
    assert contract["checkpoint_root"] == run.ROOT.as_posix()


def test_frozen_actual_result_reports_small_proxy_gain_not_acceptance():
    result = run.paired.shared._read_json(run.RESULT)
    assert result["status"] == "OBSERVED_DEVELOPMENT_REGRESSION_COMPLETE_NO_ACCEPTANCE"
    assert result["provider_calls"] == 60 and result["estimated_cost_cny"] == pytest.approx(1.917294)
    assert result["primary"]["target_counts"] == {"SUFFICIENT": 21, "INSUFFICIENT": 4, "AMBIGUOUS": 5}
    assert result["common_valid_binary"]["cases"] == 25
    assert result["paired_binary_agreement_transitions"] == {"both_agree": 21, "baseline_only": 1, "candidate_only": 2, "neither_agrees": 1}
    for arm, correct, refused in (("v5_2_1", 22, 2), ("v5_3", 23, 1)):
        values = result["primary"]["arms"][arm]
        assert values["valid_correct_binary"] == correct and values["all_case_failures"] == 0
        assert values["confusion"]["sufficient_as_insufficient"] == refused
        assert values["confusion"]["insufficient_as_sufficient"] == 1
        assert values["ambiguous_outcomes"] == {"SUFFICIENT": 3, "INSUFFICIENT": 2, "FAILURE": 0}
        assert sum(values["confusion"].values()) == 25
        assert result["usage"][arm]["calls"] == 30
    for subset in result["family_sensitivity"].values():
        assert subset["cases"] == 29 and subset["binary_cases"] == 24
        assert subset["arms"]["v5_2_1"]["valid_correct_binary"] == 21
        assert subset["arms"]["v5_3"]["valid_correct_binary"] == 22
    assert sum(result["all_outcome_transitions"].values()) == 30
    assert result["formal_pass_gate"] is None
    assert not any(result[k] for k in ("promotion_allowed", "dev_opened", "original_class_support_rule_met", "original_formal_rule_changed"))
    assert "private_transitions" not in result and "target_class" not in json.dumps(result)
    assert run.canonical(run.paired.shared._read_json(run.CONTRACT)) == result["contract_canonical_sha256"]
