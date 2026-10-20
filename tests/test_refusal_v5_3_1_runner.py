import hashlib
import json

import httpx
import pytest

from experiments.evals import refusal_v5_3_1_runner as run
from tests.test_refusal_v5_2_annotation import Client as AnnotationClient, setup_run
from tests.test_refusal_v5_2_1_runner import Client as ClassifierClient


def classifier_fixture():
    _, rows, _ = setup_run()
    contract = {"run": "synthetic-prospective", "classifier": {"model": "qwen3.5-plus-2026-04-20"},
        "max_calls_total": 6, "controls": {"failed_pair_is_terminal": True, "max_failed_attempts_per_arm": 3},
        "arms": {a: {"max_calls": 3, "max_output_tokens": 2048, "request_input_byte_bounds": [10000] * 3} for a in run.prep.BUILDERS},
        "pricing_snapshot": {"input_cny_per_1m_tokens": 3, "output_cny_per_1m_tokens": 18, "hard_cost_cap_cny": 15}}
    return contract, rows


def test_annotation_resume_and_unknown_charge_never_repeat_completed_requests(tmp_path):
    contract, rows, _ = setup_run()
    client = AnnotationClient()
    options = {"probe_loader": lambda: (contract, rows), "root": tmp_path / "good", "client": client}
    assert run.run_annotation(**options, max_new_calls=1)["completed"] == 1
    assert run.run_annotation(**options)["completed"] == 3
    assert run.run_annotation(**options)["calls"] == client.calls == 3
    bad = AnnotationClient(fail=True)
    options.update(root=tmp_path / "unknown", client=bad)
    with pytest.raises(RuntimeError, match="failure saved"):
        run.run_annotation(**options)
    with pytest.raises(RuntimeError, match="unknown charge"):
        run.run_annotation(**options)
    assert bad.calls == 1
    assert "SECRET" not in (tmp_path / "unknown" / "attempts.jsonl").read_text()
    for cap in (-1, True, 51):
        with pytest.raises(ValueError):
            run.run_annotation(max_new_calls=cap)
    with pytest.raises(RuntimeError, match="bounded client"):
        run.run_annotation()


def test_classifier_resume_raw_recovery_and_terminal_schema_failures_without_targets(tmp_path, monkeypatch):
    contract, rows = classifier_fixture()
    original = run.shared._read_jsonl
    def guarded(path):
        assert "targets" not in str(path)
        return original(path)
    monkeypatch.setattr(run.shared, "_read_jsonl", guarded)
    client, root = ClassifierClient(), tmp_path / "paired"
    options = {"root": root, "client": client, "probe_loader": lambda: (contract, rows)}
    run.run_classification(**options, max_new_calls=1)
    raw = run.engine.request_record(rows[0], "v5_3_1", 1, contract, client)
    run.paired.write_json(root / "pending_request.json", {k: raw[k] for k in ("call_index", "arm", "question_id")})
    run.paired.write_json(tmp_path / "paired_raw_response.json", raw)
    assert len(run.run_classification(**options)) == 6
    run.run_classification(**options)
    assert client.calls == 6
    bad = ClassifierClient(bad=True)
    options.update(root=tmp_path / "bad", client=bad)
    events = run.run_classification(**options)
    assert len(events) == 6 and all(e["result"] is None for e in events)
    run.run_classification(**options)
    assert bad.calls == 6
    assert set(run.control.BUILDERS) == {"v5_1", "v5_2"}  # Old modules stay isolated.


def test_classifier_unknown_usage_locks_caps_and_pending_block_new_calls(tmp_path):
    contract, rows = classifier_fixture()
    def loader():
        return contract, rows
    unknown = ClassifierClient(unknown=True)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="unknown usage"):
            run.run_classification(root=tmp_path / "unknown", client=unknown, probe_loader=loader)
    assert unknown.calls == 1
    good, root = ClassifierClient(), tmp_path / "pending"
    run.run_classification(root=root, client=good, max_new_calls=0, probe_loader=loader)
    run.paired.write_json(root / "pending_request.json", {"call_index": 0, "arm": "v5_3", "question_id": rows[0]["question_id"]})
    with pytest.raises(RuntimeError, match="unknown charge"):
        run.run_classification(root=root, client=good, probe_loader=loader)
    (root / "runner.lock").touch()
    with pytest.raises(FileExistsError):
        run.run_classification(root=root, client=good, probe_loader=loader)
    contract["pricing_snapshot"]["hard_cost_cap_cny"] = 0
    with pytest.raises(RuntimeError, match="spend cap"):
        run.run_classification(root=tmp_path / "budget", client=good, probe_loader=loader)
    for cap in (-1, True, 101):
        with pytest.raises(ValueError):
            run.run_classification(max_new_calls=cap)
    assert good.calls == 0


def test_authorization_and_label_freeze_are_required_before_contracts(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "load_rows", lambda: pytest.fail("inputs loaded without consent"))
    for value in (False, None, 1, "continue"):
        with pytest.raises(RuntimeError, match="explicit two-stage"):
            run.prepare_contract("annotation", authorization_confirmed=value)
    plan = run.read_plan()
    _, rows, _ = setup_run()
    monkeypatch.setattr(run, "FREEZE", tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError):
        run.execution_contract("classification", plan, rows, "endpoint")
    run.prep.prior.selection.prior._write_json(run.FREEZE, {"cases": 49, "inputs_sha256": plan["inputs_sha256"], "classifier_predictions_created": False})
    with pytest.raises(RuntimeError, match="complete preprediction"):
        run.execution_contract("classification", plan, rows, "endpoint")


def test_bounded_client_pins_region_endpoint_and_complete_body_before_send(tmp_path, monkeypatch):
    url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    env = tmp_path / "fake.env"
    env.write_text(f"DASHSCOPE_PHASE_B_API_KEY=offline\nDASHSCOPE_PHASE_B_BASE_URL={url}\n", encoding="utf-8")
    _, rows, _ = setup_run()
    request_body = run.body("annotation", rows[0])
    contract = {"endpoint_sha256": hashlib.sha256(url.encode()).hexdigest(), "timeout_seconds": 60,
        "request_body_sha256": [run.canonical(request_body)]}
    sent = []
    def respond(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "fake", "object": "chat.completion", "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "{}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
    monkeypatch.setattr(run.httpx, "HTTPTransport", lambda **_: httpx.MockTransport(respond))
    with run.bounded_client("annotation", contract, env) as client:
        assert client.max_retries == 0 and client.timeout == 60 and client._client.timeout.read == 60
        parameters = dict(request_body)
        thinking = parameters.pop("enable_thinking")
        client.chat.completions.create(**parameters, extra_body={"enable_thinking": thinking})
        for endpoint, payload in ((url + "/embeddings", request_body),
                (url + "/chat/completions", request_body | {"model": "other"}),
                (url + "/chat/completions", request_body | {"target_class": "SECRET"})):
            with pytest.raises(RuntimeError, match="unapproved request"):
                client._client.post(endpoint, json=payload)
    assert sent == [request_body]
    with pytest.raises(RuntimeError, match="endpoint changed"):
        run.bounded_client("annotation", contract | {"endpoint_sha256": "changed"}, env)


def test_metrics_use_new_arms_and_keep_failures_and_ambiguity_separate():
    value = run.metric_adapter.population_metrics([
        {"target_class": "SUFFICIENT", "v5_3": "SUFFICIENT", "v5_3_1": "INSUFFICIENT"},
        {"target_class": "INSUFFICIENT", "v5_3": "SUFFICIENT", "v5_3_1": None},
        {"target_class": "AMBIGUOUS", "v5_3": "INSUFFICIENT", "v5_3_1": "SUFFICIENT"}])
    arm = value["arms"]["v5_3_1"]
    assert value["cases"] == 3 and value["binary_cases"] == 2
    assert arm["binary_agreement_possible_bounds"] == [0, .5]
    assert arm["admitted_binary_proxy_agreement"] is None and arm["ambiguous_outcomes"]["SUFFICIENT"] == 1


def test_export_and_review_freeze_use_new_namespace_and_require_all_fifty(tmp_path, monkeypatch):
    contract, short, _ = setup_run()
    rows = [{**short[0], "question_id": f"TRAIN_Q{i:03}"} for i in range(50)]
    contract.update(max_calls=50, input_byte_bounds=[10000] * 50)
    root = tmp_path / "blind_annotation"
    monkeypatch.setattr(run, "DATA", tmp_path)
    monkeypatch.setattr(run, "ANNOTATION_ROOT", root)
    monkeypatch.setattr(run, "load_probe", lambda stage: (contract, rows))
    monkeypatch.setattr(run.annotation, "DATA", tmp_path)
    monkeypatch.setattr(run.annotation, "FREEZE", tmp_path / "public_freeze.json")
    monkeypatch.setattr(run.annotation, "CONTRACT", tmp_path / "contract.json")
    write = run.prep.prior.selection.prior
    write._write_json(run.annotation.CONTRACT, contract)
    write._write_jsonl(tmp_path / "inputs.jsonl", rows)
    client = AnnotationClient()
    run.run_annotation(root=root, client=client, max_new_calls=1)
    with pytest.raises(RuntimeError, match="all50"):
        run.export_draft()
    run.run_annotation(root=root, client=client)
    assert run.export_draft() == 50 and client.calls == 50
    draft = run.shared._read_jsonl(tmp_path / "blind_draft_annotations.jsonl")
    write._write_jsonl(tmp_path / "review.jsonl", draft)
    result = run.freeze_review(tmp_path / "review.jsonl")
    assert result["cases"] == 50 and result["counts"] == {"SUFFICIENT": 50, "INSUFFICIENT": 0, "AMBIGUOUS": 0}
    assert not result["class_support_at_least_five_each"] and not result["classifier_predictions_created"]
    assert len(run.shared._read_jsonl(tmp_path / "targets.jsonl")) == 50
    assert client.calls == 50  # Export/freeze only replay accounted saved responses.


def test_final_evaluation_waits_for_all_outcomes_and_keeps_case_labels_private(tmp_path, monkeypatch):
    contract, short = classifier_fixture()
    rows = [{**short[0], "question_id": f"TRAIN_Q{i:03}"} for i in range(50)]
    labels = [{"question_id": row["question_id"], "target_class": "SUFFICIENT" if i < 21 else "INSUFFICIENT" if i < 45 else "AMBIGUOUS"} for i, row in enumerate(rows)]
    targets = tmp_path / "targets.jsonl"
    run.prep.prior.selection.prior._write_jsonl(targets, labels)
    contract.update(max_calls_total=100, case_order=[r["question_id"] for r in rows],
        targets_sha256=run.shared._sha256(targets), target_counts={"SUFFICIENT": 21, "INSUFFICIENT": 24, "AMBIGUOUS": 5},
        arms={a: {"max_calls": 50, "max_output_tokens": 2048, "request_input_byte_bounds": [10000] * 50} for a in run.prep.BUILDERS})
    monkeypatch.setattr(run, "load_probe", lambda stage: (contract, rows))
    monkeypatch.setattr(run, "DATA", tmp_path)
    monkeypatch.setattr(run, "ROOT", tmp_path / "paired")
    monkeypatch.setattr(run, "RESULT", tmp_path / "public.json")
    with pytest.raises(RuntimeError, match="all100"):
        run.evaluate()
    client = ClassifierClient()
    run.run_classification(root=run.ROOT, client=client)
    value = run.evaluate()
    assert value["provider_calls"] == client.calls == 100
    assert value["primary"]["binary_cases"] == value["common_valid_binary"]["cases"] == 45
    assert value["paired_binary_transitions"] == {"both_agree": 21, "baseline_only": 0, "candidate_only": 0, "neither_agrees": 24}
    assert not value["promotion_allowed"] and value["formal_pass_gate"] is None
    assert "target_class" not in json.dumps(value) and "private_transitions" not in value
    with pytest.raises(RuntimeError, match="overwrite"):
        run.evaluate()


def test_actual_annotation_stop_is_not_a_quality_result_or_retry_permission():
    value = run.shared._read_json(run.REPORT / "v5_3_1_prospective_annotation_stop.json")
    assert value["status"] == "ANNOTATION_STOPPED_MISSING_CITATION_NO_CLASSIFICATION"
    assert value["provider_calls"] == value["annotation_calls"] == value["original_schema_failures"] == 3
    assert value["format_only_recoveries"] == value["accounted_valid_or_format_recovered_annotations"] == 2
    assert value["unattempted_annotation_cases"] == 47 and value["classification_calls"] == 0
    assert value["estimated_cost_cny"] == pytest.approx(.0748575)
    assert value["usage"] == {"prompt_tokens": 14112, "completion_tokens": 975, "total_tokens": 15087}
    assert value["missing_evidence_locators_for_supporting_source_ids"] == ["Source 5"]
    assert not any(value[k] for k in ("labels_frozen", "quality_metrics_available", "promotion_allowed", "dev_opened", "paid_retries", "pending_exists", "runner_lock_exists"))
    assert '"question":' not in json.dumps(value) and "raw_response\":" not in json.dumps(value)
