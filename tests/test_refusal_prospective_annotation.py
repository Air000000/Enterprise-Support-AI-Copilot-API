import hashlib
import json

import pytest

from experiments.evals import refusal_prospective_annotation as run
from tests.test_refusal_v5_2_annotation import Client, setup_run


def test_preparing_execution_requires_explicit_authorization_before_loading_inputs(monkeypatch):
    monkeypatch.setattr(run, "load_inputs", lambda: pytest.fail("inputs read without authorization"))
    for value in (False, None, "continue", 1):
        with pytest.raises(RuntimeError, match="explicit PR80"):
            run.prepare_contract(authorization_confirmed=value)


def test_resume_recovery_retains_raw_without_paid_retry(tmp_path):
    contract, rows, value = setup_run()
    contract["max_calls"] = 3
    def loader():
        return contract, rows
    value["annotation"]["notes"] = "Source 1 (Line 1) supports this fact."
    bad = Client(raw=json.dumps(value))
    with pytest.raises(RuntimeError, match="failure saved"):
        run.run_annotation(root=tmp_path, client=bad, probe_loader=loader)
    with pytest.raises(RuntimeError, match="resume-after-audit"):
        run.run_annotation(root=tmp_path, client=bad, probe_loader=loader)
    assert bad.calls == 1
    before = (tmp_path / "attempts.jsonl").read_bytes()
    recovery = {"call_index": 0, "raw_sha256": hashlib.sha256(bad.raw.encode()).hexdigest(),
                "annotation": value["annotation"] | {"notes": "Source 1:L1 supports this fact."}}
    run.prep.selection.prior._write_jsonl(tmp_path / "citation_recoveries.jsonl", [recovery])
    good = Client()
    assert run.run_annotation(root=tmp_path, client=good, probe_loader=loader)["completed"] == 3
    assert run.run_annotation(root=tmp_path, client=good, probe_loader=loader)["calls"] == 3
    assert good.calls == 2 and (tmp_path / "attempts.jsonl").read_bytes().startswith(before)


def test_unknown_charge_lock_budget_and_call_caps_block(tmp_path):
    contract, rows, _ = setup_run()
    def loader():
        return contract, rows
    unknown = Client(fail=True)
    with pytest.raises(RuntimeError, match="failure saved"):
        run.run_annotation(root=tmp_path / "unknown", client=unknown, probe_loader=loader)
    with pytest.raises(RuntimeError, match="unknown charge"):
        run.run_annotation(root=tmp_path / "unknown", client=unknown, probe_loader=loader)
    assert unknown.calls == 1
    good = Client()
    contract["hard_accounting_cap_cny"] = 0
    with pytest.raises(RuntimeError, match="cap reached"):
        run.run_annotation(root=tmp_path / "budget", client=good, probe_loader=loader)
    (tmp_path / "budget" / "runner.lock").touch()
    with pytest.raises(FileExistsError):
        run.run_annotation(root=tmp_path / "budget", client=good, probe_loader=loader)
    for cap in (True, -1, 51):
        with pytest.raises(ValueError):
            run.run_annotation(root=tmp_path, max_new_calls=cap)
    assert good.calls == 0


def test_contract_and_preparation_tamper_fail_before_any_paid_request(tmp_path, monkeypatch):
    manifest = run.shared._read_json(run.prep.REPORT)
    value = run.execution_contract(manifest)
    assert value["max_calls"] == 50 and value["hard_accounting_cap_cny"] == 10
    assert value["controls"]["paid_retries"] is False and value["controls"]["sdk_retries"] == 0
    assert run.canonical(manifest) == run.PREPARATION_SHA256
    path = tmp_path / "tampered.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run.prep, "REPORT", path)
    monkeypatch.setattr(run.shared, "_read_jsonl", lambda path: pytest.fail("private rows read"))
    with pytest.raises(RuntimeError, match="preparation changed"):
        run.load_inputs()


def test_pre_prediction_review_freeze_is_complete_and_never_forces_binary(tmp_path, monkeypatch):
    _, short, value = setup_run()
    rows = [{**short[0], "question_id": f"TRAIN_Q{i:03}"} for i in range(50)]
    draft = [{"question_id": row["question_id"], **value} for row in rows]
    review = [r | {"annotation": r["annotation"] | {"target_class": "AMBIGUOUS"}} for r in draft]
    root = tmp_path / "blind_annotation"
    root.mkdir()
    events = [{"prompt_tokens": 100, "completion_tokens": 100, "result": draft[i]} for i in range(50)]
    run.prep.selection.prior._write_jsonl(root / "attempts.jsonl", events)
    for name, values in (("blind_draft_annotations.jsonl", draft), ("review_working.jsonl", review), ("inputs.jsonl", rows)):
        run.prep.selection.prior._write_jsonl(tmp_path / name, values)
    contract = tmp_path / "contract.json"
    run.prep.selection.prior._write_json(contract, {"synthetic": True})
    monkeypatch.setattr(run, "DATA", tmp_path)
    monkeypatch.setattr(run, "FREEZE", tmp_path / "freeze.json")
    monkeypatch.setattr(run, "CONTRACT", contract)
    monkeypatch.setattr(run, "accounted_draft", lambda **kw: ({"estimated_cost_cny": .1}, events, rows, draft))
    (tmp_path / "paired_run_v1").mkdir()
    with pytest.raises(RuntimeError, match="classifier checkpoint"):
        run.freeze_review(tmp_path / "review_working.jsonl", root=root)
    (tmp_path / "paired_run_v1").rmdir()
    result = run.freeze_review(tmp_path / "review_working.jsonl", root=root)
    assert result["counts"] == {"SUFFICIENT": 0, "INSUFFICIENT": 0, "AMBIGUOUS": 50}
    assert result["class_changes_from_draft"] == 50 and not result["classifier_predictions_created"]
    assert not result["class_support_at_least_five_each"] and not result["promotion_allowed"]
    assert result["targets_sha256"] == run.shared._sha256(tmp_path / "targets.jsonl")
    with pytest.raises(RuntimeError, match="overwrite"):
        run.freeze_review(tmp_path / "review_working.jsonl", root=root)
