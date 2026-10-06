import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.evals import refusal_v5_paired_runner as probe


def setup_probe(tmp_path, monkeypatch):
    contract = probe.shared._read_json(probe.CONTRACT_PATH)
    ids = probe.shared._read_json(probe.REPORT_DIR / "v5_train_selection.json")["selected_question_ids"]
    rows = [{"question_id": qid, "question": f"Example {index}",
             "sources": [{"source_id": f"Source {rank}", "content": f"Evidence {rank}"}
                         for rank in range(1, 15)]} for index, qid in enumerate(ids)]
    inputs = tmp_path / "inputs.jsonl"
    for row in rows:
        probe.shared._append_jsonl(inputs, row)
    contract["frozen_inputs"]["inputs_path"] = str(inputs)
    contract["frozen_inputs"]["inputs_sha256"] = probe.shared._sha256(inputs)
    for arm, builder in probe.BUILDERS.items():
        contract["arms"][arm]["request_input_byte_bounds"] = [
            sum(len(m["content"].encode()) for m in builder(probe.shared._classifier_input_from_row(r))) + 512
            for r in rows]
    path = tmp_path / "contract.json"
    probe.write_json(path, contract)
    monkeypatch.setattr(probe, "CONTRACT_PATH", path)
    monkeypatch.setattr(probe, "CONTRACT_SHA256", probe.canonical_sha256(contract))
    return contract, rows


def response(arm, *, bad_quote=False, missing=False, usage=True, finish_reason="stop"):
    payload = {"decision": "SUFFICIENT", "reason": "Visible support", "supporting_source_ids": ["Source 1"]}
    if arm == "v5":
        payload = {"requirements": [{"requirement": "Explain the issue", "status": "SUPPORTED",
                                     "evidence": [{"source_id": "Source 1", "quote": "Invented" if bad_quote else "Evidence 1"}],
                                     "reason": "Visible support"}], "unresolved_ambiguities": []}
        if missing:
            payload["requirements"][0].update(status="MISSING", evidence=[])
    return SimpleNamespace(
        id="request-id", usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10, total_tokens=110) if usage else None,
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)), finish_reason=finish_reason)],
    )


def fake_client(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_paired_resume_preserves_raw_checks_order_and_blindness(tmp_path, monkeypatch):
    contract, rows = setup_probe(tmp_path, monkeypatch)
    root = tmp_path / "run"
    calls = []
    original_read_jsonl = probe.shared._read_jsonl
    original_read_json = probe.shared._read_json

    def read_jsonl(path):
        assert Path(path) != Path(contract["frozen_inputs"]["targets_path"])
        return original_read_jsonl(path)

    def read_json(path):
        assert Path(path).name != "v5_train_annotation_freeze.json"
        return original_read_json(path)

    monkeypatch.setattr(probe.shared, "_read_jsonl", read_jsonl)
    monkeypatch.setattr(probe.shared, "_read_json", read_json)

    def create(**kwargs):
        calls.append(kwargs)
        arm = "v5" if kwargs["max_tokens"] == 2048 else "v4"
        assert "TRAIN_Q" not in json.dumps(kwargs["messages"])
        assert kwargs["temperature"] == 0 and kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["extra_body"] == {"enable_thinking": False}
        return response(arm, bad_quote=len(calls) == 2, missing=arm == "v5" and len(calls) > 3)

    client = fake_client(create)
    with pytest.raises(RuntimeError, match="failure checkpointed"):
        probe.run_paid(output_dir=root, client=client)
    with pytest.raises(RuntimeError, match="resume-after-audit"):
        probe.run_paid(output_dir=root, client=client)
    with pytest.raises(RuntimeError, match="all 40"):
        probe.evaluate(output_dir=root)
    assert len(calls) == 2
    events = probe.run_paid(output_dir=root, client=client, resume_after_audit=True)
    probe.run_paid(output_dir=root, client=client)
    assert len(calls) == 41 and calls[1] == calls[2] and calls[0] != calls[2]
    assert [c["max_tokens"] for c in calls[:5]] == [512, 2048, 2048, 2048, 512]
    assert events[1]["result"] is None and events[1]["raw_model_admission"] == "SUFFICIENT"
    assert "Invented" in events[1]["raw_response"]
    assert events[2]["result"]["requirements"][0]["status"] == "SUPPORTED"
    assert events[3]["result"]["decision"] == "INSUFFICIENT"
    assert not (root / "pending_request.json").exists() and not (root / "runner.lock").exists()
    monkeypatch.setattr(probe.shared, "_read_jsonl", original_read_jsonl)
    monkeypatch.setattr(probe.shared, "_read_json", original_read_json)
    summary = probe.evaluate(output_dir=root)
    assert summary["common_valid_pairs"] == 20 and summary["provider_calls"] == 41
    assert summary["estimated_cost_cny"] == pytest.approx(41 * 0.00048)
    assert summary["arms"]["v5"]["failure_case_count"] == 1
    assert summary["arms"]["v5"]["case_denominator"] == 20
    assert summary["arms"]["v4"]["sufficient_recall_against_ai_proxy"] == 1
    assert summary["arms"]["v4"]["insufficient_recall_against_ai_proxy"] == 0
    assert len(summary["correlated_families"]) == 2
    assert summary["v5_admission_attempts"][0]["final_valid_decision"] is None
    assert not summary["dev_opened"] and not summary["promotion_allowed"]
    with pytest.raises(FileExistsError):
        probe.evaluate(output_dir=root)
    # A hand-edited gate result must fail raw replay, before another request.
    path = root / "v5" / "attempts.jsonl"
    altered = original_read_jsonl(path)
    altered[1]["result"]["decision"] = "INSUFFICIENT"
    path.write_text("".join(json.dumps(e) + "\n" for e in altered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="raw/derived"):
        probe.run_paid(output_dir=root, client=client, resume_after_audit=True)
    assert len(calls) == 41 and len(rows) == 20


@pytest.mark.parametrize("failure", ["provider", "unknown_usage", "truncated", "bad_json", "bad_quote"])
def test_failure_stops_without_automatic_retry(tmp_path, monkeypatch, failure):
    contract, rows = setup_probe(tmp_path, monkeypatch)
    root = tmp_path / "run"
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if failure == "provider":
            raise RuntimeError("SECRET MUST NOT BE SAVED")
        result = response("v4", usage=failure != "unknown_usage", finish_reason="length" if failure == "truncated" else "stop")
        if failure == "bad_json":
            result.choices[0].message.content = "bad JSON"
        if failure == "bad_quote":
            return response("v5", bad_quote=True)  # Invalid v4 schema too; charged failure.
        return result

    with pytest.raises(RuntimeError):
        probe.run_paid(output_dir=root, client=fake_client(create))
    assert len(calls) == 1
    raw = (root / "v4" / "attempts.jsonl").read_text(encoding="utf-8")
    assert "SECRET" not in raw
    record = json.loads(raw)
    assert (record["result"] is not None) == (failure == "unknown_usage")
    if failure in {"provider", "unknown_usage"}:
        assert record["estimated_cost_cny"] is None
        with pytest.raises(RuntimeError, match="unknown usage"):
            probe.run_paid(output_dir=root, client=fake_client(create), resume_after_audit=True)
    else:
        assert probe.load_state(root, rows, contract)[3] == pytest.approx(0.00048)
        with pytest.raises(RuntimeError, match="resume-after-audit"):
            probe.run_paid(output_dir=root, client=fake_client(create))
    assert len(calls) == 1


def test_frozen_identity_pending_lock_and_reservation_are_fail_closed(tmp_path, monkeypatch):
    contract, rows = setup_probe(tmp_path, monkeypatch)
    root = tmp_path / "run"
    probe.load_state(root, rows, contract, create=True)
    pending = root / "pending_request.json"
    probe.write_json(pending, {"call_index": 0, "question_id": rows[0]["question_id"], "arm": "v4"})
    with pytest.raises(RuntimeError, match="uncheckpointed"):
        probe.load_state(root, rows, contract)
    pending.unlink()
    lock = root / "runner.lock"
    lock.touch()
    with pytest.raises(FileExistsError):
        probe.run_paid(output_dir=root, client=fake_client(lambda **kwargs: pytest.fail("unexpected request")))
    assert lock.exists()
    lock.unlink()
    contract["pricing_snapshot"]["hard_cost_cap_cny"] = 0.00001
    monkeypatch.setattr(probe, "load_probe", lambda inputs_path=None: (contract, rows))
    with pytest.raises(RuntimeError, match="insufficient budget"):
        probe.run_paid(output_dir=root, client=fake_client(lambda **kwargs: pytest.fail("unexpected request")))
    assert not pending.exists()
    probe.write_json(root / "v4_old.json", {})
    with pytest.raises(RuntimeError, match="unexpected paired"):
        probe.load_state(root, rows, contract)


def test_frozen_input_and_contract_cannot_be_changed(tmp_path, monkeypatch):
    contract, _ = setup_probe(tmp_path, monkeypatch)
    probe.load_probe()
    path = Path(contract["frozen_inputs"]["inputs_path"])
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="input SHA"):
        probe.load_probe()
    path = probe.CONTRACT_PATH
    changed = probe.shared._read_json(path)
    changed["arms"]["v5"]["max_output_tokens"] = 1
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="contract changed"):
        probe.load_probe()


def test_failure_budget_and_raw_usage_tampering_block_requests(tmp_path, monkeypatch):
    contract, rows = setup_probe(tmp_path, monkeypatch)
    root = tmp_path / "run"
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        result = response("v4")
        result.choices[0].message.content = "bad JSON"
        return result

    client = fake_client(create)
    for audited in (False, True):
        with pytest.raises(RuntimeError, match="failure checkpointed"):
            probe.run_paid(output_dir=root, client=client, resume_after_audit=audited)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        probe.run_paid(output_dir=root, client=client, resume_after_audit=True)
    assert len(calls) == 2
    path = root / "v4" / "attempts.jsonl"
    records = probe.shared._read_jsonl(path)
    records[0]["total_tokens"] += 1
    path.write_text("".join(json.dumps(e) + "\n" for e in records), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid checkpoint usage"):
        probe.load_state(root, rows, contract)
