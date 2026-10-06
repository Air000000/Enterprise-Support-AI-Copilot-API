import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.evals import refusal_v5_1_paired_runner as probe

paired = probe.paired


def setup_probe(tmp_path, monkeypatch):
    contract = paired.shared._read_json(probe.CONTRACT_PATH)
    ids = paired.shared._read_json(paired.REPORT_DIR / "v5_train_selection.json")["selected_question_ids"]
    rows = [{"question_id": qid, "question": f"Synthetic task {index}",
             "sources": [{"source_id": f"Source {rank}", "content": f"Evidence {rank}\n\nCondition\n"}
                         for rank in range(1, 15)]} for index, qid in enumerate(ids)]
    inputs = tmp_path / "inputs.jsonl"
    for row in rows:
        paired.shared._append_jsonl(inputs, row)
    contract["frozen_inputs"]["inputs_path"] = str(inputs)
    contract["frozen_inputs"]["inputs_sha256"] = paired.shared._sha256(inputs)
    for arm, builder in probe.BUILDERS.items():
        contract["arms"][arm]["request_input_byte_bounds"] = [
            sum(len(m["content"].encode()) for m in builder(paired.shared._classifier_input_from_row(r))) + 512
            for r in rows]
    path = tmp_path / "contract.json"
    paired.write_json(path, contract)
    monkeypatch.setattr(probe, "CONTRACT_PATH", path)
    monkeypatch.setattr(probe, "CONTRACT_SHA256", paired.canonical_sha256(contract))
    return contract, rows


def response(arm, *, bad_id=False, usage=True):
    payload = {"decision": "SUFFICIENT", "reason": "Synthetic support", "supporting_source_ids": ["Source 1"]}
    if arm == "v5_1":
        payload = {"requirements": [{"requirement": "Explain task", "status": "SUPPORTED",
                                     "evidence": [{"source_id": "Source 1", "segment_ids": [99] if bad_id else [1, 3]}],
                                     "reason": "Synthetic support"}], "unresolved_ambiguities": []}
    return SimpleNamespace(
        id="synthetic-request", usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10, total_tokens=110) if usage else None,
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)), finish_reason="stop")])


def client(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def state(root, rows, contract):
    return paired.load_state(root, rows, contract, builders=probe.BUILDERS,
                             result_parser=probe.parse_result, contract_sha256=probe.CONTRACT_SHA256)


def test_terminal_failure_resumes_next_pair_no_retry_labels_blind_and_all_case_reporting(tmp_path, monkeypatch):
    contract, rows = setup_probe(tmp_path, monkeypatch)
    root, calls = tmp_path / "run", []
    original = paired.shared._read_json
    original_lines = paired.shared._read_jsonl

    def blind_json(path):
        assert Path(path).name != "v5_train_annotation_freeze.json"
        return original(path)

    def blind_jsonl(path):
        assert Path(path) != Path(contract["frozen_inputs"]["targets_path"])
        return original_lines(path)

    monkeypatch.setattr(paired.shared, "_read_json", blind_json)
    monkeypatch.setattr(paired.shared, "_read_jsonl", blind_jsonl)

    def create(**kwargs):
        calls.append(kwargs)
        arm = "v5_1" if kwargs["max_tokens"] == 2048 else "v4"
        assert "TRAIN_Q" not in json.dumps(kwargs["messages"])
        return response(arm, bad_id=len(calls) == 2)

    provider = client(create)
    with pytest.raises(RuntimeError, match="failure checkpointed"):
        probe.run_paid(output_dir=root, client=provider)
    assert len(calls) == 2 and state(root, rows, contract)[1] == 2
    with pytest.raises(RuntimeError, match="resume-after-audit"):
        probe.run_paid(output_dir=root, client=provider)
    with pytest.raises(RuntimeError, match="all 40 attempted"):
        probe.evaluate(output_dir=root)
    events = probe.run_paid(output_dir=root, client=provider, resume_after_audit=True)
    probe.run_paid(output_dir=root, client=provider)
    assert len(calls) == 40 and calls[1] != calls[2] and len(events) == 40
    assert [c["max_tokens"] for c in calls[:4]] == [512, 2048, 2048, 512]
    assert events[1]["result"] is None and events[1]["raw_model_admission"] == "SUFFICIENT"
    assert events[2]["result"]["citation_selections"] == [[{"source_id": "Source 1", "segment_ids": [1, 3]}]]
    assert [e["quote"] for e in events[2]["result"]["requirements"][0]["evidence"]] == ["Evidence 1\n", "Condition\n"]
    monkeypatch.setattr(paired.shared, "_read_json", original)
    monkeypatch.setattr(paired.shared, "_read_jsonl", original_lines)
    summary = probe.evaluate(output_dir=root)
    assert summary["contract_sha256"] == probe.CONTRACT_SHA256
    assert summary["common_valid_pairs"] == 19 and summary["case_denominator"] == 20
    assert summary["provider_calls"] == 40 and summary["estimated_cost_cny"] == pytest.approx(0.0192)
    assert summary["arms"]["v5_1"]["valid_predictions"] == 19
    assert summary["arms"]["v5_1"]["failure_case_count"] == 1
    assert summary["arms"]["v5_1"]["common_valid_class_denominators"] == {"SUFFICIENT": 13, "INSUFFICIENT": 6}
    assert summary["arms"]["v5_1"]["sufficient_recall_against_ai_proxy"] == 1
    assert summary["arms"]["v5_1"]["valid_and_agree_all_case_count"] == 13
    assert summary["arms"]["v5_1"]["valid_and_agree_all_case_rate"] == 13 / 20
    assert summary["cases"][0]["predictions"]["v5_1"] is None
    assert summary["paired_transitions_against_ai_proxy"]["missing_valid_prediction"] == 1
    assert summary["v5_1_admission_attempts"][0]["final_valid_decision"] is None
    assert not summary["promotion_allowed"] and not summary["dev_opened"]
    with pytest.raises(FileExistsError):
        probe.evaluate(output_dir=root)
    # Replay must use the new selection parser, not accept edited extracted quotes.
    path = root / "v5_1" / "attempts.jsonl"
    records = original_lines(path)
    records[1]["result"]["requirements"][0]["evidence"][0]["quote"] = "Edited"
    path.write_text("".join(json.dumps(e) + "\n" for e in records), encoding="utf-8")
    with pytest.raises(RuntimeError, match="raw/derived"):
        probe.run_paid(output_dir=root, client=provider)
    assert len(calls) == 40


@pytest.mark.parametrize("failure", ["provider", "usage", "pending", "old_identity"])
def test_unknown_charge_and_old_checkpoint_block_resume(tmp_path, monkeypatch, failure):
    contract, rows = setup_probe(tmp_path, monkeypatch)
    root, calls = tmp_path / "run", []

    def create(**kwargs):
        calls.append(kwargs)
        if failure == "provider":
            raise RuntimeError("SECRET")
        return response("v4", usage=False)

    if failure == "old_identity":
        paired.load_state(root, rows, contract, create=True)
    elif failure == "pending":
        paired.load_state(root, rows, contract, create=True, builders=probe.BUILDERS,
                          result_parser=probe.parse_result, contract_sha256=probe.CONTRACT_SHA256)
        paired.write_json(root / "pending_request.json", {"call_index": 0, "arm": "v4", "question_id": rows[0]["question_id"]})
    else:
        with pytest.raises(RuntimeError, match="unknown usage"):
            probe.run_paid(output_dir=root, client=client(create))
        assert "SECRET" not in (root / "v4" / "attempts.jsonl").read_text(encoding="utf-8")
    with pytest.raises(RuntimeError):
        probe.run_paid(output_dir=root, client=client(create), resume_after_audit=True)
    assert len(calls) == (1 if failure in {"provider", "usage"} else 0)


def test_frozen_contract_prices_boundaries_and_sources(tmp_path, monkeypatch):
    real = paired.shared._read_json(probe.CONTRACT_PATH)
    assert paired.canonical_sha256(real) == probe.CONTRACT_SHA256
    assert real["max_calls_total"] == real["required_outcomes_total"] == 40
    assert real["controls"]["failed_pair_is_terminal"] and real["controls"]["authorization_required"]
    for arm in probe.BUILDERS:
        limits = real["arms"][arm]
        bound = (sum(limits["request_input_byte_bounds"]) * 3 + 20 * limits["max_output_tokens"] * 18) / 1_000_000
        assert limits["conservative_cost_bound_cny"] == pytest.approx(bound)
        assert limits["max_calls"] == 20
    assert sum(a["conservative_cost_bound_cny"] for a in real["arms"].values()) == pytest.approx(2.982159)
    assert real["pricing_snapshot"]["hard_cost_cap_cny"] == 3
    contract, _ = setup_probe(tmp_path, monkeypatch)
    probe.load_probe()
    monkeypatch.setattr(probe.citation, "CLASSIFIER_SYSTEM_PROMPT_V5_1_DEV", "Changed")
    with pytest.raises(RuntimeError, match="prompt changed"):
        probe.load_probe()
    source = Path(probe.citation.__file__).read_text(encoding="utf-8")
    assert hashlib.sha256(source.encode()).hexdigest() == contract["citation_selector_normalized_source_sha256"]


def test_all_format_failures_keep_null_quality_and_no_41st_call(tmp_path, monkeypatch):
    contract, rows = setup_probe(tmp_path, monkeypatch)
    root, calls = tmp_path / "run", []

    def create(**kwargs):
        calls.append(kwargs)
        result = response("v4")
        result.choices[0].message.content = "invalid JSON"
        return result

    provider = client(create)
    for index in range(40):
        with pytest.raises(RuntimeError, match="failure checkpointed"):
            probe.run_paid(output_dir=root, client=provider, resume_after_audit=index > 0)
        assert len(calls) == index + 1
    probe.run_paid(output_dir=root, client=provider, resume_after_audit=True)
    assert len(calls) == 40 and state(root, rows, contract)[1] == 40
    summary = probe.evaluate(output_dir=root)
    assert summary["common_valid_pairs"] == 0
    for arm in probe.BUILDERS:
        stats = summary["arms"][arm]
        assert stats["case_denominator"] == stats["failed_attempts"] == stats["provider_calls"] == 20
        assert stats["valid_predictions"] == stats["valid_and_agree_all_case_count"] == 0
        assert stats["agreement_with_ai_proxy"] is None
        assert stats["sufficient_recall_against_ai_proxy"] is None
        assert stats["insufficient_recall_against_ai_proxy"] is None
        assert stats["common_valid_class_denominators"] == {"SUFFICIENT": 0, "INSUFFICIENT": 0}
    assert summary["paired_transitions_against_ai_proxy"]["missing_valid_prediction"] == 20
