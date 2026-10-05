import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.evals import refusal_development_probe as probe


def _setup(tmp_path, monkeypatch):
    contract = probe.reuse.shared._read_json(probe.CONTRACT_PATH)
    reviews = probe.reuse.shared._read_jsonl(contract["frozen_inputs"]["review_path"])
    rows = [{"question_id": r["question_id"], "question": f"Example {index}",
             "sources": [{"source_id": f"Source {rank}", "content": f"Evidence {rank}"}
                         for rank in range(1, 15)]}
            for index, r in enumerate(reviews)]
    inputs = tmp_path / "inputs.jsonl"
    for row in rows:
        probe.reuse.shared._append_jsonl(inputs, row)
    contract["frozen_inputs"]["classifier_inputs_path"] = str(inputs)
    contract["frozen_inputs"]["classifier_inputs_sha256"] = probe.reuse.shared._sha256(inputs)
    path = tmp_path / "contract.json"
    probe.reuse.shared._write_json(path, contract)
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    monkeypatch.setattr(probe, "CONTRACT_PATH", path)
    monkeypatch.setattr(probe, "CONTRACT_SHA256", digest)
    return contract, rows


def test_resume_keeps_review_out_of_paid_loop_and_evaluates_descriptively(tmp_path, monkeypatch):
    contract, rows = _setup(tmp_path, monkeypatch)
    output = tmp_path / "run"
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        assert kwargs["messages"][0]["content"] == probe.CLASSIFIER_SYSTEM_PROMPT_V4_DEV
        assert "TRAIN_Q" not in json.dumps(kwargs["messages"])
        assert kwargs["temperature"] == 0 and kwargs["max_tokens"] == 512
        return SimpleNamespace(
            id=f"req-{len(calls)}",
            choices=[SimpleNamespace(message=SimpleNamespace(content=(
                "bad JSON" if len(calls) == 3 else json.dumps({
                    "decision": "INSUFFICIENT", "reason": "Missing support",
                    "supporting_source_ids": [],
                })
            )))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10, total_tokens=110),
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    original_read = probe.reuse.shared._read_jsonl

    def blind_read(path):
        assert Path(path) != Path(contract["frozen_inputs"]["review_path"])
        return original_read(path)

    monkeypatch.setattr(probe.reuse.shared, "_read_jsonl", blind_read)
    with pytest.raises(RuntimeError, match="invalid JSON"):
        probe.run_paid(output_dir=output, client=client)
    with pytest.raises(RuntimeError, match="all 50 predictions"):
        probe.evaluate(output_dir=output)
    probe.run_paid(output_dir=output, client=client)
    probe.run_paid(output_dir=output, client=client)
    assert len(calls) == 51
    assert calls[2]["messages"] == calls[3]["messages"]
    assert calls[0]["messages"] != calls[3]["messages"]
    monkeypatch.setattr(probe.reuse.shared, "_read_jsonl", original_read)
    summary = probe.evaluate(output_dir=output)
    assert summary["binary_cases"] == 48 and len(summary["questionable_descriptive_only"]) == 2
    assert summary["agreement_with_ai_review"] == 31 / 48
    assert summary["balanced_agreement_with_ai_review"] == 0.5
    assert summary["provider_calls"] == 51 and summary["failed_attempts"] == 1
    assert summary["total_tokens"] == 5610
    assert summary["estimated_cost_cny"] == pytest.approx(51 * (100 * 3 + 10 * 18) / 1_000_000)
    assert not summary["promotion_allowed"] and not summary["historical_fail_changed"]
    with pytest.raises(FileExistsError):
        probe.evaluate(output_dir=output)
    identity = output / "run_identity.json"
    identity.write_text('{}')
    with pytest.raises(RuntimeError, match="different development run"):
        probe.run_paid(output_dir=output, client=client)
    assert len(calls) == 51


def test_contract_and_unbound_checkpoint_fail_closed(tmp_path, monkeypatch):
    contract, rows = _setup(tmp_path, monkeypatch)
    output = tmp_path / "run"
    output.mkdir()
    (output / "predictions.jsonl").write_text("")
    with pytest.raises(RuntimeError, match="unexpected checkpoint"):
        probe.run_paid(output_dir=output)
    contract["classifier"]["max_calls"] += 1
    probe.reuse.shared._write_json(probe.CONTRACT_PATH, contract)
    with pytest.raises(RuntimeError, match="contract changed"):
        probe.load_probe()


def test_committed_contract_and_prompt_stay_frozen():
    contract = probe.reuse.shared._read_json(probe.CONTRACT_PATH)
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == probe.CONTRACT_SHA256
    assert hashlib.sha256(probe.CLASSIFIER_SYSTEM_PROMPT_V4_DEV.encode()).hexdigest() == (
        contract["classifier"]["prompt_sha256"]
    )
    assert contract["classifier"]["max_calls"] == 55
    assert contract["pricing_snapshot"]["hard_cost_cap_cny"] == 1.5
    assert contract["controls"]["promotion_allowed"] is False


def test_cli_uses_explicit_env_without_printing_key(tmp_path, monkeypatch, capsys):
    _setup(tmp_path, monkeypatch)
    env_file = tmp_path / "local.env"
    env_file.write_text(
        "DASHSCOPE_PHASE_B_API_KEY=private_test_key_never_print\n"
        "DASHSCOPE_PHASE_B_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1\n"
    )
    monkeypatch.setenv("DASHSCOPE_PHASE_B_API_KEY", "stale")
    monkeypatch.setenv("DASHSCOPE_PHASE_B_BASE_URL", "https://wrong.example/v1")
    monkeypatch.setattr(sys, "argv", [
        "probe", "--env-file", str(env_file), "--output-dir", str(tmp_path / "run")
    ])
    probe.main()
    output = capsys.readouterr().out
    assert "PREFLIGHT=PASS" in output and "PROVIDER_CALLS_THIS_COMMAND=0" in output
    assert "AUTH_KEY_SOURCE=DASHSCOPE_PHASE_B_API_KEY" in output
    assert "private_test_key" not in output
