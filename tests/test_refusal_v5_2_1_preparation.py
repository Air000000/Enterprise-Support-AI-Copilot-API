"""Offline preparation bindings, not provider outcome measurements."""

import copy
import json

import pytest

from experiments.evals import refusal_v5_2_1_preparation as prep


def test_frozen_preparation_matches_requests_and_keeps_execution_inert():
    frozen = json.loads(prep.REPORT.read_text(encoding="utf-8"))
    assert frozen == prep.build_preparation()
    assert frozen["cases"] == 14 and frozen["max_calls_total"] == 28
    assert frozen["provider_calls"] == 0 and frozen["planning_cost_cny"] < 3
    assert frozen["controls"]["authorization_required"]
    assert not frozen["controls"]["executable_run_contract"]
    assert not frozen["controls"]["paid_runner_ready"]
    assert not frozen["controls"]["dev_opened"] and not frozen["controls"]["promotion_allowed"]
    assert frozen["checkpoint_root"] != prep.prior.ROOT.as_posix()
    rows = prep.prior.controls.classifier_rows(prep.prior.controls.load_cases())
    canonical = prep.prior.controls.prep.prior.canonical_sha256
    for index, row in enumerate(rows):
        value = prep.prior.paired.shared._classifier_input_from_row(row)
        old, new = [builder(value) for builder in prep.BUILDERS.values()]
        assert old[1:] == new[1:] and old[0] != new[0]
        for arm, messages in zip(prep.BUILDERS, (old, new), strict=True):
            assert canonical(messages) == frozen["arms"][arm]["request_sha256"][index]
            payload = json.loads(messages[1]["content"])
            assert set(payload) == {"question", "sources"}
            assert row["question_id"] not in messages[1]["content"]
    assert prep.candidate.validate_requirement_coverage_response is prep.prior.coverage.validate_requirement_coverage_response
    # Fresh baseline matches the old frozen v5.2 requests, but uses a separate future journal.
    old_contract = prep.prior.paired.shared._read_json(prep.prior.CONTRACT_PATH)
    assert frozen["arms"]["v5_2"]["request_sha256"] == old_contract["arms"]["v5_2"]["request_sha256"]


def test_context_or_budget_drift_fails_and_frozen_outputs_cannot_be_overwritten(monkeypatch, tmp_path):
    original = prep.prior.controls.load_cases()
    changed = copy.deepcopy(original)
    changed[0]["sources"][0]["content"] += "Unexpected change."
    monkeypatch.setattr(prep.prior.controls, "load_cases", lambda: changed)
    with pytest.raises(RuntimeError, match="controls changed"):
        prep.build_preparation()
    monkeypatch.setattr(prep.prior.controls, "load_cases", lambda: original)
    monkeypatch.setitem(prep.PRICING, "hard_cost_cap_cny", 0)
    with pytest.raises(RuntimeError, match="budget exceeded"):
        prep.build_preparation()
    output = tmp_path / "freeze.json"
    prep.prior.controls.prep.prior._write_json(output, {"immutable": True})
    with pytest.raises(FileExistsError):
        prep.prior.controls.prep.prior._write_json(output, {"immutable": False})
    assert json.loads(output.read_text(encoding="utf-8")) == {"immutable": True}
