import copy
import hashlib
from pathlib import Path

import pytest

from experiments.evals import refusal_natural_diagnostic_preparation as prep


def rows():
    return [{"question_id": f"TRAIN_Q{i:03}", "question": "A fictional test question?", "sources": [
        {"source_id": f"Source {rank}", "content": f"Fictional source {rank}.\n"}
        for rank in range(1, 15)]} for i in range(30)]


def test_preflight_reuses_baseline_preserves_payload_and_excludes_metadata():
    data = rows()
    budget = prep.paired_preflight(data)
    assert budget["arms"]["v5_1"] == prep.natural.paired_preflight(data)["arms"]["v5_1"]
    assert budget["planned_calls"] == 60 and budget["planning_cost_cny"] < 8
    for row in data:
        value = prep.paired.shared._classifier_input_from_row(row)
        old, new = [builder(value) for builder in prep.BUILDERS.values()]
        assert old[1:] == new[1:]
        assert all(key not in new[1]["content"] for key in (row["question_id"], "target_class", "notes"))
    with pytest.raises(RuntimeError, match="complete frozen 30"):
        prep.paired_preflight(data[:-1])
    too_big = copy.deepcopy(data)
    too_big[0]["sources"][0]["content"] = "A" * 256_000
    with pytest.raises(RuntimeError, match="context band"):
        prep.paired_preflight(too_big)


def test_public_manifest_cannot_upgrade_proxy_diagnosis_into_acceptance():
    value = prep.paired.shared._read_json(prep.REPORT)
    assert value["status"] == "PREPARED_NOT_RUN_NOT_AUTHORIZED" and value["provider_calls"] == 0
    assert value["target_counts"] == {"SUFFICIENT": 21, "INSUFFICIENT": 4, "AMBIGUOUS": 5}
    assert value["binary_cases"] + value["ambiguous_cases"] == value["cases"] == 30
    assert value["metrics"]["binary_target_agreement_denominator_including_failures"] == 25
    assert value["controls"]["formal_pass_gate"] is None
    for key in ("paid_runner_ready", "executable_run_contract", "dev_opened", "promotion_allowed",
                "original_class_support_rule_met", "original_formal_rule_changed", "independent_or_representative_acceptance_claim"):
        assert value["controls"][key] is False
    assert value["controls"]["max_calls_total"] == 60
    assert value["preflight"]["pricing_snapshot"]["hard_cost_cap_cny"] == 8
    selection = prep.paired.shared._read_json(prep.natural.REPORT)
    freeze = prep.paired.shared._read_json(prep.prior.REPORT / "v5_2_annotation_freeze.json")
    assert value["case_order"] == selection["selected_question_ids"]
    assert value["inputs_sha256"] == freeze["inputs_sha256"]
    assert value["targets_sha256"] == freeze["targets_sha256"]
    assert value["annotation_freeze_canonical_sha256"] == prep.natural.prior.canonical_sha256(freeze)
    assert value["preflight"]["arms"]["v5_1"] == selection["paired_preflight"]["arms"]["v5_1"]
    for name, expected in value["source_hashes"].items():
        assert hashlib.sha256((Path("experiments/evals") / name).read_text(encoding="utf-8").encode()).hexdigest() == expected


def test_frozen_target_change_is_rejected_before_preflight(tmp_path, monkeypatch):
    targets = tmp_path / "targets.jsonl"
    targets.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(prep, "TARGETS", targets)
    with pytest.raises(RuntimeError, match="target bindings changed"):
        prep.build_preparation()
