import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.evals import refusal_v5_3_1_preparation as prep
from tests.test_refusal_natural_diagnostic_preparation import rows


def fifty_rows():
    base = rows()
    return [{**base[i % len(base)], "question_id": f"TRAIN_Q{i:03}"} for i in range(50)]


def test_preflight_preserves_annotation_and_baseline_but_changes_only_candidate_prompt():
    data = fifty_rows()
    old, value = prep.prior.preflight(data), prep.preflight(data)
    assert set(value["arms"]) == {"annotation", "v5_3", "v5_3_1"}
    assert value["max_calls_total"] == 150 and value["planning_cost_cny"] < 25
    for arm in ("annotation", "v5_3"):
        assert value["arms"][arm] == old["arms"][arm]
    for i, row in enumerate(data):
        dirty = row | {"annotation": {"target_class": "SECRET", "notes": "DO_NOT_SEND"}}
        item = prep.shared._classifier_input_from_row(dirty)
        baseline, candidate = [builder(item) for builder in prep.BUILDERS.values()]
        assert baseline[1:] == candidate[1:] and baseline[0] != candidate[0]
        assert prep.canonical(candidate) == value["arms"]["v5_3_1"]["request_sha256"][i]
        assert all(marker not in json.dumps(candidate) for marker in (row["question_id"], "SECRET", "DO_NOT_SEND"))
    with pytest.raises(RuntimeError, match="complete frozen 50"):
        prep.preflight(data[:-1])
    huge = copy.deepcopy(data)
    huge[0]["sources"][0]["content"] = "A" * 256_000
    with pytest.raises(RuntimeError, match="context band"):
        prep.preflight(huge)
    expensive = copy.deepcopy(data)
    for row in expensive:
        row["sources"][0]["content"] = "A" * 100_000
    with pytest.raises(RuntimeError, match="stage budgets"):
        prep.preflight(expensive)


def test_manifest_is_inert_and_bound_without_private_rows_in_public_output(monkeypatch):
    frozen = prep.shared._read_json(prep.prior.REPORT)
    monkeypatch.setattr(prep.readiness, "load_inputs", lambda: (frozen, fifty_rows()))
    monkeypatch.setattr(prep, "PRIOR_STATE_PATHS", ())
    value = prep.build_preparation()
    assert value["inputs_sha256"] == frozen["inputs_sha256"]
    assert value["case_order"] == frozen["selected_question_ids"]
    assert value["selection_canonical_sha256"] == prep.canonical(frozen)
    assert value["status"] == "PREPARED_NOT_ANNOTATED_NOT_AUTHORIZED" and value["provider_calls"] == 0
    assert not value["labels_frozen"] and value["targets_sha256"] is None
    for key in ("executable_run_contract", "runner_ready", "dev_opened", "promotion_allowed",
                "old_execution_permissions_reused", "independent_acceptance_claim", "generation_allowed", "retrieval_allowed"):
        assert value["controls"][key] is False
    assert value["controls"]["formal_pass_gate"] is None
    assert "not independent expert gold" in value["provenance"]
    assert all(key not in json.dumps(value) for key in ('"question":', '"content":', '"target_class":'))
    assert value.pop("plan_canonical_sha256") == prep.canonical(value)
    for name, expected in value["source_hashes"].items():
        assert hashlib.sha256((Path("experiments/evals") / name).read_text(encoding="utf-8").encode()).hexdigest() == expected
    assert hashlib.sha256(prep.PROTOCOL.read_text(encoding="utf-8").encode()).hexdigest() == value["protocol_normalized_sha256"]


def test_existing_local_state_blocks_preparation_before_preflight(tmp_path, monkeypatch):
    artifact = tmp_path / "targets.jsonl"
    artifact.touch()
    monkeypatch.setattr(prep.readiness, "load_inputs", lambda: ({}, []))
    monkeypatch.setattr(prep, "PRIOR_STATE_PATHS", (artifact,))
    monkeypatch.setattr(prep, "preflight", lambda _: pytest.fail("existing outcomes bypassed"))
    with pytest.raises(RuntimeError, match="requires audit"):
        prep.build_preparation()


def test_saved_manifest_binds_fifty_cases_frozen_sources_and_protocol():
    value = prep.shared._read_json(prep.REPORT)
    frozen = prep.shared._read_json(prep.prior.REPORT)
    assert len(value["case_order"]) == len(set(value["case_order"])) == 50
    assert value["case_order"] == frozen["selected_question_ids"]
    assert value["inputs_sha256"] == frozen["inputs_sha256"]
    assert value["selection_canonical_sha256"] == prep.canonical(frozen)
    assert value["preflight"]["pricing"]["checked_date"] == "2026-10-20"
    assert value["preflight"]["stage_caps_cny"] == {"annotation": 10, "paired_classification": 15}
    for arm in ("annotation", "v5_3"):
        assert value["preflight"]["arms"][arm] == frozen["preflight"]["arms"][arm]
    for arm in value["preflight"]["arms"].values():
        assert len(arm["request_sha256"]) == len(arm["input_byte_plus_framing_bounds"]) == 50
    assert value["provider_calls"] == 0 and not value["labels_frozen"] and value["targets_sha256"] is None
    assert not value["controls"]["executable_run_contract"] and not value["controls"]["runner_ready"]
    assert all(key not in json.dumps(value) for key in ('"question":', '"content":', '"target_class":'))
    assert value.pop("plan_canonical_sha256") == prep.canonical(value)
    for name, digest in value["source_hashes"].items():
        assert hashlib.sha256((Path("experiments/evals") / name).read_text(encoding="utf-8").encode()).hexdigest() == digest
    assert hashlib.sha256(prep.PROTOCOL.read_text(encoding="utf-8").encode()).hexdigest() == value["protocol_normalized_sha256"]


def test_paid_cli_flags_do_not_exist():
    for flag in ("--execute", "--run-paid", "--confirm-pr80-authorization"):
        result = subprocess.run([sys.executable, "-B", "-X", "utf8", "-m",
            "experiments.evals.refusal_v5_3_1_preparation", flag], capture_output=True, timeout=30)
        assert result.returncode == 2 and b"unrecognized arguments" in result.stderr
