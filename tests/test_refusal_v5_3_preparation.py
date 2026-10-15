import copy
import hashlib
from pathlib import Path

import pytest

from experiments.evals import refusal_v5_3_preparation as prep
from tests.test_refusal_natural_diagnostic_preparation import rows


def test_preflight_preserves_baseline_and_payload_with_context_and_count_guards():
    data = rows()
    value = prep.paired_preflight(data)
    assert value["arms"]["v5_2_1"] == prep.prior.paired_preflight(data)["arms"]["v5_2_1"]
    assert set(value["arms"]) == {"v5_2_1", "v5_3"}
    assert value["planned_calls"] == 60 and value["planning_cost_cny"] < 8
    for row in data:
        item = prep.prior.paired.shared._classifier_input_from_row(row)
        baseline, candidate = [builder(item) for builder in prep.BUILDERS.values()]
        assert baseline[1:] == candidate[1:] and baseline[0] != candidate[0]
        assert all(key not in candidate[1]["content"] for key in (row["question_id"], "target_class", "notes"))
    with pytest.raises(RuntimeError, match="complete frozen 30"):
        prep.paired_preflight(data[:-1])
    oversized = copy.deepcopy(data)
    oversized[0]["sources"][0]["content"] = "A" * 256_000
    with pytest.raises(RuntimeError, match="context band"):
        prep.paired_preflight(oversized)
    expensive = copy.deepcopy(data)
    for row in expensive:
        row["sources"][0]["content"] = "A" * 100_000
    with pytest.raises(RuntimeError, match="budget exceeded"):
        prep.paired_preflight(expensive)


def test_manifest_binds_frozen_population_sources_protocol_and_observed_provenance():
    value = prep.prior.paired.shared._read_json(prep.REPORT)
    previous = prep.prior.paired.shared._read_json(prep.prior.REPORT)
    assert value["status"] == "PREPARED_NOT_RUN_NOT_AUTHORIZED" and value["provider_calls"] == 0
    for key in ("case_order", "inputs_sha256", "targets_sha256", "target_counts", "metrics", "known_correlated_pair"):
        assert value[key] == previous[key]
    assert value["previous_preparation_canonical_sha256"] == prep.canonical(previous)
    assert value["observed_prior_result_canonical_sha256"] == prep.canonical(
        prep.prior.paired.shared._read_json(prep.prior.prior.REPORT / "natural_v5_2_1_result.json"))
    assert value["preflight"]["arms"]["v5_2_1"] == previous["preflight"]["arms"]["v5_2_1"]
    assert value["preflight"]["pricing_snapshot"]["checked_date"] == "2026-10-15"
    assert value["preflight"]["pricing_snapshot"]["hard_cost_cap_cny"] == 8
    assert value["checkpoint_root"] == "data/refusal_natural_v5_3/run_v1"
    assert "after observing" in value["provenance"] and "Fresh concurrent" in value["baseline_policy"]
    for key in ("paid_runner_ready", "executable_run_contract", "dev_opened", "promotion_allowed",
                "original_class_support_rule_met", "original_formal_rule_changed", "independent_or_representative_acceptance_claim"):
        assert value["controls"][key] is False
    assert value["controls"]["formal_pass_gate"] is None
    assert value["controls"]["max_calls_total"] == 60 and value["controls"]["max_calls_per_arm"] == 30
    assert value["controls"]["sdk_max_retries"] == 0
    for name, expected in value["source_hashes"].items():
        assert hashlib.sha256((Path("experiments/evals") / name).read_text(encoding="utf-8").encode()).hexdigest() == expected
    assert hashlib.sha256(prep.PROTOCOL.read_text(encoding="utf-8").encode()).hexdigest() == value["protocol_normalized_sha256"]


def test_changed_previous_preparation_is_rejected_before_loading_private_inputs(monkeypatch):
    monkeypatch.setattr(prep.prior, "build_preparation", lambda: {})
    monkeypatch.setattr(prep.prior.paired.v2, "load_classifier_inputs", lambda *a, **kw: pytest.fail("private inputs read"))
    with pytest.raises(RuntimeError, match="previous frozen preparation changed"):
        prep.build_preparation()
