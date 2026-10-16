import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.evals import refusal_prospective_preparation as prep
from tests.test_refusal_natural_diagnostic_preparation import rows
from tests.test_refusal_v5_preparation import fixture_rows


def test_selection_excludes_old_families_and_deduplicates_new_batch_deterministically():
    snapshots, results = fixture_rows()
    snapshots[1]["question"] = snapshots[0]["question"].upper()
    for index, (snapshot, result) in enumerate(zip(snapshots, results, strict=True)):
        result["reranked_document_ids"] = [f"doc{index}"] * 14
        for candidate in snapshot["fused_candidates"]:
            candidate["document_id"] = f"doc{index}"
    packets, population = prep.select_new_packets(snapshots, results, {"TRAIN_Q001"}, cases=1)
    assert packets[0]["question_id"] == "TRAIN_Q003" and population["eligible_before_within_batch_dedup"] == 1
    selected, _ = prep.select_new_packets(snapshots, results, set(), cases=2)
    assert len({p["question"].casefold() for p in selected}) == 2
    assert prep.select_new_packets(snapshots[::-1], results[::-1], set(), cases=2)[0] == selected
    assert all(p["annotation"]["target_class"] is None and len(p["sources"]) == 14 for p in selected)
    assert "HIDDEN" not in json.dumps(selected)
    with pytest.raises(RuntimeError, match="not enough fresh"):
        prep.select_new_packets(snapshots, results, set(), cases=3)
    with pytest.raises(RuntimeError, match="not enough fresh"):
        prep.select_new_packets(*fixture_rows(), set(), cases=2)  # Same leading document.
    for cases in (0, -1, True):
        with pytest.raises(ValueError):
            prep.select_new_packets(snapshots, results, set(), cases=cases)
    snapshots[0]["question_id"] = results[0]["question_id"] = "DEV_Q001"
    with pytest.raises(RuntimeError, match="TRAIN-only"):
        prep.select_new_packets(snapshots, results, set(), cases=1)


def test_preflight_binds_unchanged_clean_requests_and_enforces_context_count_and_budget():
    base = rows()
    data = [{**base[i % len(base)], "question_id": f"TRAIN_Q{i:03}"} for i in range(50)]
    budget = prep.preflight(data)
    assert budget["max_calls_total"] == 150 and budget["planning_cost_cny"] < 25
    assert budget["stage_caps_cny"] == {"annotation": 10, "paired_classification": 15}
    for index, row in enumerate(data):
        dirty = {**row, "annotation": {"target_class": "LEAK", "notes": "SECRET"}}
        messages = {"annotation": prep.annotation.messages(dirty), **{
            arm: builder(prep.shared._classifier_input_from_row(dirty)) for arm, builder in prep.paired.BUILDERS.items()}}
        for arm, value in messages.items():
            assert prep.canonical(value) == budget["arms"][arm]["request_sha256"][index]
            assert all(marker not in value[1]["content"] for marker in (row["question_id"], "target_class", "SECRET"))
        assert messages["v5_2_1"][1:] == messages["v5_3"][1:]
    with pytest.raises(RuntimeError, match="complete frozen 50"):
        prep.preflight(data[:-1])
    oversized = copy.deepcopy(data)
    oversized[0]["sources"][0]["content"] = "A" * 256_000
    with pytest.raises(RuntimeError, match="context band"):
        prep.preflight(oversized)
    expensive = copy.deepcopy(data)
    for row in expensive:
        row["sources"][0]["content"] = "A" * 100_000
    with pytest.raises(RuntimeError, match="stage budgets"):
        prep.preflight(expensive)


def test_manifest_is_unlabeled_inert_private_and_bound_to_frozen_sources():
    manifest = prep.shared._read_json(prep.REPORT)
    previous = prep.shared._read_json(prep.selection.REPORT)
    assert manifest["cases"] == len(set(manifest["selected_question_ids"])) == 50
    assert not set(manifest["selected_question_ids"]) & set(previous["recorded_excluded_question_ids"] + previous["family_proxy_excluded_question_ids"] + previous["selected_question_ids"])
    assert manifest["previous_selection_canonical_sha256"] == prep.canonical(previous) == prep.PREVIOUS_SELECTION_SHA256
    assert manifest["provider_calls"] == 0 and not manifest["labels_frozen"] and manifest["targets_sha256"] is None
    assert not any(manifest["controls"][key] for key in ("runner_ready", "executable_run_contract", "dev_opened", "gold_fields_used", "promotion_allowed"))
    assert manifest["controls"]["formal_pass_gate"] is None and manifest["controls"]["no_paid_retries"]
    assert "not independent expert gold" in manifest["provenance"]
    assert all(key not in json.dumps(manifest) for key in ('"question":', '"content":', '"target_class":'))
    for name, expected in manifest["source_hashes"].items():
        assert hashlib.sha256((Path("experiments/evals") / name).read_text(encoding="utf-8").encode()).hexdigest() == expected
    assert hashlib.sha256(prep.PROTOCOL.read_text(encoding="utf-8").encode()).hexdigest() == manifest["protocol_normalized_sha256"]


def test_changed_snapshot_or_previous_selection_stops_before_reading_rows(monkeypatch):
    monkeypatch.setattr(prep.shared, "_sha256", lambda path: "changed")
    with pytest.raises(RuntimeError, match="artifact changed"):
        prep.prepare("unused")
    monkeypatch.setattr(prep.shared, "_sha256", lambda path: prep.selection.prior.reuse.EXPECTED_SNAPSHOT_SHA256
                        if path.name == "train_fused_snapshot.jsonl" else prep.selection.prior.reuse.EXPECTED_RESULTS_SHA256)
    monkeypatch.setattr(prep.shared, "_read_json", lambda path: {})
    monkeypatch.setattr(prep.shared, "_read_jsonl", lambda path: pytest.fail("rows read before selection check"))
    with pytest.raises(RuntimeError, match="previous frozen selection"):
        prep.prepare("unused")
