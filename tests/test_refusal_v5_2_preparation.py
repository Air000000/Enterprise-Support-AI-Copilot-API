import copy
import json

import pytest

from experiments.evals import refusal_v5_2_preparation as prep
from tests.test_refusal_v5_preparation import fixture_rows


def test_family_proxy_selection_and_clean_paired_payload():
    snapshots, results = fixture_rows()
    snapshots[1]["question"] = snapshots[0]["question"].upper()
    for index, result in enumerate(results):
        result["reranked_document_ids"] = [f"doc-{index}"] * 14
        for candidate in snapshots[index]["fused_candidates"]:
            candidate["document_id"] = f"doc-{index}"
    excluded = prep.family_exclusions(snapshots, results, {"TRAIN_Q001"})
    assert excluded == {"TRAIN_Q001", "TRAIN_Q002"}
    packets, population = prep.prior.select_packets(snapshots, results, excluded, cases=1, seed=prep.SEED)
    assert population == 1 and packets[0]["question_id"] == "TRAIN_Q003"
    assert packets[0]["annotation"]["target_class"] is None
    budget = prep.paired_preflight(packets)
    assert budget["planned_calls"] == 2 and not budget["executable_run_contract"]
    assert all(a["max_output_tokens"] == 2048 for a in budget["arms"].values())
    assert "HIDDEN" not in json.dumps(packets) and '"annotation":' not in json.dumps(budget)
    assert budget == prep.paired_preflight(copy.deepcopy(packets))
    with pytest.raises(RuntimeError, match="not enough"):
        prep.prior.select_packets(snapshots, results, excluded, cases=2, seed=prep.SEED)
    with pytest.raises(RuntimeError, match="empty population"):
        prep.paired_preflight([])


def test_prepare_manifest_is_inert_and_private_packets_not_tracked():
    manifest = json.loads(prep.REPORT.read_text(encoding="utf-8"))
    assert manifest["cases"] == len(set(manifest["selected_question_ids"])) == 30
    assert not set(manifest["selected_question_ids"]) & set(
        manifest["recorded_excluded_question_ids"] + manifest["family_proxy_excluded_question_ids"])
    assert manifest["provider_calls"] == 0 and not manifest["dev_opened"]
    assert not manifest["labels_frozen"] and not manifest["promotion_allowed"]
    assert manifest["paired_preflight"]["planned_calls"] == 60
    assert all(key not in json.dumps(manifest) for key in ('"question":', '"content":', '"target_class":'))
