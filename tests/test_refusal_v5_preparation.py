import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from experiments.evals.refusal_v5_preparation import select_packets
from experiments.evals.refusal_structured_evidence import CLASSIFIER_SYSTEM_PROMPT_V5_DEV
from experiments.evals.refusal_development_policy import CLASSIFIER_SYSTEM_PROMPT_V4_DEV


def fixture_rows():
    snapshots, results = [], []
    for qid in ("TRAIN_Q001", "TRAIN_Q002", "TRAIN_Q003"):
        candidates = [{"chunk_id": f"c{i}", "document_id": "PRIVATE", "content": f"Text {i}"}
                      for i in range(14)]
        snapshots.append({"question_id": qid, "question": f"Question {qid}",
                          "fused_candidates": candidates})
        results.append({"question_id": qid, "reranked_chunk_ids": [c["chunk_id"] for c in candidates],
                        "reranked_document_ids": ["PRIVATE"] * 14, "gold_answer": "HIDDEN"})
    return snapshots, results


def test_selection_excludes_prior_ids_and_preserves_ordered_visible_payload():
    snapshots, results = fixture_rows()
    packets, population = select_packets(snapshots, results, {"TRAIN_Q001"}, cases=2)
    assert population == 2
    assert [p["question_id"] for p in packets] == ["TRAIN_Q002", "TRAIN_Q003"]
    assert select_packets(snapshots[::-1], results[::-1], {"TRAIN_Q001"}, cases=2)[0] == packets
    assert set(packets[0]) == {"question_id", "question", "sources", "annotation"}
    assert packets[0]["sources"] == [{"source_id": f"Source {i + 1}", "content": f"Text {i}"}
                                     for i in range(14)]
    assert packets[0]["annotation"]["target_class"] is None


@pytest.mark.parametrize("defect", ["dev", "duplicate", "missing_chunk", "wrong_doc", "short_top14"])
def test_frozen_identity_errors_stop_selection(defect):
    snapshots, results = copy.deepcopy(fixture_rows())
    if defect == "dev":
        snapshots[0]["question_id"] = results[0]["question_id"] = "DEV_Q001"
    elif defect == "duplicate":
        snapshots.append(snapshots[0])
    elif defect == "missing_chunk":
        snapshots[0]["fused_candidates"] = []
    elif defect == "wrong_doc":
        results[0]["reranked_document_ids"][0] = "other"
    else:
        results[0]["reranked_chunk_ids"] = ["c0"]
    with pytest.raises(RuntimeError):
        select_packets(snapshots, results, set(), cases=3)


def test_insufficient_remaining_population_is_not_silently_resampled():
    snapshots, results = fixture_rows()
    with pytest.raises(RuntimeError):
        select_packets(snapshots, results, {"TRAIN_Q001", "TRAIN_Q002"}, cases=2)


def test_frozen_paired_plan_keeps_population_prompts_and_cost_bound():
    report_dir = Path("experiments/evals/reports/refusal_evidence_sufficiency")
    selection = json.loads((report_dir / "v5_train_selection.json").read_text(encoding="utf-8"))
    contract = json.loads((report_dir / "v5_train_run_contract.json").read_text(encoding="utf-8"))
    targets = [json.loads(line) for line in (report_dir / "v5_train_targets.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(selection["excluded_question_ids"]) == 220
    assert len(selection["selected_question_ids"]) == 20
    assert not set(selection["excluded_question_ids"]) & set(selection["selected_question_ids"])
    assert [t["question_id"] for t in targets] == selection["selected_question_ids"]
    assert Counter(t["target_class"] for t in targets) == {"SUFFICIENT": 14, "INSUFFICIENT": 6}
    assert contract["frozen_inputs"]["inputs_sha256"] == selection["inputs_sha256"]
    assert hashlib.sha256((report_dir / "v5_train_targets.jsonl").read_bytes()).hexdigest() == contract["frozen_inputs"]["targets_sha256"]
    assert not contract["controls"]["promotion_allowed"]
    assert not contract["controls"]["dev_opened"]
    total = 0
    for name, prompt in (("v4", CLASSIFIER_SYSTEM_PROMPT_V4_DEV), ("v5", CLASSIFIER_SYSTEM_PROMPT_V5_DEV)):
        arm = contract["arms"][name]
        assert arm["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
        assert arm["max_calls"] == 22
        bounds, output = arm["request_input_byte_bounds"], arm["max_output_tokens"]
        assert len(bounds) == 20
        expected = (sum(bounds) * 3 + 20 * output * 18 + 2 * (max(bounds) * 3 + output * 18)) / 1_000_000
        assert arm["conservative_cost_bound_cny"] == expected
        total += expected
    assert total == contract["pricing_snapshot"]["conservative_cost_bound_cny"]
    assert total <= contract["pricing_snapshot"]["hard_cost_cap_cny"] == 3
