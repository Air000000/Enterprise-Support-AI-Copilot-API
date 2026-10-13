import copy
import json

import pytest

from experiments.evals import refusal_v5_2_controls as controls
from experiments.evals.refusal_citation_selection import build_classifier_messages_v5_1_dev
from experiments.evals.refusal_requirement_coverage import build_classifier_messages_v5_2_dev
from experiments.evals.refusal_evidence_sufficiency import ClassifierInput, ContextSource


def test_fixed_pairs_preserve_context_and_do_not_leak_targets_into_requests():
    cases = controls.load_cases()
    rows = controls.classifier_rows(cases)
    assert len(rows) == 14 and all(set(r) == {"question_id", "question", "sources"} for r in rows)
    for row in rows:
        value = ClassifierInput(row["question"], tuple(
            ContextSource(s["source_id"], "", "", s["content"]) for s in row["sources"]))
        old, new = build_classifier_messages_v5_1_dev(value), build_classifier_messages_v5_2_dev(value)
        assert old[1] == new[1]
        payload = json.loads(new[1]["content"])
        assert set(payload) == {"question", "sources"}
        assert "CONTROL_" not in new[1]["content"]
        assert all(key not in new[1]["content"] for key in ("expected_class", "rationale", '"family"'))
        assert payload["sources"][0]["segments"][0]["content"] == row["sources"][0]["content"]


@pytest.mark.parametrize("defect", ["duplicate", "metadata", "class", "question", "unchanged", "empty"])
def test_bad_control_pairs_fail_before_any_request(defect):
    rows = copy.deepcopy(controls.load_cases())
    if defect == "duplicate":
        rows[1]["question_id"] = rows[0]["question_id"]
    elif defect == "metadata":
        rows[0]["sources"][0]["document_id"] = "metadata must not be accepted"
    elif defect == "class":
        rows[0]["expected_class"] = "AMBIGUOUS"
    elif defect == "question":
        rows[1]["question"] = "A different question"
    elif defect == "unchanged":
        rows[1]["sources"] = rows[0]["sources"]
    else:
        rows[0]["sources"][0]["content"] = " "
    with pytest.raises(ValueError):
        controls.classifier_rows(rows)


def test_frozen_manifest_is_only_a_preflight_and_natural_shortfall_remains():
    value = json.loads(controls.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert value["cases_canonical_sha256"] == controls.prep.prior.canonical_sha256(controls.load_cases())
    assert value["cases"] == 14 and value["pairs"] == 7
    assert value["expected_class_counts"] == {"SUFFICIENT": 7, "INSUFFICIENT": 7}
    assert value["provider_calls"] == 0 and not value["dev_opened"] and not value["promotion_allowed"]
    assert value["natural_insufficient_cases_unchanged"] == 4
    assert not value["natural_class_support_met_unchanged"]
    assert value["paired_preflight"]["planned_calls"] == 28
    assert not value["paired_preflight"]["executable_run_contract"] and not value["paid_runner_ready"]
    assert value["paired_preflight"] == controls.prep.paired_preflight(controls.classifier_rows(controls.load_cases()))
