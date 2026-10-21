"""Synthetic reproductions of observed shapes; never repair or rescore real traces."""

import json

import pytest

from experiments.evals import refusal_v5_3_1_runner as run
from tests.test_refusal_version_audit_clarification import trace


@pytest.mark.parametrize("failure", ("nested_ambiguities", "duplicate_citation_fields"))
def test_observed_bad_shapes_remain_rejected_by_both_frozen_arms(failure):
    value, payload = trace("What is ElmCache 3.2's timeout?",
        "ElmCache 3.2 timeout is 40 seconds.", "SUPPORTED")
    # ponytail: reproduce two shapes with the existing fixture and strict gate.
    if failure == "nested_ambiguities":
        payload["coverage_checks"]["unresolved_ambiguities"] = payload.pop("unresolved_ambiguities")
        raw, error = json.dumps(payload), "coverage response schema mismatch"
    else:
        original = json.dumps({"source_id": "Source 1", "segment_ids": [1]})
        duplicate = '{"source_id": "Source 1", "segment_ids": [1], "source_id": "Source 1", "segment_ids": [1]}'
        raw = json.dumps(payload).replace(original, duplicate, 1)
        assert raw != json.dumps(payload)
        error = "duplicate JSON field"
    assert isinstance(json.loads(raw), dict)  # JSON-object shape alone is insufficient.
    for arm in run.prep.BUILDERS:
        with pytest.raises(ValueError, match=error):
            run.parse_result(arm, raw, {"question": value.question, "sources": [
                {"source_id": s.source_id, "content": s.content} for s in value.sources]}, "stop")


def test_published_posthoc_counts_bind_original_result_and_keep_scores_unchanged():
    audit = run.shared._read_json(run.REPORT / "v5_3_1_posthoc_audit.json")
    result = run.shared._read_json(run.RESULT)
    freeze = run.shared._read_json(run.FREEZE)
    assert audit["status"] == "POST_HOC_DIAGNOSTIC_NOT_RESCORING"
    assert audit["bindings"]["original_result_sha256"] == run.canonical(result)
    assert audit["bindings"]["classification_contract_sha256"] == result["contract_sha256"]
    assert audit["bindings"]["targets_sha256"] == freeze["targets_sha256"] == result["targets_sha256"]
    assert audit["bindings"]["reviewed_annotations_sha256"] == freeze["review_sha256"]
    assert audit["bindings"]["classification_journal_sha256"] == result["journal_sha256"]
    assert audit["population_outcomes"] == result["provider_calls"] == 100
    assert audit["selected_cases"] == 10 and audit["inspected_arm_outcomes"] == 20
    assert audit["binary_disagreement_cases"] == 8 and audit["common_valid_binary_disagreements"] == 4
    assert audit["common_valid_both_wrong_cases"] == result["paired_binary_transitions"]["neither_agrees"] == 2
    assert sum(a["all_case_failures"] for a in result["primary"]["arms"].values()) == audit["schema_failures"] == 4
    assert sum(f["responses"] for f in audit["schema_families"].values()) == 4
    assert audit["failed_finish_reasons"] == {"stop": 4} and audit["failed_usage_all_known"]
    assert audit["failed_completion_tokens_range"] == [651, 1099]
    categories = audit["semantic_primary_hypotheses"].values()
    assert sum(c["wrong_valid_outcomes"] for c in categories) == audit["wrong_valid_outcomes"] == 10
    for arm in run.prep.BUILDERS:
        metrics = result["primary"]["arms"][arm]
        assert sum(c["arms"][arm] for c in categories) == 50 - metrics["valid_correct_binary"] - metrics["binary_failures"] == 5
        assert audit["original_primary_agreement"][arm] == metrics["binary_agreement_including_failures"] == .86
    assert audit["new_provider_calls"] == 0
    assert not any(audit[k] for k in ("raw_repaired", "labels_changed", "metrics_changed", "frozen_prompts_changed",
        "validator_changed", "dev_opened", "gold_or_qrels_opened", "external_document_links_opened", "promotion_allowed"))
    assert not any(k in json.dumps(audit) for k in ('"question_id":', '"question":', '"raw_response":', '"target_class":'))
