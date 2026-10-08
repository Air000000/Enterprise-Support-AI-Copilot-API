"""Synthetic, hand-authored responses: gate checks, not provider quality scores."""

import copy
import json

import pytest

from experiments.evals import refusal_citation_selection as old
from experiments.evals import refusal_requirement_coverage as candidate
from experiments.evals.refusal_evidence_sufficiency import ClassifierInput, ContextSource


def given(question="What are possible causes of an Atlas cache warning?"):
    return ClassifierInput(question, (
        ContextSource("Source 1", "PRIVATE_CHUNK", "PRIVATE_DOC",
                      "Atlas warnings can have several causes.\nOld releases permit source downloads.\n"),
        ContextSource("Source 2", "PRIVATE_CHUNK_2", "PRIVATE_DOC_2",
                      "Atlas release 13 permits source downloads for its supported platform.\n"),
    ))


def response():
    support = {"status": "SUPPORTED", "evidence": [{"source_id": "Source 1", "segment_ids": [1]}],
               "reason": "General possible causes, not a unique incident diagnosis."}
    return {"requirements": [{"requirement": "Describe possible causes", **copy.deepcopy(support)}],
            "coverage_checks": {"requested_outcomes": copy.deepcopy(support),
                                "applicability": copy.deepcopy(support),
                                "version_and_time": {"status": "NOT_REQUIRED", "evidence": [],
                                                     "reason": "No version or time claim requested."}},
            "unresolved_ambiguities": []}


def validate(payload, question=None):
    return candidate.validate_requirement_coverage_response(json.dumps(payload), given(question) if question else given())


def test_payload_is_unchanged_full_context_and_general_cause_not_forced_to_refuse():
    new_messages = candidate.build_classifier_messages_v5_2_dev(given())
    assert new_messages[1] == old.build_classifier_messages_v5_1_dev(given())[1]
    assert "PRIVATE_" not in json.dumps(new_messages)
    assert new_messages[0]["content"] == candidate.CLASSIFIER_SYSTEM_PROMPT_V5_2_DEV
    result = validate(response())
    assert result["decision"] == "SUFFICIENT"
    assert len(result["requirements"]) == 3  # Material requirement + two applicable audits.
    assert result["coverage_checks"]["version_and_time"]["status"] == "NOT_REQUIRED"
    assert result["requirements"][-1]["evidence"][0]["quote"] == "Atlas warnings can have several causes.\n"


@pytest.mark.parametrize("facet", candidate.COVERAGE_FACETS)
def test_omitting_any_mandatory_coverage_facet_is_an_error_not_a_prediction(facet):
    payload = response()
    del payload["coverage_checks"][facet]
    with pytest.raises(ValueError, match="all coverage facets"):
        validate(payload)


def test_declared_time_gap_and_incident_applicability_gap_block_admission():
    payload = response()
    payload["coverage_checks"]["version_and_time"] = {
        "status": "MISSING", "evidence": [], "reason": "Historical permission does not prove latest availability."}
    assert validate(payload, "Can I download the latest Atlas release source?")["decision"] == "INSUFFICIENT"
    # Same handcrafted material checks without the extra audit would pass the old gate.
    original = {k: v for k, v in payload.items() if k != "coverage_checks"}
    assert old.validate_citation_selection_response(json.dumps(original), given())["decision"] == "SUFFICIENT"
    payload = response()
    payload["coverage_checks"]["applicability"].update(
        status="CONDITIONAL", reason="Possible causes do not establish this incident's cause or fix.")
    result = validate(payload, "What caused this incident and how do I fix it?")
    assert result["decision"] == "INSUFFICIENT"
    assert result["gaps"][0]["requirement"] == "Coverage audit: applicability"


def test_specific_version_with_visible_support_and_all_gap_statuses_use_existing_gate():
    payload = response()
    payload["coverage_checks"]["version_and_time"] = {
        "status": "SUPPORTED", "evidence": [{"source_id": "Source 2", "segment_ids": [1]}],
        "reason": "Visible applicability explicitly covers requested release 13."}
    assert validate(payload, "Can release 13 source be obtained?")["decision"] == "SUFFICIENT"
    for status in ("MISSING", "CONDITIONAL", "CONFLICTING"):
        payload["coverage_checks"]["requested_outcomes"].update(status=status)
        assert validate(payload)["decision"] == "INSUFFICIENT"
    payload["coverage_checks"]["requested_outcomes"]["status"] = "SUPPORTED"
    payload["unresolved_ambiguities"] = ["Material target ambiguity."]
    assert validate(payload)["decision"] == "INSUFFICIENT"


def test_invalid_audit_schema_or_locators_are_errors_not_correct_refusals():
    mutations = [
        lambda p: p.update(decision="SUFFICIENT"),
        lambda p: p.update(requirements=[]),
        lambda p: p.update(coverage_checks=[]),
        lambda p: p["coverage_checks"].update(extra={}),
        lambda p: p["coverage_checks"].update(applicability=None),
        lambda p: p["coverage_checks"]["applicability"].update(requirement="extra"),
        lambda p: p["coverage_checks"]["applicability"].update(status="NOT_REQUIRED", evidence=[]),
        lambda p: p["coverage_checks"]["applicability"].update(status="UNKNOWN"),
        lambda p: p["coverage_checks"]["applicability"].update(evidence=[]),
        lambda p: p["coverage_checks"]["applicability"]["evidence"][0].update(segment_ids=[True]),
        lambda p: p["coverage_checks"]["applicability"]["evidence"][0].update(source_id="Source 99"),
        lambda p: p["coverage_checks"]["version_and_time"].update(reason=" "),
        lambda p: p["coverage_checks"]["version_and_time"].update(evidence=[{}]),
    ]
    for mutate in mutations:
        payload = response()
        mutate(payload)
        with pytest.raises(ValueError):
            validate(payload)
    for raw in ("bad JSON", "[]", '{"coverage_checks":{},"coverage_checks":{}}'):
        with pytest.raises(ValueError):
            candidate.validate_requirement_coverage_response(raw, given())


def test_falsely_supported_or_not_required_still_cannot_be_detected_mechanically():
    payload = response()
    question = "Can I download the latest Atlas release source?"
    assert validate(payload, question)["decision"] == "SUFFICIENT"  # Wrong NOT_REQUIRED still passes.
    payload["coverage_checks"]["version_and_time"] = {
        "status": "SUPPORTED", "evidence": [{"source_id": "Source 1", "segment_ids": [2]}],
        "reason": "General historical permission incorrectly treated as latest-release proof."}
    assert validate(payload, question)["decision"] == "SUFFICIENT"  # True quote, wrong semantic status.
