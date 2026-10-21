"""Fictional plumbing checks, NOT evidence of real-model quality or reliability."""

import copy
import json

import pytest

from experiments.evals import refusal_single_contract as candidate
from experiments.evals import refusal_version_audit_clarification as frozen
from experiments.evals.refusal_evidence_sufficiency import ClassifierInput, ContextSource
from tests.test_refusal_version_audit_clarification import CASES, trace


@pytest.mark.parametrize("question,text,status,decision", CASES)
def test_single_prompt_preserves_full_payload_and_gate(question, text, status, decision):
    value, payload = trace(question, text + "\n\nEnd of fictional context.\n", status)
    before = frozen.build_classifier_messages_v5_3_1_dev(value)
    messages = candidate.build_classifier_messages_v5_4_dev(value)
    assert messages[1:] == before[1:] and messages[0] != before[0]
    assert messages[0]["content"] == candidate.CLASSIFIER_SYSTEM_PROMPT_V5_4_DEV
    assert candidate.validate_requirement_coverage_response is frozen.validate_requirement_coverage_response
    assert frozen.build_classifier_messages_v5_3_1_dev(value) == before
    assert "PRIVATE_" not in json.dumps(messages)
    result = candidate.validate_requirement_coverage_response(json.dumps(payload), value)
    assert result["decision"] == decision


def test_only_complete_template_has_correct_root_and_citation_arrays():
    prompt = candidate.CLASSIFIER_SYSTEM_PROMPT_V5_4_DEV
    assert prompt.count('{"requirements":') == 1
    template = prompt[prompt.index('{"requirements":'):]
    value, _ = trace("How do I open ElmCache's help menu?",
        "Open Help from ElmCache's toolbar.", "NOT_REQUIRED")
    result = candidate.validate_requirement_coverage_response(template, value)
    assert result["decision"] == "SUFFICIENT"
    payload = json.loads(template)
    ref = {"source_id": "Source 1", "segment_ids": [1]}
    value = ClassifierInput(value.question, value.sources + (
        ContextSource("Source 2", "PRIVATE_CHUNK_2", "PRIVATE_DOC_2", "A second fictional instruction."),))
    payload["requirements"][0]["evidence"] = [ref, {"source_id": "Source 2", "segment_ids": [1]}]
    result = candidate.validate_requirement_coverage_response(json.dumps(payload), value)
    assert len(result["requirements"][0]["evidence"]) == 2


@pytest.mark.parametrize("failure", ("nested_ambiguities", "duplicate_fields", "bad_id", "boolean_id"))
def test_malformed_outputs_are_rejected_not_repaired(failure):
    value, payload = trace(*CASES[0][:3])
    if failure == "nested_ambiguities":
        payload["coverage_checks"]["unresolved_ambiguities"] = payload.pop("unresolved_ambiguities")
    elif failure in {"bad_id", "boolean_id"}:
        payload["requirements"][0]["evidence"][0]["segment_ids"] = [99 if failure == "bad_id" else True]
    raw = json.dumps(payload)
    if failure == "duplicate_fields":
        raw = raw.replace('"segment_ids": [1]', '"segment_ids": [1], "segment_ids": [1]', 1)
    original = raw
    with pytest.raises(ValueError):
        candidate.validate_requirement_coverage_response(raw, value)
    assert raw == original


def test_candidate_does_not_delete_conditions_or_repair_semantic_mistakes():
    value, payload = trace(*CASES[0][:3])
    payload["requirements"].append({"requirement": "Handwritten optional path incorrectly made necessary",
        "status": "CONDITIONAL", "evidence": [], "reason": "Fixture deliberately makes a semantic error."})
    original = copy.deepcopy(payload)
    result = candidate.validate_requirement_coverage_response(json.dumps(payload), value)
    assert result["decision"] == "INSUFFICIENT" and payload == original
    payload["requirements"].pop()
    payload["coverage_checks"]["version_and_time"] = {
        "status": "NOT_REQUIRED", "evidence": [], "reason": "Deliberately ignores an explicit fixed version."}
    # Structural validation cannot detect this semantic error; never claim otherwise.
    result = candidate.validate_requirement_coverage_response(json.dumps(payload), value)
    assert result["decision"] == "SUFFICIENT"
    assert result["coverage_checks"]["version_and_time"]["status"] == "NOT_REQUIRED"
