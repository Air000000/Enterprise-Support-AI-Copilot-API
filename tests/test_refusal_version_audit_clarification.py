"""Fictional handwritten traces check plumbing, NOT real model improvement."""

import copy
import json

import pytest

from experiments.evals import refusal_version_audit_clarification as candidate
from experiments.evals.refusal_evidence_sufficiency import ClassifierInput, ContextSource


CASES = (
    ("What is ElmCache 3.2's timeout?", "ElmCache 3.2 timeout is 40 seconds.", "SUPPORTED", "SUFFICIENT"),
    ("What is ElmCache 4.0's timeout?", "ElmCache 3.2 only: timeout is 40 seconds. No coverage of 4.0.", "MISSING", "INSUFFICIENT"),
    ("What is ElmCache 3.4's timeout?", "For all ElmCache 3.x releases, timeout is 40 seconds.", "SUPPORTED", "SUFFICIENT"),
    ("What is ElmCache's current timeout in 2026?", "Archived 2023 ElmCache timeout: 40 seconds; no later coverage.", "MISSING", "INSUFFICIENT"),
    ("How do I open ElmCache's help menu?", "Open Help from ElmCache's toolbar.", "NOT_REQUIRED", "SUFFICIENT"),
    ("What is ElmCache 3.2's timeout?", "Two equally authoritative active ElmCache 3.2 policies disagree: 40 vs 70 seconds; no precedence.", "CONFLICTING", "INSUFFICIENT"),
)


def trace(question, text, status):
    value = ClassifierInput(question, (ContextSource("Source 1", "PRIVATE_CHUNK", "PRIVATE_DOC", text),))
    ref = [{"source_id": "Source 1", "segment_ids": [1]}]
    facet = {"status": status, "evidence": ref if status in {"SUPPORTED", "CONFLICTING"} else [],
             "reason": "Handwritten fixture, not a model prediction."}
    outcome = copy.deepcopy(facet) if status != "NOT_REQUIRED" else {
        "status": "SUPPORTED", "evidence": ref, "reason": "Visible requested procedure."}
    return value, {"requirements": [{"requirement": "Answer the requested claim in scope", **copy.deepcopy(outcome)}],
        "coverage_checks": {"requested_outcomes": copy.deepcopy(outcome), "applicability": copy.deepcopy(outcome),
                            "version_and_time": facet}, "unresolved_ambiguities": []}


@pytest.mark.parametrize("question,text,status,decision", CASES)
def test_candidate_only_changes_system_prompt_and_reuses_gate(question, text, status, decision):
    value, payload = trace(question, text, status)
    old = candidate.prior.build_classifier_messages_v5_3_dev(value)
    new = candidate.build_classifier_messages_v5_3_1_dev(value)
    assert new[1:] == old[1:] and new[0] != old[0]
    assert new[0]["content"] == candidate.CLASSIFIER_SYSTEM_PROMPT_V5_3_1_DEV
    assert new[0]["content"].startswith(candidate.prior.CLASSIFIER_SYSTEM_PROMPT_V5_3_DEV)
    assert candidate.prior.build_classifier_messages_v5_3_dev(value) == old
    assert "PRIVATE_" not in json.dumps(new)
    assert set(json.loads(new[1]["content"])) == {"question", "sources"}
    assert candidate.validate_requirement_coverage_response is candidate.prior.validate_requirement_coverage_response
    assert candidate.validate_requirement_coverage_response(json.dumps(payload), value)["decision"] == decision


@pytest.mark.parametrize("case_index,wrong_status,decision", [(0, "NOT_REQUIRED", "SUFFICIENT"), (1, "SUPPORTED", "INSUFFICIENT")])
def test_prompt_candidate_does_not_silently_repair_semantically_wrong_outputs(case_index, wrong_status, decision):
    question, text, status, _ = CASES[case_index]
    value, payload = trace(question, text, status)
    payload["coverage_checks"]["version_and_time"] = {
        "status": wrong_status, "evidence": [] if wrong_status == "NOT_REQUIRED" else [{"source_id": "Source 1", "segment_ids": [1]}],
        "reason": "Deliberately wrong semantic status with structurally valid references."}
    original = copy.deepcopy(payload)
    result = candidate.validate_requirement_coverage_response(json.dumps(payload), value)
    assert result["coverage_checks"]["version_and_time"] == original["coverage_checks"]["version_and_time"]
    assert result["decision"] == decision and payload == original
