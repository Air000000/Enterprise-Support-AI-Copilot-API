"""Synthetic example/schema regression only; no provider accuracy measurement."""

import copy
import json

import pytest

from experiments.evals import refusal_coverage_citation_example as candidate
from experiments.evals import refusal_requirement_coverage as prior
from experiments.evals.refusal_evidence_sufficiency import ClassifierInput, ContextSource


def given():
    return ClassifierInput("What is the current Atlas release 14 cache timeout?", (
        ContextSource("Source 1", "private_chunk_1", "private_doc_1",
                      "Atlas release 14 current cache timeout is 30 seconds.\n"),
        ContextSource("Source 2", "private_chunk_2", "private_doc_2",
                      "Atlas release 14 current cache timeout is 90 seconds.\n"),
    ))


def response(outcome):
    support = candidate._SUPPORTED_AUDIT
    return {
        "requirements": [{"requirement": "Establish current release 14 timeout", **copy.deepcopy(outcome)}],
        "coverage_checks": {"requested_outcomes": copy.deepcopy(outcome),
                            "applicability": copy.deepcopy(support),
                            "version_and_time": copy.deepcopy(support)},
        "unresolved_ambiguities": [],
    }


def test_only_system_examples_change_and_validator_is_the_same_callable():
    original = prior.build_classifier_messages_v5_2_dev(given())
    revised = candidate.build_classifier_messages_v5_2_1_dev(given())
    assert original[1:] == revised[1:]
    assert original[0] != revised[0]
    assert "private_" not in json.dumps(revised)
    assert candidate._EMPTY_AUDIT not in revised[0]["content"]
    assert prior.CLASSIFIER_SYSTEM_PROMPT_V5_2_DEV.count(candidate._EMPTY_AUDIT) == 3
    assert candidate.validate_requirement_coverage_response is prior.validate_requirement_coverage_response


@pytest.mark.parametrize("example,decision", [
    (candidate._SUPPORTED_AUDIT, "SUFFICIENT"),
    (candidate._CONFLICTING_AUDIT, "INSUFFICIENT"),
])
def test_prompt_examples_pass_existing_gate_in_their_own_fictional_context(example, decision):
    assert json.dumps(example) in candidate.CLASSIFIER_SYSTEM_PROMPT_V5_2_1_DEV
    context = given()
    if decision == "SUFFICIENT":
        context = ClassifierInput(context.question, context.sources[:1])
    result = candidate.validate_requirement_coverage_response(json.dumps(response(example)), context)
    assert result["decision"] == decision
    assert result["requirements"][0]["evidence"][0]["quote"] == context.sources[0].content


@pytest.mark.parametrize("status", ["SUPPORTED", "CONFLICTING"])
@pytest.mark.parametrize("evidence", [
    [], [{"source_id": "Source 99", "segment_ids": [1]}],
    [{"source_id": "Source 1", "segment_ids": [True]}],
    [{"source_id": "Source 1", "segment_ids": [2]}],
])
def test_invalid_audit_references_still_fail_without_repair(status, evidence):
    payload = response(candidate._SUPPORTED_AUDIT)
    payload["coverage_checks"]["applicability"].update(status=status, evidence=evidence)
    assert payload["requirements"][0]["evidence"] == candidate._SUPPORTED_AUDIT["evidence"]
    raw = json.dumps(payload)
    with pytest.raises(ValueError):
        candidate.validate_requirement_coverage_response(raw, given())
    assert json.dumps(payload) == raw


def test_not_required_remains_restricted_to_version_and_time():
    payload = response(candidate._SUPPORTED_AUDIT)
    no_constraint = {"status": "NOT_REQUIRED", "evidence": [], "reason": "No material time constraint."}
    payload["coverage_checks"]["applicability"] = no_constraint
    with pytest.raises(ValueError, match="invalid NOT_REQUIRED"):
        candidate.validate_requirement_coverage_response(json.dumps(payload), given())
    payload["coverage_checks"]["applicability"] = copy.deepcopy(candidate._SUPPORTED_AUDIT)
    payload["coverage_checks"]["version_and_time"] = no_constraint
    # False semantic NOT_REQUIRED still passes mechanically: the fix does not adjudicate meaning.
    assert candidate.validate_requirement_coverage_response(json.dumps(payload), given())["decision"] == "SUFFICIENT"
