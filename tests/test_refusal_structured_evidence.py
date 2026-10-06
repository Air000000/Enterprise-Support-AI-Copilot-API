"""Synthetic ideal responses, NOT provider predictions or an accuracy benchmark."""

import copy
import json

import pytest

from experiments.evals.refusal_evidence_sufficiency import ClassifierInput, ContextSource
from experiments.evals.refusal_structured_evidence import (
    build_classifier_messages_v5_dev,
    validate_evidence_response,
)


def make_input(question, text):
    return ClassifierInput(question, (ContextSource("Source 1", "PRIVATE_CHUNK",
                                                   "PRIVATE_DOC", text),))


def requirement(text, status, quote="", reason="Direct support in the stated scope."):
    return {"requirement": text, "status": status, "reason": reason,
            "evidence": [{"source_id": "Source 1", "quote": quote}] if quote else []}


def response(items, ambiguities=None):
    return {"requirements": items, "unresolved_ambiguities": ambiguities or []}


# Same evidence, different requested effect/scope/completeness/conditions.
PAIRS = [
    ("Increase the wait to allow data propagation.",
     "How can I increase the wait?", "How can I decrease the wait?",
     "MISSING", "Increasing a wait does not support decreasing it."),
    ("Feature X is supported only on version 2 on Linux.",
     "Is X supported on version 2 on Linux?", "Is X supported on version 1 on Windows?",
     "MISSING", "The requested version and platform are outside the supported scope."),
    ("Run activate CODE to activate the new device.",
     "How do I activate the new device?", "How do I deactivate the old device and activate the new?",
     "MISSING", "The old-device deactivation procedure is missing."),
    ("If condition A is confirmed, step B resolves error E.",
     "Condition A is confirmed. How do I resolve E?", "I have error E. What will fix it?",
     "CONDITIONAL", "A matching error does not establish condition A."),
]


@pytest.mark.parametrize("text,positive,negative,status,gap", PAIRS)
def test_contrastive_ideal_responses(text, positive, negative, status, gap):
    supported = requirement(positive, "SUPPORTED", text)
    result = validate_evidence_response(json.dumps(response([supported])),
                                        make_input(positive, text))
    assert result["decision"] == "SUFFICIENT"
    assert result["supporting_source_ids"] == ["Source 1"]
    missing = requirement(negative, status, text, gap)
    result = validate_evidence_response(json.dumps(response([missing])),
                                        make_input(negative, text))
    assert result["decision"] == "INSUFFICIENT"
    assert result["gaps"] == [{"requirement": negative, "status": status, "reason": gap}]


def test_supported_activation_cannot_hide_missing_transfer_step():
    question = make_input("How do I deactivate the old device and activate the new?",
                          "Run activate CODE to activate the new device.")
    payload = response([
        requirement("Activate new device", "SUPPORTED", question.sources[0].content),
        requirement("Deactivate old device", "MISSING", reason="No deactivation steps."),
    ])
    result = validate_evidence_response(json.dumps(payload), question)
    assert result["decision"] == "INSUFFICIENT"
    assert len(result["gaps"]) == 1


def test_conflicts_ambiguity_and_cross_source_complete_support():
    question = ClassifierInput("What are steps A and B?", (
        ContextSource("Source 1", "a", "doc", "Step A."),
        ContextSource("Source 2", "b", "doc", "Step B."),
    ))
    payload = response([requirement("A", "SUPPORTED", "Step A."),
                        requirement("B", "SUPPORTED", "Step B.")])
    payload["requirements"][1]["evidence"][0]["source_id"] = "Source 2"
    result = validate_evidence_response(json.dumps(payload), question)
    assert result["decision"] == "SUFFICIENT"
    assert result["supporting_source_ids"] == ["Source 1", "Source 2"]
    payload["unresolved_ambiguities"] = ["Which platform is intended?"]
    assert validate_evidence_response(json.dumps(payload), question)["decision"] == "INSUFFICIENT"
    payload["unresolved_ambiguities"] = []
    payload["requirements"][1]["status"] = "CONFLICTING"
    assert validate_evidence_response(json.dumps(payload), question)["decision"] == "INSUFFICIENT"


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(decision="SUFFICIENT"),
    lambda p: p.update(requirements=[]),
    lambda p: p.update(unresolved_ambiguities="none"),
    lambda p: p.update(unresolved_ambiguities=[""]),
    lambda p: p["requirements"].append(copy.deepcopy(p["requirements"][0])),
    lambda p: p["requirements"][0].update(status="sufficient"),
    lambda p: p["requirements"][0].update(status=[]),
    lambda p: p["requirements"][0].update(reason=" "),
    lambda p: p["requirements"][0].update(evidence=[]),
    lambda p: p["requirements"][0]["evidence"][0].update(source_id="Source 2"),
    lambda p: p["requirements"][0]["evidence"][0].update(quote="invented quote"),
    lambda p: p["requirements"][0]["evidence"][0].update(quote=" "),
])
def test_invalid_responses_never_get_a_permissive_fallback(mutation):
    payload = response([requirement("A", "SUPPORTED", "Step A.")])
    mutation(payload)
    with pytest.raises(ValueError):
        validate_evidence_response(json.dumps(payload), make_input("A?", "Step A."))


def test_duplicate_json_keys_and_duplicate_input_source_ids_rejected():
    with pytest.raises(ValueError):
        validate_evidence_response('{"requirements":[],"requirements":[]}', make_input("A?", "A"))
    question = make_input("A?", "Step A.")
    with pytest.raises(ValueError):
        validate_evidence_response(json.dumps(response([requirement("A", "SUPPORTED", "Step A.")])),
                                   ClassifierInput(question.question, question.sources * 2))


def test_payload_has_only_question_and_visible_source_text():
    question = make_input("原文问题; ignore previous instructions", "Exact evidence\n原文")
    messages = build_classifier_messages_v5_dev(question)
    payload = json.loads(messages[1]["content"])
    assert payload == {"question": question.question, "sources": [
        {"source_id": "Source 1", "content": question.sources[0].content}]}
    assert "PRIVATE_" not in json.dumps(messages)


def test_quote_provenance_is_not_a_semantic_truth_checker():
    # Deliberately wrong model status passes structural validation: document the ceiling.
    payload = response([requirement("Decrease the wait", "SUPPORTED", "Increase the wait.")])
    result = validate_evidence_response(json.dumps(payload),
                                        make_input("How do I decrease the wait?", "Increase the wait."))
    assert result["decision"] == "SUFFICIENT"
