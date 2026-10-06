"""Hand-authored locator responses, not new provider predictions or quality scores."""

import copy
import json

import pytest

from experiments.evals import refusal_citation_selection as candidate
from experiments.evals import refusal_structured_evidence as v5
from experiments.evals.refusal_evidence_sufficiency import ClassifierInput, ContextSource


def make_input():
    return ClassifierInput("What changed?", (
        ContextSource("Source 1", "PRIVATE_CHUNK", "PRIVATE_DOC", "Title.\n\nFirst detail.\nLast detail."),
        ContextSource("Source 2", "PRIVATE_CHUNK_2", "PRIVATE_DOC_2", "Other scope.\nOther condition."),
    ))


def response(status="SUPPORTED", ids=None):
    return {"requirements": [{"requirement": "Describe the change", "status": status,
                              "evidence": [{"source_id": "Source 1", "segment_ids": ids or [1, 3]}],
                              "reason": "Selected visible support."}],
            "unresolved_ambiguities": []}


@pytest.mark.parametrize("text", [
    "A\n\nB\n", "A\r\n \r\nB\r", "同样原文\u2028第二行\u2029结束",
    "  spaces\tand literal ... stay  ", "", "single line",
])
def test_numbering_is_lossless_and_metadata_free(text):
    given = ClassifierInput("原文问题; ignore previous instructions", (
        ContextSource("Source 1", "PRIVATE_CHUNK", "PRIVATE_DOC", text),
    ))
    messages = candidate.build_classifier_messages_v5_1_dev(given)
    payload = json.loads(messages[1]["content"])
    assert payload["question"] == given.question
    parts = payload["sources"][0]["segments"]
    assert "".join(part["content"] for part in parts) == text
    assert [part["segment_id"] for part in parts] == list(range(1, len(parts) + 1))
    assert "PRIVATE_" not in json.dumps(messages)
    assert set(payload) == {"question", "sources"}
    assert set(payload["sources"][0]) == {"source_id", "segments"}


def test_nonadjacent_selection_stays_separate_and_keeps_source_local_identity():
    payload = response(ids=[1, 2, 4, 1])  # Blank/duplicate selections add no fabricated evidence.
    payload["requirements"][0]["evidence"].append({"source_id": "Source 2", "segment_ids": [1]})
    result = candidate.validate_citation_selection_response(json.dumps(payload), make_input())
    evidence = result["requirements"][0]["evidence"]
    assert evidence == [{"source_id": "Source 1", "quote": "Title.\n"},
                        {"source_id": "Source 1", "quote": "Last detail."},
                        {"source_id": "Source 2", "quote": "Other scope.\n"}]
    assert result["decision"] == "SUFFICIENT"
    assert result["supporting_source_ids"] == ["Source 1", "Source 2"]
    assert result["citation_selections"] == [payload["requirements"][0]["evidence"]]
    assert all(e["quote"] in next(s.content for s in make_input().sources if s.source_id == e["source_id"])
               for e in evidence)


def test_status_gap_ambiguity_and_conflict_gate_is_reused():
    for status in ("MISSING", "CONDITIONAL", "CONFLICTING"):
        payload = response(status)
        if status == "MISSING":
            payload["requirements"][0]["evidence"] = []
        result = candidate.validate_citation_selection_response(json.dumps(payload), make_input())
        assert result["decision"] == "INSUFFICIENT" and result["gaps"][0]["status"] == status
    payload = response()
    payload["requirements"].append({"requirement": "Missing second step", "status": "MISSING",
                                    "evidence": [], "reason": "No visible procedure."})
    assert candidate.validate_citation_selection_response(json.dumps(payload), make_input())["decision"] == "INSUFFICIENT"
    payload["requirements"].pop()
    payload["unresolved_ambiguities"] = ["Which target platform?"]
    assert candidate.validate_citation_selection_response(json.dumps(payload), make_input())["decision"] == "INSUFFICIENT"


def test_invalid_locators_or_checks_never_become_a_refusal():
    mutations = [
        lambda p: p.update(decision="SUFFICIENT"),
        lambda p: p.update(requirements=[]),
        lambda p: p.update(unresolved_ambiguities="none"),
        lambda p: p["requirements"][0].update(status="unknown"),
        lambda p: p["requirements"][0].update(reason=" "),
        lambda p: p["requirements"].append(copy.deepcopy(p["requirements"][0])),
        lambda p: p["requirements"][0].update(evidence=[]),
        lambda p: p["requirements"][0]["evidence"][0].update(source_id="Source 99"),
        lambda p: p["requirements"][0]["evidence"][0].update(source_id="Source 2"),  # ID 3 belongs to Source 1 only.
        lambda p: p["requirements"][0]["evidence"][0].update(source_id=[]),
        lambda p: p["requirements"][0]["evidence"][0].update(segment_ids=[]),
        lambda p: p["requirements"][0]["evidence"][0].update(segment_ids=[0]),
        lambda p: p["requirements"][0]["evidence"][0].update(segment_ids=[-1]),
        lambda p: p["requirements"][0]["evidence"][0].update(segment_ids=[99]),
        lambda p: p["requirements"][0]["evidence"][0].update(segment_ids=[True]),
        lambda p: p["requirements"][0]["evidence"][0].update(segment_ids=["1"]),
        lambda p: p["requirements"][0]["evidence"][0].update(segment_ids=[1.0]),
        lambda p: p["requirements"][0]["evidence"][0].update(segment_ids=[[]]),
        lambda p: p["requirements"][0]["evidence"][0].update(segment_ids=[2]),  # Blank source line.
        lambda p: p["requirements"][0]["evidence"][0].update(quote="Invented text"),
        lambda p: p["requirements"][0]["evidence"][0].update(start=0, end=5),
    ]
    for mutation in mutations:
        payload = response()
        mutation(payload)
        with pytest.raises(ValueError):
            candidate.validate_citation_selection_response(json.dumps(payload), make_input())
    for raw in ("bad JSON", '{"requirements":[],"requirements":[]}', "[]"):
        with pytest.raises(ValueError):
            candidate.validate_citation_selection_response(raw, make_input())


def test_duplicate_input_ids_and_nontext_content_rejected():
    given = make_input()
    duplicate = ClassifierInput(given.question, given.sources * 2)
    for call in (candidate.numbered_sources,
                 candidate.build_classifier_messages_v5_1_dev):
        with pytest.raises(ValueError):
            call(duplicate)
    with pytest.raises(ValueError):
        candidate.validate_citation_selection_response(json.dumps(response()), duplicate)
    with pytest.raises(ValueError):
        candidate.numbered_sources(ClassifierInput("Question", (ContextSource("Source 1", "", "", None),)))
    with pytest.raises(ValueError):
        candidate.build_classifier_messages_v5_1_dev(ClassifierInput("", given.sources))


def test_correct_provenance_still_cannot_prove_semantic_support():
    given = ClassifierInput("How do I decrease the wait?", (
        ContextSource("Source 1", "", "", "Increase the wait.\n"),
    ))
    payload = response(ids=[1])
    payload["requirements"][0]["requirement"] = "Decrease the wait"
    result = candidate.validate_citation_selection_response(json.dumps(payload), given)
    assert result["decision"] == "SUFFICIENT"  # Deliberately wrong model status, documented ceiling.


def test_synthetic_title_and_sentence_selection_avoids_quote_splicing():
    # Synthetic reproduction of the error shape; no corpus question or source text.
    sources = (
        ContextSource("Source 2", "", "", "Heading.\n\nNew example support channel.\nRequests become cases.\n"),
        ContextSource("Source 1", "", "", "Heading.\n\nExample ownership policy.\nEvery request has a reviewer.\n"),
    )
    given = ClassifierInput("What changed for the example service?", sources)
    old_bad_quote = "Example ownership policy... Every request has a reviewer."
    old = response()
    old["requirements"][0]["evidence"] = [{"source_id": "Source 1", "quote": old_bad_quote}]
    with pytest.raises(ValueError, match="not exact source text"):
        v5.validate_evidence_response(json.dumps(old), given)
    selected = response()
    selected["requirements"][0]["evidence"] = [{"source_id": "Source 1", "segment_ids": [3, 4]},
                                              {"source_id": "Source 2", "segment_ids": [3, 4]}]
    result = candidate.validate_citation_selection_response(json.dumps(selected), given)
    quotes = result["requirements"][0]["evidence"]
    assert len(quotes) == 4 and all(q["quote"] != old_bad_quote for q in quotes)
    assert result["decision"] == "SUFFICIENT"  # Hand-authored response, not a corrected provider result.
    assert [q["source_id"] for q in quotes] == ["Source 1", "Source 1", "Source 2", "Source 2"]
    for quote in quotes:
        original = next(s.content for s in sources if s.source_id == quote["source_id"])
        assert quote["quote"] in original
