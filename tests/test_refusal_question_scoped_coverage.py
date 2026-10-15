"""Hand-authored fictional traces test wiring/gates, NOT model semantic behavior."""

import copy
import json

import pytest

from experiments.evals import refusal_question_scoped_coverage as candidate
from experiments.evals.refusal_evidence_sufficiency import ClassifierInput, ContextSource


def context(question, *texts):
    return ClassifierInput(question, tuple(ContextSource(f"Source {i}", "PRIVATE_CHUNK", "PRIVATE_DOC", text)
                                           for i, text in enumerate(texts, 1)))


def ref(source, *segments):
    return {"source_id": source, "segment_ids": list(segments or (1,))}


def response(status="SUPPORTED", evidence=None, version=False):
    evidence = [ref("Source 1")] if evidence is None else evidence
    support = {"status": "SUPPORTED", "evidence": copy.deepcopy(evidence), "reason": "Visible scope and evidence."}
    outcome = {"status": status, "evidence": copy.deepcopy(evidence), "reason": "Requested claim or material gap."}
    return {"requirements": [{"requirement": "Answer the requested claim within its stated scope", **copy.deepcopy(outcome)}],
            "coverage_checks": {"requested_outcomes": outcome, "applicability": support,
                                "version_and_time": copy.deepcopy(support) if version else {
                                    "status": "NOT_REQUIRED", "evidence": [], "reason": "No material version/time constraint."}},
            "unresolved_ambiguities": []}


def cases():
    platform = context("How do I search for a square bracket in CobaltDocs?",
        "CobaltDocs escapes a square bracket by prefixing it with a backslash.\n",
        "DeltaIndex percent-sign search needs escape mode; its setting is not reported.\n")
    fixes = context("Which upgrades fix CobaltUI7's missing line breaks?",
        "For CobaltUI7's missing line-break defect, use client7.2+ and plugin3.4+.\n",
        "CobaltUI7's unrelated long-label wrapping defect uses client7.1/plugin3.3; it is not the line-break defect.\n")
    source_texts = ("K17 is a security update for CobaltSvc.\n",
        "In CobaltSvc after K17, a global loader with the guard unset causes linker failures; guard=true isolates the conflicting module.\n"
        "Collect the loader trace and guard value before choosing this remedy.\n")
    unknown = context("K17 caused my CobaltSvc linker failure. What explains this incident and what fixes it? Loader and guard state were not reported.", *source_texts)
    known = context("After K17 CobaltSvc failed during linking with the documented global loader and guard unset. Explain this mechanism and remedy.", *source_texts)
    general = context("After K17 CobaltSvc failed during linking. Which possible cause and diagnostic checks do these notes describe?", *source_texts)
    conflict = context("What is the current CobaltDocs cache lifetime as of 2026-10-15?",
        "CobaltDocs current policy as of 2026-10-15: cache lifetime30 seconds; no precedence over the other policy.\n",
        "CobaltDocs current policy as of 2026-10-15: cache lifetime90 seconds; equal authority to the other policy.\n")
    gap = response("CONDITIONAL", [ref("Source 2", 1, 2)], version=True)
    gap["coverage_checks"]["applicability"].update(
        status="CONDITIONAL", reason="Observed loader/guard prerequisites are missing; topic overlap is not incident proof.")
    return [(platform, response(), "SUFFICIENT"), (fixes, response(version=True), "SUFFICIENT"),
            (unknown, gap, "INSUFFICIENT"),
            (known, response(evidence=[ref("Source 2", 1, 2)], version=True), "SUFFICIENT"),
            (general, response(evidence=[ref("Source 2", 1, 2)], version=True), "SUFFICIENT"),
            (conflict, response("CONFLICTING", [ref("Source 1"), ref("Source 2")], version=True), "INSUFFICIENT")]


@pytest.mark.parametrize("value,payload,decision", cases())
def test_policy_wiring_and_handwritten_boundary_traces(value, payload, decision):
    old = candidate.prior.build_classifier_messages_v5_2_1_dev(value)
    new = candidate.build_classifier_messages_v5_3_dev(value)
    assert old[1:] == new[1:] and old[0] != new[0]
    assert new[0]["content"].startswith(candidate.prior.CLASSIFIER_SYSTEM_PROMPT_V5_2_1_DEV)
    assert "PRIVATE_" not in json.dumps(new)
    assert candidate.validate_requirement_coverage_response is candidate.prior.validate_requirement_coverage_response
    result = candidate.validate_requirement_coverage_response(json.dumps(payload), value)
    assert result["decision"] == decision
    selected = result["requirements"][0]["evidence"][0]
    original = next(s.content for s in value.sources if s.source_id == selected["source_id"])
    assert selected["quote"] in original


def test_no_posthoc_requirement_removal_evidence_repair_or_semantic_oracle():
    value, payload, _ = cases()[0]
    payload["requirements"].append({"requirement": "Reconcile the optional DeltaIndex parser configuration",
        "status": "CONDITIONAL", "evidence": [ref("Source 2")], "reason": "Invented extra applicability demand."})
    assert candidate.validate_requirement_coverage_response(json.dumps(payload), value)["decision"] == "INSUFFICIENT"
    assert len(payload["requirements"]) == 2  # Gate does not silently delete a model's extra requirement.
    payload = response()
    payload["coverage_checks"]["applicability"]["evidence"] = []
    with pytest.raises(ValueError):
        candidate.validate_requirement_coverage_response(json.dumps(payload), value)
    unknown, _, _ = cases()[2]
    wrong = response(evidence=[ref("Source 1")], version=True)
    # A valid locator to a related bulletin still cannot mechanically prove incident causality.
    assert candidate.validate_requirement_coverage_response(json.dumps(wrong), unknown)["decision"] == "SUFFICIENT"
