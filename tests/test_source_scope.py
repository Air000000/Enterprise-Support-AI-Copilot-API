"""Offline contract checks; canned judgments are NOT real-model quality evidence."""

import copy
import json

from fastapi.testclient import TestClient
import pytest

import main
from experiments.evals.refusal_evidence_sufficiency import ClassifierInput, ContextSource
from experiments.evals.refusal_requirement_coverage import validate_requirement_coverage_response
from rag_runtime import source_scope as scope
from tests.test_hybrid_search import (ASK_BODY, add_document,
    ask_runtime as ask_runtime, runtime as runtime)


def item():
    return ClassifierInput("How do I recover fictional LarchWorker in RuntimeGreen?", (
        ContextSource("Source 1", "PRIVATE_CHUNK", "PRIVATE_DOC",
                      "RuntimeBlue and RuntimeGreen share this recovery path.\r\n\r\nUse console only when sandbox mode is enabled.\r\n"),
        ContextSource("Source 3", "PRIVATE_OTHER", "PRIVATE_DOC2", "A different example.\nPlanned delivery only."),
        ContextSource("Source 4", "PRIVATE_EMPTY", "PRIVATE_DOC3", "\n \r\n")))


def judgment():
    support = dict(status="SUPPORTED", evidence=["Source 1"], reason="Visible runtime compatibility bridge and sandbox prerequisite.")
    return dict(requirements=[dict(requirement="Recover the requested worker", **support)],
        coverage_checks=dict(requested_outcomes=copy.deepcopy(support), applicability=copy.deepcopy(support),
            version_and_time=dict(status="NOT_REQUIRED", evidence=[], reason="No version/time constraint.")),
        unresolved_ambiguities=[])


def test_full_sources_are_the_only_evidence_unit_with_exact_text_and_ids():
    value = item()
    messages = scope.build_messages(value)
    payload = json.loads(messages[1]["content"])
    assert payload == dict(question=value.question, sources=[dict(source_id=s.source_id, content=s.content) for s in value.sources])
    assert "PRIVATE_" not in json.dumps(messages)
    raw = json.dumps(judgment())
    result = scope.validate_response(raw, value)
    assert result["decision"] == "SUFFICIENT" and result["evidence_unit"] == "full_source_chunk"
    assert result["supporting_source_ids"] == ["Source 1"]
    assert [e["quote"] for e in result["requirements"][0]["evidence"]] == [
        "RuntimeBlue and RuntimeGreen share this recovery path.\r\n",
        "Use console only when sandbox mode is enabled.\r\n"]
    assert result["citation_selections"][0] == [{"source_id": "Source 1", "segment_ids": [1, 3]}]
    # Legacy experiments still require their own explicit segment-selection contract.
    with pytest.raises(ValueError):
        validate_requirement_coverage_response(raw, value)


@pytest.mark.parametrize("failure", ["unknown", "legacy_ids", "boolean", "blank_source", "missing_audit",
    "extra_key", "duplicate_key", "bad_status", "empty_supported", "not_required_applicability"])
def test_invalid_source_contract_is_rejected_without_repair(failure):
    data = judgment()
    requirement = data["requirements"][0]
    if failure in ("unknown", "legacy_ids", "boolean", "blank_source", "empty_supported"):
        requirement["evidence"] = {"unknown": ["Source 99"], "legacy_ids": [{"source_id": "Source 1", "segment_ids": [3]}],
            "boolean": [True], "blank_source": ["Source 4"], "empty_supported": []}[failure]
    elif failure == "missing_audit":
        del data["coverage_checks"]["applicability"]
    elif failure == "extra_key":
        data["overall_decision"] = "SUFFICIENT"
    elif failure == "bad_status":
        requirement["status"] = "SUPPORTED|MISSING"
    elif failure == "not_required_applicability":
        data["coverage_checks"]["applicability"] = dict(status="NOT_REQUIRED", evidence=[], reason="Invalid skip.")
    before = copy.deepcopy(data)
    raw = json.dumps(data)
    if failure == "duplicate_key":
        raw = raw.replace('"evidence": ["Source 1"]', '"evidence": ["Source 1"], "evidence": []', 1)
    with pytest.raises(ValueError):
        scope.validate_response(raw, item())
    assert data == before  # Parsed projection never mutates the caller's object.


# Fixed fictional development cases for subsequent model checks, not held-out data.
SCOPE_REGRESSIONS = (
    ("Recover LarchWorker error E7 in RuntimeGreen.", "Only RuntimeBlue's startup path has the E7 workaround.", "applicability", "CONDITIONAL"),
    ("Recover LarchWorker error E7 in RuntimeGreen.", "RuntimeBlue and RuntimeGreen use the same E7 startup path. Clear its sandbox lease.", "applicability", "SUPPORTED"),
    ("Give a sandbox recovery method.", "Clear the sandbox lease. Console mode is another option only when this product supports console mode.", "applicability", "SUPPORTED"),
    ("Can I install the E7 fix now?", "The E7 fix is planned for a future release; no released fix is established here.", "version_and_time", "MISSING"),
)


@pytest.mark.parametrize("question,text,facet,status", SCOPE_REGRESSIONS)
def test_fixture_gate_outcomes_follow_supplied_judgment_without_keyword_rules(question, text, facet, status):
    value = ClassifierInput(question, (ContextSource("Source 1", "c", "d", text),))
    data = judgment()
    data["coverage_checks"][facet] = dict(status=status, evidence=[] if status == "MISSING" else ["Source 1"],
        reason="Supplied fictional judgment: visible compatibility/availability support or unresolved gap.")
    result = scope.validate_response(json.dumps(data), value)
    assert result["decision"] == ("SUFFICIENT" if status == "SUPPORTED" else "INSUFFICIENT")


def source_gate(state):
    state["gate"] = judgment()


def test_api_preserves_full_qualified_sources_scope_audit_and_noncontiguous_ids(ask_runtime):
    engine, _, state, _ = ask_runtime
    for name in ("A", "B", "C"):
        add_document(engine, content=f"{name}: sandbox recovery.\n\nConsole mode requires product support.\nFix is planned, not released.")
    add_document(engine, tenant="foreign", content="PRIVATE_FOREIGN_BODY")
    add_document(engine, category="hr", content="PRIVATE_OTHER_CATEGORY_BODY")
    source_gate(state)
    for check in state["gate"]["requirements"] + list(state["gate"]["coverage_checks"].values()):
        if check["status"] == "SUPPORTED":
            check["evidence"] = ["Source 3"]
    state["gate"]["coverage_checks"]["applicability"]["reason"] = "Visible sandbox path, conditional console option and planned fix."
    state["answer"] = "Use sandbox recovery. Console requires product support; fix remains planned. [Source 3]"
    with TestClient(main.app) as api:
        response = api.post("/rag/ask", json=ASK_BODY | {"refusal_policy": scope.POLICY})
        assert response.status_code == 200, response.text
        assert response.json()["evidence_gate"]["evidence_unit"] == "full_source_chunk"
        assert len(response.json()["sources"]) == 3
        logs = api.get("/agent-ops/retrieval-logs").json()
        assert json.loads(logs[0]["scores_json"])["refusal_policy"] == scope.POLICY
    classifier, generation = [body for _, body in state["events"]]
    assert "PRIVATE_" not in json.dumps(state["events"])
    inputs = json.loads(classifier["messages"][1]["content"])["sources"]
    payload = json.loads(generation["messages"][1]["content"])
    assert len(inputs) == 3 and payload["sources"] == inputs[-1:]
    assert payload["allowed_citation_ids"] == ["Source 3"]
    assert payload["gate_applicability"] == {k: state["gate"]["coverage_checks"]["applicability"][k] for k in ("status", "reason")}
    assert classifier["response_format"] == {"type": "json_object"}
    assert classifier["model"] == "qwen3.5-plus-2026-04-20" and classifier["max_tokens"] == 2048
    assert generation["max_tokens"] == 1024 and generation["enable_thinking"] is False


@pytest.mark.parametrize("status", ["MISSING", "CONDITIONAL", "CONFLICTING"])
def test_unresolved_applicability_never_generates(ask_runtime, status):
    engine, _, state, _ = ask_runtime
    add_document(engine, content=SCOPE_REGRESSIONS[0][1])
    source_gate(state)
    state["gate"]["coverage_checks"]["applicability"]["status"] = status
    with TestClient(main.app) as api:
        response = api.post("/rag/ask", json=ASK_BODY | {"refusal_policy": scope.POLICY})
    assert response.status_code == 200 and response.json()["retrieval_status"] == "refused_insufficient_evidence"
    assert [stage for stage, _ in state["events"]] == ["gate"]


@pytest.mark.parametrize("failure", ["unknown_source", "legacy_ids", "provider", "generation_provider", "unapproved_citation", "truncated"])
def test_new_policy_errors_fail_closed_without_retry(ask_runtime, failure):
    engine, _, state, _ = ask_runtime
    add_document(engine)
    add_document(engine)
    source_gate(state)
    if failure in ("unknown_source", "legacy_ids"):
        state["gate"]["requirements"][0]["evidence"] = ["Source 99"] if failure == "unknown_source" else [{"source_id": "Source 1", "segment_ids": [99]}]
    elif failure in ("provider", "generation_provider"):
        state["error_stage"] = "gate" if failure == "provider" else "generation"
    elif failure == "unapproved_citation":
        state["answer"] = "Use an unapproved method. [Source 2]"
    else:
        state["generation_finish"] = "length"
    with TestClient(main.app) as api:
        response = api.post("/rag/ask", json=ASK_BODY | {"refusal_policy": scope.POLICY})
        assert response.status_code == 503
        audit = api.get("/agent-ops/retrieval-logs").json()[0]
        assert audit["retrieval_status"] == "failed"
        assert "secret-key" not in response.text + audit["error_message"]
    assert [stage for stage, _ in state["events"]] == (["gate"] if failure in ("unknown_source", "legacy_ids", "provider") else ["gate", "generation"])


def test_source_provenance_cannot_prove_semantics_or_force_generated_conditions(ask_runtime):
    value = ClassifierInput("Fix RuntimeGreen.", (ContextSource("Source 1", "c", "d", "RuntimeBlue only; console needs a license."),))
    # Deliberately false but structurally valid judgment: no keyword heuristic overrides it.
    assert scope.validate_response(json.dumps(judgment()), value)["decision"] == "SUFFICIENT"
    engine, _, state, _ = ask_runtime
    add_document(engine, content=value.sources[0].content)
    source_gate(state)
    state["answer"] = "Use the console. [Source 1]"  # Deliberately drops the prerequisite.
    with TestClient(main.app) as api:
        response = api.post("/rag/ask", json=ASK_BODY | {"refusal_policy": scope.POLICY})
    assert response.status_code == 200  # Canned transport proves no semantic reliability claim.
