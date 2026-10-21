"""Zero-call fictional probe plan; no credentials, private data or execution flag."""

import hashlib
import json
from pathlib import Path

from experiments.evals import refusal_single_contract as candidate
from experiments.evals import refusal_version_audit_clarification as baseline
from experiments.evals.refusal_evidence_sufficiency import ClassifierInput, ContextSource

CASES_PATH = Path(__file__).with_name("reports") / "refusal_evidence_sufficiency/v5_4_synthetic_cases.json"
BUILDERS = {"v5_3_1": baseline.build_classifier_messages_v5_3_1_dev,
            "v5_4": candidate.build_classifier_messages_v5_4_dev}


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def planned_calls():
    calls = []
    for index, case in enumerate(json.loads(CASES_PATH.read_text(encoding="utf-8"))):
        value = ClassifierInput(case["question"], tuple(ContextSource(
            source["source_id"], "LOCAL_CHUNK", "LOCAL_DOC", source["content"]) for source in case["sources"]))
        # ponytail: reuse both message builders; explicit allowlist excludes targets/rubrics.
        for arm in tuple(BUILDERS)[::1 if index % 2 == 0 else -1]:
            body = {"model": "qwen3.5-plus-2026-04-20", "temperature": 0,
                "enable_thinking": False, "max_tokens": 2048, "response_format": {"type": "json_object"},
                "messages": BUILDERS[arm](value)}
            calls.append((case["id"], arm, body))
    return calls


def plan():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    calls = planned_calls()
    bounds = [sum(len(m["content"].encode("utf-8")) for m in body["messages"]) + 512 for _, _, body in calls]
    sources = ("refusal_single_contract.py", "refusal_version_audit_clarification.py",
        "refusal_question_scoped_coverage.py", "refusal_coverage_citation_example.py",
        "refusal_requirement_coverage.py", "refusal_citation_selection.py",
        "refusal_structured_evidence.py", "refusal_development_policy.py",
        "refusal_evidence_sufficiency.py", Path(__file__).name)
    result = {"run": "v5_3_1_vs_v5_4_fictional_v1", "status": "FROZEN_PLAN_NOT_AUTHORIZED",
        "region": "Singapore", "model": "qwen3.5-plus-2026-04-20",
        "cases": len(cases), "target_counts": {label: sum(c["expected"]["decision"] == label for c in cases)
            for label in ("SUFFICIENT", "INSUFFICIENT")},
        "cases_sha256": canonical(cases),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_text(
            encoding="utf-8").encode("utf-8")).hexdigest() for name in sources},
        "prompt_sha256": {arm: hashlib.sha256(body["messages"][0]["content"].encode("utf-8")).hexdigest()
            for _, arm, body in calls[:2]},
        "call_order": [[name, arm] for name, arm, _ in calls],
        "request_body_sha256": [canonical(body) for _, _, body in calls],
        "input_byte_plus_framing_bounds": bounds, "max_calls": 24, "max_calls_per_arm": 12,
        "max_output_tokens": 2048, "timeout_seconds": 30, "sdk_retries": 0, "follow_redirects": False,
        "planning_rates_cny_per_million": {"input": 3, "output": 18},
        "fixed_accounting_cny_per_usd": 7.5,
        "planning_cost_cny": round((sum(bounds) * 3 + len(calls) * 2048 * 18) / 1_000_000, 6),
        "proposed_accounting_cap_cny": 2, "invoice_guarantee": False,
        "format_failure_stops": True, "semantic_mismatch_stops": False,
        "network_provider_unknown_charge_stops": True, "automatic_resume": False,
        "provider_calls": 0, "runner_ready": False, "promotion_allowed": False}
    result["plan_sha256"] = canonical(result)
    return result


if __name__ == "__main__":
    print(json.dumps(plan(), ensure_ascii=False, indent=2))
