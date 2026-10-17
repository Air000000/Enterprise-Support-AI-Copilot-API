"""Fixed fictional paired classification check; default is zero-call planning."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx

from experiments.evals import refusal_version_audit_clarification as candidate
from experiments.evals.refusal_evidence_sufficiency import ClassifierInput, ContextSource

MODEL = "qwen3.5-plus-2026-04-20"
MAX_CALLS, MAX_BODY_BYTES, MAX_TOTAL_BYTES = 12, 24576, 196608
CASES = (
    ("fixed_version", "What is ElmCache 3.2's timeout?", "ElmCache 3.2 timeout is 40 seconds.", "SUPPORTED", "SUFFICIENT"),
    ("version_mismatch", "What is ElmCache 4.0's timeout?", "ElmCache 3.2 only: timeout is 40 seconds. No coverage of 4.0.", "MISSING", "INSUFFICIENT"),
    ("compatible_range", "What is ElmCache 3.4's timeout?", "For all ElmCache 3.x releases, timeout is 40 seconds.", "SUPPORTED", "SUFFICIENT"),
    ("outdated", "What is ElmCache's current timeout in 2026?", "Archived 2023 ElmCache timeout: 40 seconds; no later coverage.", "MISSING", "INSUFFICIENT"),
    ("no_version", "How do I open ElmCache's help menu?", "Open Help from ElmCache's toolbar.", "NOT_REQUIRED", "SUFFICIENT"),
    ("unrelated_gap", "What encryption mode does ElmCache 3.2 use?", "ElmCache 3.2 manual: timeout is 40 seconds. This manual does not specify its encryption mode.", "SUPPORTED", "INSUFFICIENT"),
)
BUILDERS = {"v5_3": candidate.prior.build_classifier_messages_v5_3_dev,
            "v5_3_1": candidate.build_classifier_messages_v5_3_1_dev}


def planned_calls():
    calls = []
    for index, (name, question, text, _, _) in enumerate(CASES):
        value = ClassifierInput(question, (ContextSource("Source 1", "LOCAL_CHUNK", "LOCAL_DOC", text),))
        # ponytail: alternate the first arm; fixed order, no randomization framework.
        for arm in tuple(BUILDERS)[::1 if index % 2 == 0 else -1]:
            body = {"model": MODEL, "temperature": 0, "max_tokens": 2048,
                    "response_format": {"type": "json_object"}, "enable_thinking": False,
                    "messages": BUILDERS[arm](value)}
            calls.append((name, arm, value, body))
    return calls


def plan():
    calls = planned_calls()
    result = {"mode": "PLAN_ONLY", "model": MODEL, "max_calls": MAX_CALLS,
            "max_body_bytes": MAX_BODY_BYTES, "max_total_bytes": MAX_TOTAL_BYTES, "budget_cny": 1,
            "timeout_seconds": 30, "retries": 0, "thinking_enabled": False, "cases": CASES,
            "call_order": [[name, arm] for name, arm, _, _ in calls]}
    result["plan_sha256"] = hashlib.sha256(json.dumps(
        {"plan": result, "bodies": [body for _, _, _, body in calls]}, sort_keys=True).encode()).hexdigest()
    return result


def execute(env_file, expected_plan_sha256):
    from dotenv import dotenv_values
    from openai import OpenAI

    frozen = plan()
    if expected_plan_sha256 != frozen["plan_sha256"]:
        raise RuntimeError("Plan changed; do not execute")
    config = dotenv_values(env_file)
    base = config.get("DASHSCOPE_PHASE_B_BASE_URL", "")
    parsed = urlsplit(base)
    host = parsed.hostname or ""
    if (not config.get("DASHSCOPE_PHASE_B_API_KEY") or parsed.scheme != "https"
            or parsed.username or parsed.password or parsed.port or parsed.query or parsed.fragment
            or not (host == "dashscope-intl.aliyuncs.com" or
                    host.startswith("ws-") and host.endswith(".ap-southeast-1.maas.aliyuncs.com"))
            or parsed.path.rstrip("/") != "/compatible-mode/v1"):
        raise RuntimeError("Dedicated Singapore classifier credentials required")
    calls, rows, outcomes = planned_calls(), [], []
    report = {"plan": frozen, "status": "FAILED_NO_RERUN", "calls": rows, "outcomes": outcomes,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    started = None

    def before_send(request):
        nonlocal started
        index, size = len(rows), len(request.content)
        if (index >= MAX_CALLS or request.method != "POST" or
                str(request.url) != base.rstrip("/") + "/chat/completions" or
                json.loads(request.content) != calls[index][3] or size > MAX_BODY_BYTES or
                sum(row["bytes"] for row in rows) + size > MAX_TOTAL_BYTES):
            raise RuntimeError("Unapproved request or exhausted budget")
        rows.append({"case": calls[index][0], "arm": calls[index][1], "bytes": size,
                     "status": None, "transport_phase": "awaiting_headers", "total_tokens": None})
        started = time.perf_counter()

    def after_headers(response):
        row = rows[-1]
        row.update(status=response.status_code, transport_phase="reading_body")
        response.read()
        row["transport_phase"] = "complete"
        if response.status_code == 200:
            payload = response.json()
            for field in ("total_tokens", "prompt_tokens", "completion_tokens"):
                value = payload.get("usage", {}).get(field)
                row[field] = value if type(value) is int and value >= 0 else None
            choice = payload["choices"][0]
            row.update(finish_reason=choice.get("finish_reason"), completion_text=choice["message"].get("content"))

    for name in ("httpx", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        with httpx.Client(transport=httpx.HTTPTransport(retries=0), trust_env=False, follow_redirects=False,
                          event_hooks={"request": [before_send], "response": [after_headers]}) as http:
            with OpenAI(api_key=config["DASHSCOPE_PHASE_B_API_KEY"], base_url=base,
                        max_retries=0, timeout=30, http_client=http) as client:
                for name, arm, value, body in calls:
                    try:
                        response = client.chat.completions.create(
                            **{k: v for k, v in body.items() if k != "enable_thinking"},
                            extra_body={"enable_thinking": False})
                        choice = response.choices[0]
                        raw = choice.message.content
                        if (choice.finish_reason != "stop" or not isinstance(raw, str) or not raw.strip()
                                or choice.message.tool_calls or choice.message.refusal):
                            raise RuntimeError("Incomplete classifier response")
                        gate = candidate.validate_requirement_coverage_response(raw, value)
                        target = next(case for case in CASES if case[0] == name)
                        observed = gate["coverage_checks"]["version_and_time"]
                        outcomes.append({"case": name, "arm": arm, "decision": gate["decision"],
                            "version_and_time": observed, "expected_version_status": target[3],
                            "expected_decision": target[4], "facet_matches": observed["status"] == target[3],
                            "decision_matches": gate["decision"] == target[4]})
                    finally:
                        if rows and started is not None:
                            rows[-1]["latency_ms"] = round((time.perf_counter() - started) * 1000)
        report["status"] = "COMPLETED_DEVELOPMENT_CHECK_NO_PROMOTION"
    except Exception as error:
        from rag_runtime.hybrid_ask import safe_error_details
        report["error_types"] = safe_error_details(error)["error_types"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Paid calls; requires prior human authorization")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args()
    if not args.execute:
        result = plan()
    else:
        if not args.env_file or not args.expected_plan_sha256:
            parser.error("Execution requires --env-file and --expected-plan-sha256")
        try:
            result = execute(args.env_file, args.expected_plan_sha256)
        except Exception as error:
            result = {"status": "PREFLIGHT_FAILED_ZERO_CALLS", "error_type": type(error).__name__}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if "FAILED" in result.get("status", ""):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
