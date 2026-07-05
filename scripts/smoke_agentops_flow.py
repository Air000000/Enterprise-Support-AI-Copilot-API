# scripts/smoke_agentops_flow.py

from __future__ import annotations

import json
import os
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "support")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "support")


def request_json(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    token: str | None = None,
) -> Any:
    url = f"{BASE_URL}{path}"

    data = None
    headers = {
        "Accept": "application/json",
    }
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"

    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(
        url=url,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            if not body:
                return None
            return json.loads(body)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{method} {path} failed with HTTP {exc.code}:\n{body}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Cannot connect to {BASE_URL}. "
            "Start the API server with: uvicorn main:app --reload"
        ) from exc


def assert_condition(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def tool_names(tool_calls: list[dict[str, Any]]) -> set[str]:
    return {
        tool_call["tool_name"]
        for tool_call in tool_calls
    }


def main() -> int:
    print(f"[1/7] Checking API server: {BASE_URL}")
    health = request_json("GET", "/health")
    print("health:", health)

    token_response = request_json(
        "POST",
        "/auth/token",
        {
            "username": AUTH_USERNAME,
            "password": AUTH_PASSWORD,
        },
    )
    token = token_response["access_token"]

    print("[2/7] Creating ticket preview")
    preview = request_json(
        "POST",
        "/agent/ticket/preview",
        {
            "message": "VPN 连不上，重启客户端也没用",
            "category": "it",
        },
        token=token,
    )

    agent_run_id = preview["agent_run_id"]
    approval_request_id = preview["approval_request_id"]
    draft = preview["draft"]

    print("agent_run_id:", agent_run_id)
    print("approval_request_id:", approval_request_id)
    print("should_create_ticket:", preview["should_create_ticket"])

    assert_condition(
        preview["should_create_ticket"] is True,
        "Expected preview.should_create_ticket to be true.",
    )
    assert_condition(
        approval_request_id is not None,
        "Expected preview.approval_request_id to be present.",
    )
    assert_condition(
        isinstance(draft, dict),
        "Expected preview.draft to be a dict.",
    )

    print("[3/7] Inspecting preview tool calls")
    preview_tool_calls = request_json(
        "GET",
        f"/agent-ops/runs/{agent_run_id}/tool-calls",
        token=token,
    )

    preview_tool_names = tool_names(preview_tool_calls)
    print("preview tool calls:", sorted(preview_tool_names))

    assert_condition(
        "search_kb" in preview_tool_names,
        "Expected search_kb tool_call after preview.",
    )
    assert_condition(
        "classify_ticket" in preview_tool_names,
        "Expected classify_ticket tool_call after preview.",
    )

    for tool_call in preview_tool_calls:
        if tool_call["tool_name"] in {"search_kb", "classify_ticket"}:
            assert_condition(
                tool_call["status"] == "success",
                f"Expected {tool_call['tool_name']} status to be success.",
            )

    print("[4/7] Inspecting pending approval request")
    approval_requests = request_json(
        "GET",
        f"/agent-ops/runs/{agent_run_id}/approval-requests",
        token=token,
    )

    matching_approval = next(
        item
        for item in approval_requests
        if item["id"] == approval_request_id
    )

    print("approval status before confirm:", matching_approval["status"])

    assert_condition(
        matching_approval["status"] == "pending",
        "Expected approval_request status to be pending before confirm.",
    )

    print("[5/7] Confirming ticket creation")
    confirm = request_json(
        "POST",
        "/agent/ticket/confirm",
        {
            "agent_run_id": agent_run_id,
            "approval_request_id": approval_request_id,
            "draft": draft,
        },
        token=token,
    )

    ticket = confirm["ticket"]
    print("ticket_id:", ticket["id"])
    print("ticket_status:", ticket["status"])
    print("tool_call_id:", confirm["tool_call_id"])

    assert_condition(
        ticket["status"] == "open",
        "Expected created ticket status to be open.",
    )

    print("[6/7] Inspecting full tool call chain")
    full_tool_calls = request_json(
        "GET",
        f"/agent-ops/runs/{agent_run_id}/tool-calls",
        token=token,
    )

    full_tool_names = tool_names(full_tool_calls)
    print("full tool calls:", sorted(full_tool_names))

    expected_tools = {
        "search_kb",
        "classify_ticket",
        "create_ticket",
    }

    assert_condition(
        expected_tools.issubset(full_tool_names),
        f"Expected tool calls {expected_tools}, got {full_tool_names}.",
    )

    for tool_call in full_tool_calls:
        if tool_call["tool_name"] in expected_tools:
            assert_condition(
                tool_call["status"] == "success",
                f"Expected {tool_call['tool_name']} status to be success.",
            )

    approval_requests_after_confirm = request_json(
        "GET",
        f"/agent-ops/runs/{agent_run_id}/approval-requests",
        token=token,
    )

    matching_approval_after_confirm = next(
        item
        for item in approval_requests_after_confirm
        if item["id"] == approval_request_id
    )

    print(
        "approval status after confirm:",
        matching_approval_after_confirm["status"],
    )

    assert_condition(
        matching_approval_after_confirm["status"] == "approved",
        "Expected approval_request status to be approved after confirm.",
    )

    print("[7/7] Inspecting AgentOps metrics")
    metrics = request_json("GET", "/agent-ops/metrics/summary", token=token)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    assert_condition(
        metrics["total_agent_runs"] >= 1,
        "Expected total_agent_runs >= 1.",
    )
    assert_condition(
        metrics["total_tool_calls"] >= 3,
        "Expected total_tool_calls >= 3.",
    )
    assert_condition(
        metrics["successful_tool_calls"] >= 3,
        "Expected successful_tool_calls >= 3.",
    )
    assert_condition(
        metrics["approved_approval_requests"] >= 1,
        "Expected approved_approval_requests >= 1.",
    )

    print("\nSmoke test passed.")
    print(
        "Validated: preview -> search_kb/classify_ticket -> approval "
        "-> confirm -> create_ticket -> metrics"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\nSmoke test failed.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
