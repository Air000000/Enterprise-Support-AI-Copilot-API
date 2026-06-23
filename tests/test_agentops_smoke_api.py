# tests/test_agentops_smoke_api.py

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def make_search_result(
    document_id: str = "doc_vpn_guide",
    chunk_id: str = "doc_vpn_guide_chunk_1",
    title: str = "vpn_guide",
    category: str = "it",
    distance: float = 0.52,
    content: str = "VPN 故障排查说明。",
):
    return SimpleNamespace(
        document_id=document_id,
        chunk_id=chunk_id,
        title=title,
        source_path=f"experiments/docs/{category}/{title}.md",
        distance=distance,
        content=content,
        category=category,
    )


def get_tool_names(tool_calls: list[dict]) -> set[str]:
    return {tool_call["tool_name"] for tool_call in tool_calls}


def get_tool_call_by_name(tool_calls: list[dict], tool_name: str) -> dict:
    for tool_call in tool_calls:
        if tool_call["tool_name"] == tool_name:
            return tool_call

    raise AssertionError(f"tool_call not found: {tool_name}")


def test_ticket_agent_preview_confirm_agentops_smoke_flow(monkeypatch):
    """
    端到端 API smoke test。

    验证完整链路：
    preview
    -> search_kb / classify_ticket tool_calls
    -> pending approval_request
    -> confirm
    -> create_ticket tool_call
    -> approved approval_request
    -> metrics summary
    """

    def fake_search_chroma(
        query: str,
        top_k: int,
        tenant_id: str,
        category: str | None,
    ):
        return [
            make_search_result(
                document_id="doc_vpn_guide",
                chunk_id="doc_vpn_guide_chunk_1",
                title="vpn_guide",
                category="it",
                distance=0.52,
                content="VPN 故障排查说明。",
            )
        ]

    monkeypatch.setattr(
        "services.ticket_agent_service.search_chroma",
        fake_search_chroma,
    )

    preview_response = client.post(
        "/agent/ticket/preview",
        json={
            "message": "VPN 连不上，重启客户端也没用",
            "category": "it",
        },
    )

    assert preview_response.status_code == 200

    preview = preview_response.json()

    agent_run_id = preview["agent_run_id"]
    approval_request_id = preview["approval_request_id"]
    draft = preview["draft"]

    assert preview["should_create_ticket"] is True
    assert approval_request_id is not None
    assert draft is not None
    assert draft["category"] == "it"
    assert draft["priority"] == "high"
    assert preview["sources"][0]["document_id"] == "doc_vpn_guide"

    preview_tool_calls_response = client.get(
        f"/agent-ops/runs/{agent_run_id}/tool-calls"
    )

    assert preview_tool_calls_response.status_code == 200

    preview_tool_calls = preview_tool_calls_response.json()
    preview_tool_names = get_tool_names(preview_tool_calls)

    assert "search_kb" in preview_tool_names
    assert "classify_ticket" in preview_tool_names
    assert "create_ticket" not in preview_tool_names

    search_kb_tool_call = get_tool_call_by_name(
        preview_tool_calls,
        "search_kb",
    )
    classify_ticket_tool_call = get_tool_call_by_name(
        preview_tool_calls,
        "classify_ticket",
    )

    assert search_kb_tool_call["status"] == "success"
    assert classify_ticket_tool_call["status"] == "success"
    assert search_kb_tool_call["error_message"] is None
    assert classify_ticket_tool_call["error_message"] is None

    approval_requests_response = client.get(
        f"/agent-ops/runs/{agent_run_id}/approval-requests"
    )

    assert approval_requests_response.status_code == 200

    approval_requests = approval_requests_response.json()

    matching_approval = next(
        approval_request
        for approval_request in approval_requests
        if approval_request["id"] == approval_request_id
    )

    assert matching_approval["status"] == "pending"

    confirm_response = client.post(
        "/agent/ticket/confirm",
        json={
            "agent_run_id": agent_run_id,
            "approval_request_id": approval_request_id,
            "draft": draft,
        },
    )

    assert confirm_response.status_code == 201

    confirm = confirm_response.json()

    assert confirm["agent_run_id"] == agent_run_id
    assert confirm["approval_request_id"] == approval_request_id
    assert confirm["tool_call_id"] is not None
    assert confirm["ticket"]["id"] is not None
    assert confirm["ticket"]["status"] == "open"
    assert confirm["ticket"]["category"] == "it"
    assert confirm["ticket"]["priority"] == "high"

    full_tool_calls_response = client.get(
        f"/agent-ops/runs/{agent_run_id}/tool-calls"
    )

    assert full_tool_calls_response.status_code == 200

    full_tool_calls = full_tool_calls_response.json()
    full_tool_names = get_tool_names(full_tool_calls)

    assert {"search_kb", "classify_ticket", "create_ticket"}.issubset(
        full_tool_names
    )

    create_ticket_tool_call = get_tool_call_by_name(
        full_tool_calls,
        "create_ticket",
    )

    assert create_ticket_tool_call["status"] == "success"
    assert create_ticket_tool_call["error_message"] is None

    approval_requests_after_confirm_response = client.get(
        f"/agent-ops/runs/{agent_run_id}/approval-requests"
    )

    assert approval_requests_after_confirm_response.status_code == 200

    approval_requests_after_confirm = (
        approval_requests_after_confirm_response.json()
    )

    matching_approval_after_confirm = next(
        approval_request
        for approval_request in approval_requests_after_confirm
        if approval_request["id"] == approval_request_id
    )

    assert matching_approval_after_confirm["status"] == "approved"

    metrics_response = client.get("/agent-ops/metrics/summary")

    assert metrics_response.status_code == 200

    metrics = metrics_response.json()

    assert metrics["total_agent_runs"] >= 1
    assert metrics["completed_agent_runs"] >= 1
    assert metrics["total_tool_calls"] >= 3
    assert metrics["successful_tool_calls"] >= 3
    assert metrics["total_approval_requests"] >= 1
    assert metrics["approved_approval_requests"] >= 1