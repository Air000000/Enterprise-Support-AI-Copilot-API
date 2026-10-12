# tests/test_agentops_smoke_api.py

from __future__ import annotations

import json
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, local

from fastapi import HTTPException
from fastapi.testclient import TestClient

from auth import CurrentUser, get_current_user
from main import app
from models.agent_ops import AgentRun, ApprovalRequest, ToolCall
from models.ticket import Ticket
from schemas.agent_ops import ApprovalRequestUpdate
from schemas.agent_ticket import TicketAgentConfirmRequest, TicketAgentPreviewRequest

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

import services.agent_ops_service as agent_ops_service
import services.ticket_service as ticket_service
import services.ticket_agent_service as ticket_agent_service

@pytest.fixture(autouse=True)
def default_auth_user():
    user = CurrentUser(
        sub="support",
        user_id="user_demo",
        tenant_id="tenant_demo",
        role="support",
    )
    app.dependency_overrides[get_current_user] = lambda: user
    yield
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def smoke_test_engine(tmp_path, monkeypatch):
    db_path = tmp_path / "agentops_smoke_test.db"

    test_engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )

    SQLModel.metadata.create_all(test_engine)

    monkeypatch.setattr(agent_ops_service, "engine", test_engine)
    monkeypatch.setattr(ticket_service, "engine", test_engine)

    return test_engine


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


@pytest.mark.parametrize(
    ("username", "tenant_id", "user_id"),
    [
        ("support", "tenant_demo", "support_demo"),
        ("support_other", "tenant_other", "support_other"),
    ],
)
def test_ticket_agent_preview_confirm_agentops_smoke_flow(
    client,
    monkeypatch,
    smoke_test_engine,
    username,
    tenant_id,
    user_id,
):
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

    app.dependency_overrides.pop(get_current_user, None)
    login_response = client.post(
        "/auth/token",
        json={"username": username, "password": "support"},
    )
    assert login_response.status_code == 200
    client.headers["Authorization"] = (
        f"Bearer {login_response.json()['access_token']}"
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

    with Session(smoke_test_engine) as session:
        agent_run = session.get(AgentRun, agent_run_id)
        assert agent_run is not None
        assert agent_run.user_id == user_id
        assert agent_run.tenant_id == tenant_id
        assert session.exec(select(Ticket)).first() is None

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
    assert confirm["ticket"]["created_by"] == user_id
    assert confirm["ticket"]["tenant_id"] == tenant_id

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


@pytest.mark.parametrize("competing_action", ["confirm", "rejected", "cancelled"])
def test_concurrent_approval_decisions_have_one_winner(
    smoke_test_engine, monkeypatch, competing_action,
):
    monkeypatch.setattr(ticket_agent_service, "search_chroma", lambda **kwargs: [])
    preview = ticket_agent_service.preview_ticket(
        TicketAgentPreviewRequest(message="VPN 连不上", category="it"),
        tenant_id="tenant_demo",
        user_id="support_demo",
    )
    request = TicketAgentConfirmRequest(
        agent_run_id=preview.agent_run_id,
        approval_request_id=preview.approval_request_id,
        draft=preview.draft,
    )
    barrier = Barrier(2)
    decision = local()
    original_update = agent_ops_service.update_approval_request

    class RacingSession(Session):
        def get(self, entity, *args, **kwargs):
            result = super().get(entity, *args, **kwargs)
            if entity is ApprovalRequest and getattr(decision, "writing", False):
                # Both updates read pending before either writes; no timing sleeps.
                barrier.wait(timeout=10)
            return result

    monkeypatch.setattr(agent_ops_service, "Session", RacingSession)

    def racing_update(*args, **kwargs):
        decision.writing = True
        try:
            return original_update(*args, **kwargs)
        finally:
            decision.writing = False

    monkeypatch.setattr(ticket_agent_service, "update_approval_request", racing_update)

    def decide(action, actor):
        try:
            if action == "confirm":
                return ticket_agent_service.confirm_ticket(
                    request, tenant_id="tenant_demo", created_by=actor,
                )
            return racing_update(
                request.approval_request_id,
                tenant_id="tenant_demo",
                approval_request_update=ApprovalRequestUpdate(
                    status=action, approved_by=actor, decision_reason="Test decision",
                ),
            )
        except HTTPException as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(decide, "confirm", "support_one"),
            pool.submit(decide, competing_action, "support_two"),
        ]
        results = [future.result(timeout=20) for future in futures]

    errors = [result for result in results if isinstance(result, HTTPException)]
    assert len(errors) == 1, (
        "Both decisions succeeded; created ticket IDs: "
        f"{[result.ticket.id for result in results if hasattr(result, 'ticket')]}"
    )
    assert errors[0].status_code == 400
    assert errors[0].detail == "Approval request is not pending"
    winning_index = next(i for i, result in enumerate(results) if not isinstance(result, HTTPException))
    winner = ["support_one", "support_two"][winning_index]
    status = "approved" if winning_index == 0 or competing_action == "confirm" else competing_action

    for later_status in ["pending", "approved", "rejected", "cancelled"]:
        with pytest.raises(HTTPException) as exc_info:
            original_update(
                request.approval_request_id,
                tenant_id="tenant_demo",
                approval_request_update=ApprovalRequestUpdate(
                    status=later_status, approved_by="late_actor", decision_reason="Late decision",
                ),
            )
        assert exc_info.value.status_code == 400

    with Session(smoke_test_engine) as session:
        approval = session.get(ApprovalRequest, request.approval_request_id)
        assert approval.status == status
        assert approval.approved_by == winner
        assert approval.decided_at is not None
        tickets = session.exec(select(Ticket)).all()
        tool_calls = session.exec(select(ToolCall).where(ToolCall.tool_name == "create_ticket")).all()
        assert len(tickets) == len(tool_calls) == (1 if status == "approved" else 0)
        if tickets:
            assert tickets[0].created_by == winner
            assert tickets[0].tenant_id == approval.tenant_id == "tenant_demo"
            assert tool_calls[0].status == "success"
            assert json.loads(tool_calls[0].tool_output_json)["ticket_id"] == tickets[0].id
