from datetime import datetime
from types import SimpleNamespace

from fastapi.testclient import TestClient

import main
import routers.agent_ops as agent_ops_router


client = TestClient(main.app)


def test_list_agent_runs(monkeypatch):
    calls = {}

    def fake_list_agent_runs_service(
        tenant_id: str,
        status: str | None = None,
        agent_name: str | None = None,
    ):
        calls["tenant_id"] = tenant_id
        calls["status"] = status
        calls["agent_name"] = agent_name

        return [
            SimpleNamespace(
                id=1,
                tenant_id=tenant_id,
                user_id="user_demo",
                agent_name="ticket_agent",
                input_message="VPN 连不上",
                category="it",
                status="completed",
                result_summary="Ticket created: 100",
                created_at=datetime(2026, 1, 1, 10, 0, 0),
                updated_at=datetime(2026, 1, 1, 10, 1, 0),
            )
        ]

    monkeypatch.setattr(
        agent_ops_router,
        "list_agent_runs_service",
        fake_list_agent_runs_service,
    )

    response = client.get(
        "/agent-ops/runs",
        params={
            "status": "completed",
            "agent_name": "ticket_agent",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert calls["tenant_id"] == "tenant_demo"
    assert calls["status"] == "completed"
    assert calls["agent_name"] == "ticket_agent"

    assert len(data) == 1
    assert data[0]["id"] == 1
    assert data[0]["tenant_id"] == "tenant_demo"
    assert data[0]["user_id"] == "user_demo"
    assert data[0]["agent_name"] == "ticket_agent"
    assert data[0]["input_message"] == "VPN 连不上"
    assert data[0]["category"] == "it"
    assert data[0]["status"] == "completed"
    assert data[0]["result_summary"] == "Ticket created: 100"


def test_get_agent_run(monkeypatch):
    calls = {}

    def fake_get_agent_run_service(
        agent_run_id: int,
        tenant_id: str,
    ):
        calls["agent_run_id"] = agent_run_id
        calls["tenant_id"] = tenant_id

        return SimpleNamespace(
            id=agent_run_id,
            tenant_id=tenant_id,
            user_id="user_demo",
            agent_name="ticket_agent",
            input_message="VPN 连不上",
            category="it",
            status="completed",
            result_summary="Ticket created: 100",
            created_at=datetime(2026, 1, 1, 10, 0, 0),
            updated_at=datetime(2026, 1, 1, 10, 1, 0),
        )

    monkeypatch.setattr(
        agent_ops_router,
        "get_agent_run_service",
        fake_get_agent_run_service,
    )

    response = client.get("/agent-ops/runs/1")

    assert response.status_code == 200

    data = response.json()

    assert calls["agent_run_id"] == 1
    assert calls["tenant_id"] == "tenant_demo"

    assert data["id"] == 1
    assert data["tenant_id"] == "tenant_demo"
    assert data["status"] == "completed"


def test_list_tool_calls_by_run(monkeypatch):
    calls = {}

    def fake_list_tool_calls_by_run_service(
        agent_run_id: int,
        tenant_id: str,
    ):
        calls["agent_run_id"] = agent_run_id
        calls["tenant_id"] = tenant_id

        return [
            SimpleNamespace(
                id=300,
                agent_run_id=agent_run_id,
                tenant_id=tenant_id,
                tool_name="create_ticket",
                tool_input_json='{"title":"VPN 连不上"}',
                tool_output_json='{"ticket_id":100,"status":"open"}',
                status="success",
                error_message=None,
                created_at=datetime(2026, 1, 1, 10, 0, 10),
                finished_at=datetime(2026, 1, 1, 10, 0, 20),
            )
        ]

    monkeypatch.setattr(
        agent_ops_router,
        "list_tool_calls_by_run_service",
        fake_list_tool_calls_by_run_service,
    )

    response = client.get("/agent-ops/runs/1/tool-calls")

    assert response.status_code == 200

    data = response.json()

    assert calls["agent_run_id"] == 1
    assert calls["tenant_id"] == "tenant_demo"

    assert len(data) == 1
    assert data[0]["id"] == 300
    assert data[0]["agent_run_id"] == 1
    assert data[0]["tool_name"] == "create_ticket"
    assert data[0]["status"] == "success"
    assert data[0]["error_message"] is None


def test_list_approval_requests_by_run(monkeypatch):
    calls = {}

    def fake_list_approval_requests_by_run_service(
        agent_run_id: int,
        tenant_id: str,
    ):
        calls["agent_run_id"] = agent_run_id
        calls["tenant_id"] = tenant_id

        return [
            SimpleNamespace(
                id=10,
                agent_run_id=agent_run_id,
                tenant_id=tenant_id,
                approval_type="ticket_creation",
                status="approved",
                draft_json='{"title":"VPN 连不上"}',
                approved_by="user_demo",
                created_at=datetime(2026, 1, 1, 10, 0, 5),
                decided_at=datetime(2026, 1, 1, 10, 0, 8),
            )
        ]

    monkeypatch.setattr(
        agent_ops_router,
        "list_approval_requests_by_run_service",
        fake_list_approval_requests_by_run_service,
    )

    response = client.get("/agent-ops/runs/1/approval-requests")

    assert response.status_code == 200

    data = response.json()

    assert calls["agent_run_id"] == 1
    assert calls["tenant_id"] == "tenant_demo"

    assert len(data) == 1
    assert data[0]["id"] == 10
    assert data[0]["agent_run_id"] == 1
    assert data[0]["approval_type"] == "ticket_creation"
    assert data[0]["status"] == "approved"
    assert data[0]["approved_by"] == "user_demo"