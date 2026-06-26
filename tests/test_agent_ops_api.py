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
                latency_ms=123,
                retrieval_summary_json='{"top_k":3,"sources_count":1}',
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
    assert data[0]["latency_ms"] == 123
    assert data[0]["retrieval_summary_json"] == '{"top_k":3,"sources_count":1}'

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
                    latency_ms=123,
                    retrieval_summary_json='{"top_k":3,"sources_count":1}',
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

    assert data["latency_ms"] == 123
    assert data["retrieval_summary_json"] == '{"top_k":3,"sources_count":1}'

def test_list_tool_calls_by_run(monkeypatch):
    calls = {}

    def fake_list_tool_calls_by_run_service(
        agent_run_id: int,
        tenant_id: str,
        status: str | None = None,
        tool_name: str | None = None,
        error_type: str | None = None,
    ):
        calls["agent_run_id"] = agent_run_id
        calls["tenant_id"] = tenant_id
        calls["status"] = status
        calls["tool_name"] = tool_name
        calls["error_type"] = error_type

        return [
            SimpleNamespace(
                id=300,
                agent_run_id=agent_run_id,
                tenant_id=tenant_id,
                tool_name="create_ticket",
                tool_input_json='{"title":"VPN 连不上"}',
                tool_output_json='{"ticket_id":100,"status":"open"}',
                status="success",
                error_type=None,
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
    assert calls["status"] is None
    assert calls["tool_name"] is None
    assert calls["error_type"] is None

    assert len(data) == 1
    assert data[0]["id"] == 300
    assert data[0]["agent_run_id"] == 1
    assert data[0]["tool_name"] == "create_ticket"
    assert data[0]["status"] == "success"
    assert data[0]["error_type"] is None
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
                decision_reason="用户已确认创建工单。",
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
    assert data[0]["decision_reason"] == "用户已确认创建工单。"

def test_reject_approval_request(monkeypatch):
    calls = {}

    def fake_update_approval_request_service(
        approval_request_id,
        tenant_id,
        approval_request_update,
    ):
        calls["approval_request_id"] = approval_request_id
        calls["tenant_id"] = tenant_id
        calls["status"] = approval_request_update.status
        calls["approved_by"] = approval_request_update.approved_by
        calls["decision_reason"] = approval_request_update.decision_reason

        return SimpleNamespace(
            id=approval_request_id,
            agent_run_id=1,
            tenant_id=tenant_id,
            approval_type="ticket_creation",
            status=approval_request_update.status,
            draft_json='{"title":"VPN 连不上"}',
            approved_by=approval_request_update.approved_by,
            decision_reason=approval_request_update.decision_reason,
            created_at=datetime(2026, 1, 1, 10, 0, 5),
            decided_at=datetime(2026, 1, 1, 10, 0, 8),
        )

    monkeypatch.setattr(
        agent_ops_router,
        "update_approval_request_service",
        fake_update_approval_request_service,
    )

    response = client.post(
        "/agent-ops/approval-requests/10/reject",
        json={
            "reason": "该请求需要主管审批，暂不创建工单。",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert calls["approval_request_id"] == 10
    assert calls["tenant_id"] == "tenant_demo"
    assert calls["status"] == "rejected"
    assert calls["approved_by"] == "user_demo"
    assert calls["decision_reason"] == "该请求需要主管审批，暂不创建工单。"

    assert data["id"] == 10
    assert data["agent_run_id"] == 1
    assert data["tenant_id"] == "tenant_demo"
    assert data["approval_type"] == "ticket_creation"
    assert data["status"] == "rejected"
    assert data["approved_by"] == "user_demo"
    assert data["decision_reason"] == "该请求需要主管审批，暂不创建工单。"
    assert data["decided_at"] is not None

def test_cancel_approval_request(monkeypatch):
    calls = {}

    def fake_update_approval_request_service(
        approval_request_id,
        tenant_id,
        approval_request_update,
    ):
        calls["approval_request_id"] = approval_request_id
        calls["tenant_id"] = tenant_id
        calls["status"] = approval_request_update.status
        calls["approved_by"] = approval_request_update.approved_by
        calls["decision_reason"] = approval_request_update.decision_reason

        return SimpleNamespace(
            id=approval_request_id,
            agent_run_id=1,
            tenant_id=tenant_id,
            approval_type="ticket_creation",
            status=approval_request_update.status,
            draft_json='{"title":"VPN 连不上"}',
            approved_by=approval_request_update.approved_by,
            decision_reason=approval_request_update.decision_reason,
            created_at=datetime(2026, 1, 1, 10, 0, 5),
            decided_at=datetime(2026, 1, 1, 10, 0, 8),
        )

    monkeypatch.setattr(
        agent_ops_router,
        "update_approval_request_service",
        fake_update_approval_request_service,
    )

    response = client.post(
        "/agent-ops/approval-requests/10/cancel",
        json={
            "reason": "用户撤回了该请求。",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert calls["approval_request_id"] == 10
    assert calls["tenant_id"] == "tenant_demo"
    assert calls["status"] == "cancelled"
    assert calls["approved_by"] == "user_demo"
    assert calls["decision_reason"] == "用户撤回了该请求。"

    assert data["id"] == 10
    assert data["agent_run_id"] == 1
    assert data["tenant_id"] == "tenant_demo"
    assert data["approval_type"] == "ticket_creation"
    assert data["status"] == "cancelled"
    assert data["approved_by"] == "user_demo"
    assert data["decision_reason"] == "用户撤回了该请求。"
    assert data["decided_at"] is not None

def test_get_agent_ops_metrics_summary(monkeypatch):
    calls = {}

    def fake_get_agent_ops_metrics_summary_service(tenant_id: str):
        calls["tenant_id"] = tenant_id

        return SimpleNamespace(
            total_agent_runs=4,
            running_agent_runs=1,
            completed_agent_runs=1,
            failed_agent_runs=1,
            cancelled_agent_runs=1,
            total_tool_calls=3,
            pending_tool_calls=1,
            successful_tool_calls=1,
            failed_tool_calls=1,
            tool_call_error_types={
                "create_ticket_failed": 1,
            },
            total_approval_requests=4,
            pending_approval_requests=1,
            approved_approval_requests=1,
            rejected_approval_requests=1,
            cancelled_approval_requests=1,
        )

    monkeypatch.setattr(
        agent_ops_router,
        "get_agent_ops_metrics_summary_service",
        fake_get_agent_ops_metrics_summary_service,
    )

    response = client.get("/agent-ops/metrics/summary")

    assert response.status_code == 200

    data = response.json()

    assert calls["tenant_id"] == "tenant_demo"

    assert data["total_agent_runs"] == 4
    assert data["running_agent_runs"] == 1
    assert data["completed_agent_runs"] == 1
    assert data["failed_agent_runs"] == 1
    assert data["cancelled_agent_runs"] == 1

    assert data["total_tool_calls"] == 3
    assert data["pending_tool_calls"] == 1
    assert data["successful_tool_calls"] == 1
    assert data["failed_tool_calls"] == 1
    assert data["tool_call_error_types"] == {
        "create_ticket_failed": 1,
    }

    assert data["total_approval_requests"] == 4
    assert data["pending_approval_requests"] == 1
    assert data["approved_approval_requests"] == 1
    assert data["rejected_approval_requests"] == 1
    assert data["cancelled_approval_requests"] == 1


def test_list_tool_calls_by_run_with_filters(monkeypatch):
    calls = {}

    def fake_list_tool_calls_by_run_service(
        agent_run_id: int,
        tenant_id: str,
        status: str | None = None,
        tool_name: str | None = None,
        error_type: str | None = None,
    ):
        calls["agent_run_id"] = agent_run_id
        calls["tenant_id"] = tenant_id
        calls["status"] = status
        calls["tool_name"] = tool_name
        calls["error_type"] = error_type

        return [
            SimpleNamespace(
                id=301,
                agent_run_id=agent_run_id,
                tenant_id=tenant_id,
                tool_name="create_ticket",
                tool_input_json='{"title":"VPN 连不上"}',
                tool_output_json=None,
                status="failed",
                error_type="create_ticket_failed",
                error_message="create ticket failed",
                created_at=datetime(2026, 1, 1, 10, 0, 10),
                finished_at=datetime(2026, 1, 1, 10, 0, 20),
            )
        ]

    monkeypatch.setattr(
        agent_ops_router,
        "list_tool_calls_by_run_service",
        fake_list_tool_calls_by_run_service,
    )

    response = client.get(
        "/agent-ops/runs/1/tool-calls",
        params={
            "status": "failed",
            "tool_name": "create_ticket",
            "error_type": "create_ticket_failed",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert calls["agent_run_id"] == 1
    assert calls["tenant_id"] == "tenant_demo"
    assert calls["status"] == "failed"
    assert calls["tool_name"] == "create_ticket"
    assert calls["error_type"] == "create_ticket_failed"

    assert len(data) == 1
    assert data[0]["id"] == 301
    assert data[0]["status"] == "failed"
    assert data[0]["tool_name"] == "create_ticket"
    assert data[0]["error_type"] == "create_ticket_failed"


def test_list_tool_calls_with_filters(monkeypatch):
    calls = {}

    def fake_list_tool_calls_service(
        tenant_id: str,
        agent_run_id: int | None = None,
        status: str | None = None,
        tool_name: str | None = None,
        error_type: str | None = None,
    ):
        calls["tenant_id"] = tenant_id
        calls["agent_run_id"] = agent_run_id
        calls["status"] = status
        calls["tool_name"] = tool_name
        calls["error_type"] = error_type

        return [
            SimpleNamespace(
                id=301,
                agent_run_id=1,
                tenant_id=tenant_id,
                tool_name="create_ticket",
                tool_input_json='{"title":"VPN 连不上"}',
                tool_output_json=None,
                status="failed",
                error_type="create_ticket_failed",
                error_message="create ticket failed",
                created_at=datetime(2026, 1, 1, 10, 0, 10),
                finished_at=datetime(2026, 1, 1, 10, 0, 20),
            )
        ]

    monkeypatch.setattr(
        agent_ops_router,
        "list_tool_calls_service",
        fake_list_tool_calls_service,
    )

    response = client.get(
        "/agent-ops/tool-calls",
        params={
            "agent_run_id": 1,
            "status": "failed",
            "tool_name": "create_ticket",
            "error_type": "create_ticket_failed",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert calls["tenant_id"] == "tenant_demo"
    assert calls["agent_run_id"] == 1
    assert calls["status"] == "failed"
    assert calls["tool_name"] == "create_ticket"
    assert calls["error_type"] == "create_ticket_failed"

    assert len(data) == 1
    assert data[0]["id"] == 301
    assert data[0]["agent_run_id"] == 1
    assert data[0]["tool_name"] == "create_ticket"
    assert data[0]["status"] == "failed"
    assert data[0]["error_type"] == "create_ticket_failed"


def test_list_approval_requests_with_filters(monkeypatch):
    calls = {}

    def fake_list_approval_requests_service(
        tenant_id: str,
        agent_run_id: int | None = None,
        status: str | None = None,
        approval_type: str | None = None,
    ):
        calls["tenant_id"] = tenant_id
        calls["agent_run_id"] = agent_run_id
        calls["status"] = status
        calls["approval_type"] = approval_type

        return [
            SimpleNamespace(
                id=401,
                agent_run_id=1,
                tenant_id=tenant_id,
                approval_type="ticket_creation",
                status="pending",
                draft_json='{"title":"VPN 连不上"}',
                approved_by=None,
                decision_reason=None,
                created_at=datetime(2026, 1, 1, 10, 0, 10),
                decided_at=None,
            )
        ]

    monkeypatch.setattr(
        agent_ops_router,
        "list_approval_requests_service",
        fake_list_approval_requests_service,
    )

    response = client.get(
        "/agent-ops/approval-requests",
        params={
            "agent_run_id": 1,
            "status": "pending",
            "approval_type": "ticket_creation",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert calls["tenant_id"] == "tenant_demo"
    assert calls["agent_run_id"] == 1
    assert calls["status"] == "pending"
    assert calls["approval_type"] == "ticket_creation"

    assert len(data) == 1
    assert data[0]["id"] == 401
    assert data[0]["agent_run_id"] == 1
    assert data[0]["approval_type"] == "ticket_creation"
    assert data[0]["status"] == "pending"
    assert data[0]["decision_reason"] is None