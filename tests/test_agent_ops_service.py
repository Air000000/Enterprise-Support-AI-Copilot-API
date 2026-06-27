import json

import pytest
from fastapi import HTTPException
from sqlmodel import SQLModel, create_engine

from models.agent_ops import AgentRun, ApprovalRequest, ToolCall
from schemas.agent_ops import (
    AgentRunCreate,
    AgentRunUpdate,
    ApprovalRequestCreate,
    ApprovalRequestUpdate,
    RetrievalLogCreate,
    ToolCallCreate,
    ToolCallUpdate,
)
import services.agent_ops_service as agent_ops_service


@pytest.fixture()
def agent_ops_test_engine(tmp_path, monkeypatch):
    """
    创建一个临时的 SQLite 数据库，并将 agent_ops_service 的 engine 替换为这个临时数据库。
    """
    db_path = tmp_path / "agent_ops_test.db"
    test_engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )

    SQLModel.metadata.create_all(test_engine)

    monkeypatch.setattr(agent_ops_service, "engine", test_engine)

    return test_engine


def test_create_and_update_agent_run(agent_ops_test_engine):
    """
    测试创建一个 agent run，并更新它的状态和结果摘要。
    """
    agent_run = agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="tenant_demo",
            user_id="user_demo",
            input_message="VPN 连不上，重启客户端也没用",
            category="it",
        )
    )

    assert agent_run.id is not None
    assert agent_run.tenant_id == "tenant_demo"
    assert agent_run.user_id == "user_demo"
    assert agent_run.agent_name == "ticket_agent"
    assert agent_run.input_message == "VPN 连不上，重启客户端也没用"
    assert agent_run.category == "it"
    assert agent_run.status == "running"

    updated = agent_ops_service.update_agent_run(
        agent_run_id=agent_run.id,
        tenant_id="tenant_demo",
        agent_run_update=AgentRunUpdate(
            status="completed",
            result_summary="生成工单草稿并等待用户确认。",
            latency_ms=123,
            retrieval_summary_json='{"top_k":3,"sources_count":1}',
        ),
    )

    assert updated.status == "completed"
    assert updated.result_summary == "生成工单草稿并等待用户确认。"
    assert updated.updated_at is not None
    assert updated.latency_ms == 123
    assert updated.retrieval_summary_json == '{"top_k":3,"sources_count":1}'

    fetched = agent_ops_service.get_agent_run(
        agent_run_id=agent_run.id,
        tenant_id="tenant_demo",
    )

    assert fetched.id == agent_run.id
    assert fetched.status == "completed"
    assert fetched.latency_ms == 123
    assert fetched.retrieval_summary_json == '{"top_k":3,"sources_count":1}'

def test_list_agent_runs_with_filters(agent_ops_test_engine):
    """
    测试根据状态和 agent_name 过滤 agent runs。
    """
    agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="tenant_demo",
            user_id="user_demo",
            input_message="VPN 连不上",
            category="it",
            status="completed",
        )
    )

    agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="tenant_demo",
            user_id="user_demo",
            input_message="请假怎么申请",
            category="hr",
            status="running",
        )
    )

    completed_runs = agent_ops_service.list_agent_runs(
        tenant_id="tenant_demo",
        status="completed",
    )

    assert len(completed_runs) == 1
    assert completed_runs[0].status == "completed"
    assert completed_runs[0].category == "it"

    ticket_agent_runs = agent_ops_service.list_agent_runs(
        tenant_id="tenant_demo",
        agent_name="ticket_agent",
    )

    assert len(ticket_agent_runs) == 2


def test_get_agent_run_with_wrong_tenant_should_return_404(agent_ops_test_engine):
    """
    测试尝试获取一个存在的 agent run，但使用了错误的 tenant_id，应该返回 404 错误。
    """
    agent_run = agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="tenant_demo",
            user_id="user_demo",
            input_message="VPN 连不上",
            category="it",
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        agent_ops_service.get_agent_run(
            agent_run_id=agent_run.id,
            tenant_id="other_tenant",
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Agent run not found"


def test_create_and_update_tool_call(agent_ops_test_engine):
    """
    测试创建一个 tool call，并更新它的状态和输出。
    """
    agent_run = agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="tenant_demo",
            user_id="user_demo",
            input_message="VPN 连不上",
            category="it",
        )
    )

    tool_input = {
        "title": "VPN 连不上",
        "category": "it",
        "priority": "high",
    }

    tool_call = agent_ops_service.create_tool_call(
        ToolCallCreate(
            agent_run_id=agent_run.id,
            tenant_id="tenant_demo",
            tool_name="create_ticket",
            tool_input_json=json.dumps(tool_input, ensure_ascii=False),
        )
    )

    assert tool_call.id is not None
    assert tool_call.agent_run_id == agent_run.id
    assert tool_call.tenant_id == "tenant_demo"
    assert tool_call.tool_name == "create_ticket"
    assert tool_call.status == "pending"
    assert json.loads(tool_call.tool_input_json)["title"] == "VPN 连不上"

    tool_output = {
        "ticket_id": 1,
        "status": "open",
    }

    updated = agent_ops_service.update_tool_call(
        tool_call_id=tool_call.id,
        tenant_id="tenant_demo",
        tool_call_update=ToolCallUpdate(
            status="success",
            tool_output_json=json.dumps(tool_output, ensure_ascii=False),
        ),
    )

    assert updated.status == "success"
    assert updated.finished_at is not None
    assert json.loads(updated.tool_output_json)["ticket_id"] == 1


    failed_tool_call = agent_ops_service.create_tool_call(
        ToolCallCreate(
            agent_run_id=agent_run.id,
            tenant_id="tenant_demo",
            tool_name="search_kb",
            tool_input_json='{"query":"VPN 连不上"}',
        )
    )

    failed_updated = agent_ops_service.update_tool_call(
        tool_call_id=failed_tool_call.id,
        tenant_id="tenant_demo",
        tool_call_update=ToolCallUpdate(
            status="failed",
            error_type="search_kb_failed",
            error_message="Chroma search failed",
        ),
    )

    assert failed_updated.status == "failed"
    assert failed_updated.error_type == "search_kb_failed"
    assert failed_updated.error_message == "Chroma search failed"
    assert failed_updated.finished_at is not None


    tool_calls = agent_ops_service.list_tool_calls_by_run(
        agent_run_id=agent_run.id,
        tenant_id="tenant_demo",
    )

    assert len(tool_calls) == 2
    assert tool_calls[0].id == tool_call.id


def test_list_tool_calls_by_run_with_filters(agent_ops_test_engine):
    agent_run = agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="tenant_demo",
            user_id="user_demo",
            input_message="VPN 连不上",
            category="it",
        )
    )

    agent_ops_service.create_tool_call(
        ToolCallCreate(
            agent_run_id=agent_run.id,
            tenant_id="tenant_demo",
            tool_name="search_kb",
            status="success",
            tool_input_json='{"query":"VPN 连不上"}',
        )
    )

    agent_ops_service.create_tool_call(
        ToolCallCreate(
            agent_run_id=agent_run.id,
            tenant_id="tenant_demo",
            tool_name="classify_ticket",
            status="failed",
            error_type="classify_ticket_failed",
            error_message="classification failed",
            tool_input_json='{"message":"VPN 连不上"}',
        )
    )

    agent_ops_service.create_tool_call(
        ToolCallCreate(
            agent_run_id=agent_run.id,
            tenant_id="tenant_demo",
            tool_name="create_ticket",
            status="failed",
            error_type="create_ticket_failed",
            error_message="create ticket failed",
            tool_input_json='{"title":"VPN 连不上"}',
        )
    )

    failed_tool_calls = agent_ops_service.list_tool_calls_by_run(
        agent_run_id=agent_run.id,
        tenant_id="tenant_demo",
        status="failed",
    )

    assert len(failed_tool_calls) == 2
    assert {tool_call.status for tool_call in failed_tool_calls} == {"failed"}

    search_kb_tool_calls = agent_ops_service.list_tool_calls_by_run(
        agent_run_id=agent_run.id,
        tenant_id="tenant_demo",
        tool_name="search_kb",
    )

    assert len(search_kb_tool_calls) == 1
    assert search_kb_tool_calls[0].tool_name == "search_kb"
    assert search_kb_tool_calls[0].status == "success"

    create_ticket_failed_tool_calls = agent_ops_service.list_tool_calls_by_run(
        agent_run_id=agent_run.id,
        tenant_id="tenant_demo",
        error_type="create_ticket_failed",
    )

    assert len(create_ticket_failed_tool_calls) == 1
    assert create_ticket_failed_tool_calls[0].tool_name == "create_ticket"
    assert create_ticket_failed_tool_calls[0].error_type == "create_ticket_failed"

    combined_filtered_tool_calls = agent_ops_service.list_tool_calls_by_run(
        agent_run_id=agent_run.id,
        tenant_id="tenant_demo",
        status="failed",
        tool_name="create_ticket",
        error_type="create_ticket_failed",
    )

    assert len(combined_filtered_tool_calls) == 1
    assert combined_filtered_tool_calls[0].tool_name == "create_ticket"
    assert combined_filtered_tool_calls[0].status == "failed"
    assert combined_filtered_tool_calls[0].error_type == "create_ticket_failed"


def test_list_tool_calls_with_global_filters(agent_ops_test_engine):
    agent_run = agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="tenant_demo",
            user_id="user_demo",
            input_message="VPN 连不上",
            category="it",
        )
    )

    other_run = agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="tenant_demo",
            user_id="user_demo",
            input_message="邮箱无法登录",
            category="it",
        )
    )

    other_tenant_run = agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="other_tenant",
            user_id="user_demo",
            input_message="其他租户问题",
            category="it",
        )
    )

    agent_ops_service.create_tool_call(
        ToolCallCreate(
            agent_run_id=agent_run.id,
            tenant_id="tenant_demo",
            tool_name="search_kb",
            status="success",
            tool_input_json='{"query":"VPN 连不上"}',
        )
    )

    agent_ops_service.create_tool_call(
        ToolCallCreate(
            agent_run_id=agent_run.id,
            tenant_id="tenant_demo",
            tool_name="create_ticket",
            status="failed",
            error_type="create_ticket_failed",
            error_message="create ticket failed",
            tool_input_json='{"title":"VPN 连不上"}',
        )
    )

    agent_ops_service.create_tool_call(
        ToolCallCreate(
            agent_run_id=other_run.id,
            tenant_id="tenant_demo",
            tool_name="classify_ticket",
            status="failed",
            error_type="classify_ticket_failed",
            error_message="classification failed",
            tool_input_json='{"message":"邮箱无法登录"}',
        )
    )

    agent_ops_service.create_tool_call(
        ToolCallCreate(
            agent_run_id=other_tenant_run.id,
            tenant_id="other_tenant",
            tool_name="create_ticket",
            status="failed",
            error_type="create_ticket_failed",
            error_message="other tenant failed",
            tool_input_json='{"title":"其他租户问题"}',
        )
    )

    failed_tool_calls = agent_ops_service.list_tool_calls(
        tenant_id="tenant_demo",
        status="failed",
    )

    assert len(failed_tool_calls) == 2
    assert {tool_call.tenant_id for tool_call in failed_tool_calls} == {
        "tenant_demo"
    }
    assert {tool_call.status for tool_call in failed_tool_calls} == {"failed"}

    create_ticket_failed_tool_calls = agent_ops_service.list_tool_calls(
        tenant_id="tenant_demo",
        tool_name="create_ticket",
        error_type="create_ticket_failed",
    )

    assert len(create_ticket_failed_tool_calls) == 1
    assert create_ticket_failed_tool_calls[0].agent_run_id == agent_run.id
    assert create_ticket_failed_tool_calls[0].tool_name == "create_ticket"
    assert create_ticket_failed_tool_calls[0].error_type == "create_ticket_failed"

    run_filtered_tool_calls = agent_ops_service.list_tool_calls(
        tenant_id="tenant_demo",
        agent_run_id=other_run.id,
        status="failed",
    )

    assert len(run_filtered_tool_calls) == 1
    assert run_filtered_tool_calls[0].agent_run_id == other_run.id
    assert run_filtered_tool_calls[0].tool_name == "classify_ticket"
    assert run_filtered_tool_calls[0].error_type == "classify_ticket_failed"


    paginated_tool_calls = agent_ops_service.list_tool_calls(
        tenant_id="tenant_demo",
        status="failed",
        limit=1,
        offset=1,
    )

    assert len(paginated_tool_calls) == 1
    assert paginated_tool_calls[0].tenant_id == "tenant_demo"
    assert paginated_tool_calls[0].status == "failed"

def test_create_tool_call_with_missing_agent_run_should_return_404(agent_ops_test_engine):
    """
    测试尝试创建一个 tool call，但关联的 agent run 不存在，应该返回 404 错误。
    """
    with pytest.raises(HTTPException) as exc_info:
        agent_ops_service.create_tool_call(
            ToolCallCreate(
                agent_run_id=999999,
                tenant_id="tenant_demo",
                tool_name="create_ticket",
                tool_input_json="{}",
            )
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Agent run not found"


def test_get_tool_call_with_wrong_tenant_should_return_404(agent_ops_test_engine):
    """
    测试尝试获取一个存在的 tool call，但使用了错误的 tenant_id，应该返回 404 错误。
    """
    agent_run = agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="tenant_demo",
            user_id="user_demo",
            input_message="VPN 连不上",
            category="it",
        )
    )

    tool_call = agent_ops_service.create_tool_call(
        ToolCallCreate(
            agent_run_id=agent_run.id,
            tenant_id="tenant_demo",
            tool_name="create_ticket",
            tool_input_json="{}",
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        agent_ops_service.get_tool_call(
            tool_call_id=tool_call.id,
            tenant_id="other_tenant",
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Tool call not found"


def test_create_and_update_approval_request(agent_ops_test_engine):
    """
    测试创建和更新审批请求。
    """
    agent_run = agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="tenant_demo",
            user_id="user_demo",
            input_message="VPN 连不上",
            category="it",
        )
    )

    draft = {
        "title": "VPN 连不上",
        "description": "用户远程办公时 VPN 连不上。",
        "category": "it",
        "priority": "high",
    }

    approval_request = agent_ops_service.create_approval_request(
        ApprovalRequestCreate(
            agent_run_id=agent_run.id,
            tenant_id="tenant_demo",
            approval_type="ticket_creation",
            draft_json=json.dumps(draft, ensure_ascii=False),
        )
    )

    assert approval_request.id is not None
    assert approval_request.agent_run_id == agent_run.id
    assert approval_request.tenant_id == "tenant_demo"
    assert approval_request.approval_type == "ticket_creation"
    assert approval_request.status == "pending"
    assert json.loads(approval_request.draft_json)["title"] == "VPN 连不上"

    updated = agent_ops_service.update_approval_request(
        approval_request_id=approval_request.id,
        tenant_id="tenant_demo",
        approval_request_update=ApprovalRequestUpdate(
            status="approved",
            approved_by="user_demo",
            decision_reason="该请求需要主管审批，暂不创建工单。",
        ),
    )

    assert updated.status == "approved"
    assert updated.approved_by == "user_demo"
    assert updated.decision_reason == "该请求需要主管审批，暂不创建工单。"
    assert updated.decided_at is not None

    approval_requests = agent_ops_service.list_approval_requests_by_run(
        agent_run_id=agent_run.id,
        tenant_id="tenant_demo",
    )

    assert len(approval_requests) == 1
    assert approval_requests[0].id == approval_request.id


def test_create_approval_request_with_missing_agent_run_should_return_404(agent_ops_test_engine):
    """
    测试尝试创建一个审批请求，但关联的 agent run 不存在，应该返回 404 错误。
    """
    with pytest.raises(HTTPException) as exc_info:
        agent_ops_service.create_approval_request(
            ApprovalRequestCreate(
                agent_run_id=999999,
                tenant_id="tenant_demo",
                approval_type="ticket_creation",
                draft_json="{}",
            )
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Agent run not found"


def test_get_approval_request_with_wrong_tenant_should_return_404(agent_ops_test_engine):
    """
    测试尝试获取一个存在的审批请求，但使用了错误的 tenant_id，应该返回 404 错误。
    """
    agent_run = agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="tenant_demo",
            user_id="user_demo",
            input_message="VPN 连不上",
            category="it",
        )
    )

    approval_request = agent_ops_service.create_approval_request(
        ApprovalRequestCreate(
            agent_run_id=agent_run.id,
            tenant_id="tenant_demo",
            approval_type="ticket_creation",
            draft_json="{}",
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        agent_ops_service.get_approval_request(
            approval_request_id=approval_request.id,
            tenant_id="other_tenant",
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Approval request not found"

def test_get_agent_ops_metrics_summary_counts_by_tenant(agent_ops_test_engine):
    """
    测试 AgentOps summary 只统计当前 tenant 下的 run / tool_call / approval_request。
    """
    completed_run = agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="tenant_demo",
            user_id="user_demo",
            input_message="VPN 连不上",
            category="it",
            status="completed",
        )
    )

    failed_run = agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="tenant_demo",
            user_id="user_demo",
            input_message="创建工单失败",
            category="it",
            status="failed",
        )
    )

    cancelled_run = agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="tenant_demo",
            user_id="user_demo",
            input_message="取消审批",
            category="it",
            status="cancelled",
        )
    )

    running_run = agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="tenant_demo",
            user_id="user_demo",
            input_message="等待处理",
            category="hr",
            status="running",
        )
    )

    other_tenant_run = agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="other_tenant",
            user_id="user_demo",
            input_message="其他租户问题",
            category="it",
            status="completed",
        )
    )

    agent_ops_service.create_tool_call(
        ToolCallCreate(
            agent_run_id=completed_run.id,
            tenant_id="tenant_demo",
            tool_name="create_ticket",
            status="success",
        )
    )

    agent_ops_service.create_tool_call(
        ToolCallCreate(
            agent_run_id=failed_run.id,
            tenant_id="tenant_demo",
            tool_name="create_ticket",
            status="failed",
            error_type="create_ticket_failed",
            error_message="create ticket failed",
        )
    )

    agent_ops_service.create_tool_call(
        ToolCallCreate(
            agent_run_id=running_run.id,
            tenant_id="tenant_demo",
            tool_name="create_ticket",
            status="pending",
        )
    )

    agent_ops_service.create_tool_call(
        ToolCallCreate(
            agent_run_id=other_tenant_run.id,
            tenant_id="other_tenant",
            tool_name="create_ticket",
            status="success",
        )
    )

    agent_ops_service.create_approval_request(
        ApprovalRequestCreate(
            agent_run_id=completed_run.id,
            tenant_id="tenant_demo",
            status="approved",
            draft_json="{}",
            approved_by="user_demo",
        )
    )

    agent_ops_service.create_approval_request(
        ApprovalRequestCreate(
            agent_run_id=failed_run.id,
            tenant_id="tenant_demo",
            status="rejected",
            draft_json="{}",
            approved_by="user_demo",
        )
    )

    agent_ops_service.create_approval_request(
        ApprovalRequestCreate(
            agent_run_id=cancelled_run.id,
            tenant_id="tenant_demo",
            status="cancelled",
            draft_json="{}",
            approved_by="user_demo",
        )
    )

    agent_ops_service.create_approval_request(
        ApprovalRequestCreate(
            agent_run_id=running_run.id,
            tenant_id="tenant_demo",
            status="pending",
            draft_json="{}",
        )
    )

    agent_ops_service.create_approval_request(
        ApprovalRequestCreate(
            agent_run_id=other_tenant_run.id,
            tenant_id="other_tenant",
            status="approved",
            draft_json="{}",
            approved_by="user_demo",
        )
    )

    summary = agent_ops_service.get_agent_ops_metrics_summary(
        tenant_id="tenant_demo",
    )

    assert summary.total_agent_runs == 4
    assert summary.running_agent_runs == 1
    assert summary.completed_agent_runs == 1
    assert summary.failed_agent_runs == 1
    assert summary.cancelled_agent_runs == 1

    assert summary.total_tool_calls == 3
    assert summary.pending_tool_calls == 1
    assert summary.successful_tool_calls == 1
    assert summary.failed_tool_calls == 1
    assert summary.tool_call_error_types == {
        "create_ticket_failed": 1,
    }

    assert summary.total_approval_requests == 4
    assert summary.pending_approval_requests == 1
    assert summary.approved_approval_requests == 1
    assert summary.rejected_approval_requests == 1
    assert summary.cancelled_approval_requests == 1


def test_list_approval_requests_with_global_filters(agent_ops_test_engine):
    agent_run = agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="tenant_demo",
            user_id="user_demo",
            input_message="VPN 连不上",
            category="it",
        )
    )

    other_run = agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="tenant_demo",
            user_id="user_demo",
            input_message="邮箱无法登录",
            category="it",
        )
    )

    other_tenant_run = agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="other_tenant",
            user_id="user_demo",
            input_message="其他租户问题",
            category="it",
        )
    )

    agent_ops_service.create_approval_request(
        ApprovalRequestCreate(
            agent_run_id=agent_run.id,
            tenant_id="tenant_demo",
            approval_type="ticket_creation",
            status="pending",
            draft_json='{"title":"VPN 连不上"}',
        )
    )

    agent_ops_service.create_approval_request(
        ApprovalRequestCreate(
            agent_run_id=other_run.id,
            tenant_id="tenant_demo",
            approval_type="ticket_creation",
            status="rejected",
            draft_json='{"title":"邮箱无法登录"}',
        )
    )

    agent_ops_service.create_approval_request(
        ApprovalRequestCreate(
            agent_run_id=other_run.id,
            tenant_id="tenant_demo",
            approval_type="ticket_creation",
            status="pending",
            draft_json='{"ticket_id":1}',
        )
    )

    agent_ops_service.create_approval_request(
        ApprovalRequestCreate(
            agent_run_id=other_tenant_run.id,
            tenant_id="other_tenant",
            approval_type="ticket_creation",
            status="pending",
            draft_json='{"title":"其他租户问题"}',
        )
    )

    pending_approval_requests = agent_ops_service.list_approval_requests(
        tenant_id="tenant_demo",
        status="pending",
    )

    assert len(pending_approval_requests) == 2
    assert {
        approval_request.tenant_id
        for approval_request in pending_approval_requests
    } == {"tenant_demo"}
    assert {
        approval_request.status
        for approval_request in pending_approval_requests
    } == {"pending"}

    ticket_creation_approval_requests = agent_ops_service.list_approval_requests(
        tenant_id="tenant_demo",
        approval_type="ticket_creation",
    )

    assert len(ticket_creation_approval_requests) == 3
    assert {
        approval_request.approval_type
        for approval_request in ticket_creation_approval_requests
    } == {"ticket_creation"}

    run_filtered_approval_requests = agent_ops_service.list_approval_requests(
        tenant_id="tenant_demo",
        agent_run_id=other_run.id,
    )

    assert len(run_filtered_approval_requests) == 2
    assert {
        approval_request.agent_run_id
        for approval_request in run_filtered_approval_requests
    } == {other_run.id}

    combined_filtered_approval_requests = agent_ops_service.list_approval_requests(
        tenant_id="tenant_demo",
        agent_run_id=other_run.id,
        status="rejected",
        approval_type="ticket_creation",
    )

    assert len(combined_filtered_approval_requests) == 1
    assert combined_filtered_approval_requests[0].agent_run_id == other_run.id
    assert combined_filtered_approval_requests[0].status == "rejected"
    assert combined_filtered_approval_requests[0].approval_type == "ticket_creation"


    paginated_approval_requests = agent_ops_service.list_approval_requests(
        tenant_id="tenant_demo",
        status="pending",
        limit=1,
        offset=1,
    )

    assert len(paginated_approval_requests) == 1
    assert paginated_approval_requests[0].tenant_id == "tenant_demo"
    assert paginated_approval_requests[0].status == "pending"
    
def test_list_agent_runs_with_pagination(agent_ops_test_engine):
    first_run = agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="tenant_demo",
            user_id="user_demo",
            input_message="第一个请求",
            category="it",
        )
    )

    second_run = agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="tenant_demo",
            user_id="user_demo",
            input_message="第二个请求",
            category="it",
        )
    )

    third_run = agent_ops_service.create_agent_run(
        AgentRunCreate(
            tenant_id="tenant_demo",
            user_id="user_demo",
            input_message="第三个请求",
            category="it",
        )
    )

    first_page = agent_ops_service.list_agent_runs(
        tenant_id="tenant_demo",
        limit=2,
        offset=0,
    )

    second_page = agent_ops_service.list_agent_runs(
        tenant_id="tenant_demo",
        limit=2,
        offset=2,
    )

    assert len(first_page) == 2
    assert len(second_page) == 1

    assert [agent_run.id for agent_run in first_page] == [
        third_run.id,
        second_run.id,
    ]
    assert [agent_run.id for agent_run in second_page] == [first_run.id]


def test_create_and_list_retrieval_logs(agent_ops_test_engine):
    agent_ops_service.create_retrieval_log(
        RetrievalLogCreate(
            tenant_id="tenant_demo",
            endpoint="search",
            query_text="VPN 连不上怎么办？",
            top_k=3,
            category="it",
            retrieval_status="ok",
            total_hits=2,
            top_distance=0.3123,
            source_documents_json='[{"document_id":"doc_vpn"}]',
            scores_json="[0.3123, 0.4567]",
            latency_ms=123,
        )
    )

    agent_ops_service.create_retrieval_log(
        RetrievalLogCreate(
            tenant_id="tenant_demo",
            endpoint="search",
            query_text="邮箱无法登录",
            top_k=3,
            category="it",
            retrieval_status="no_context",
            total_hits=0,
            latency_ms=88,
        )
    )

    agent_ops_service.create_retrieval_log(
        RetrievalLogCreate(
            tenant_id="other_tenant",
            endpoint="search",
            query_text="其他租户问题",
            top_k=3,
            category="it",
            retrieval_status="ok",
            total_hits=1,
        )
    )

    logs = agent_ops_service.list_retrieval_logs(
        tenant_id="tenant_demo",
        endpoint="search",
        retrieval_status="ok",
        category="it",
        limit=20,
        offset=0,
    )

    assert len(logs) == 1
    assert logs[0].tenant_id == "tenant_demo"
    assert logs[0].endpoint == "search"
    assert logs[0].query_text == "VPN 连不上怎么办？"
    assert logs[0].retrieval_status == "ok"
    assert logs[0].total_hits == 2
    assert logs[0].top_distance == 0.3123
    assert logs[0].source_documents_json == '[{"document_id":"doc_vpn"}]'
    assert logs[0].scores_json == "[0.3123, 0.4567]"


def test_get_retrieval_metrics_summary(agent_ops_test_engine):
    agent_ops_service.create_retrieval_log(
        RetrievalLogCreate(
            tenant_id="tenant_demo",
            endpoint="search",
            query_text="VPN 连不上怎么办？",
            top_k=3,
            category="it",
            retrieval_status="ok",
            total_hits=2,
            top_distance=0.3,
            latency_ms=100,
        )
    )

    agent_ops_service.create_retrieval_log(
        RetrievalLogCreate(
            tenant_id="tenant_demo",
            endpoint="ask",
            query_text="邮箱无法登录怎么办？",
            top_k=3,
            category="it",
            retrieval_status="no_context",
            total_hits=0,
            top_distance=None,
            latency_ms=200,
        )
    )

    agent_ops_service.create_retrieval_log(
        RetrievalLogCreate(
            tenant_id="tenant_demo",
            endpoint="ask",
            query_text="报销流程是什么？",
            top_k=3,
            category="hr",
            retrieval_status="failed",
            total_hits=0,
            top_distance=None,
            latency_ms=300,
            error_message="LLM is unavailable",
        )
    )

    agent_ops_service.create_retrieval_log(
        RetrievalLogCreate(
            tenant_id="other_tenant",
            endpoint="search",
            query_text="其他租户问题",
            top_k=3,
            category="it",
            retrieval_status="ok",
            total_hits=1,
            top_distance=0.1,
            latency_ms=50,
        )
    )

    summary = agent_ops_service.get_retrieval_metrics_summary(
        tenant_id="tenant_demo",
    )

    assert summary.total_retrieval_logs == 3
    assert summary.ok_retrieval_logs == 1
    assert summary.no_context_retrieval_logs == 1
    assert summary.failed_retrieval_logs == 1
    assert summary.average_latency_ms == 200
    assert summary.average_top_distance == 0.3
    assert summary.endpoint_counts == {
        "search": 1,
        "ask": 2,
    }
    assert summary.category_counts == {
        "it": 2,
        "hr": 1,
    }

    ask_summary = agent_ops_service.get_retrieval_metrics_summary(
        tenant_id="tenant_demo",
        endpoint="ask",
    )

    assert ask_summary.total_retrieval_logs == 2
    assert ask_summary.ok_retrieval_logs == 0
    assert ask_summary.no_context_retrieval_logs == 1
    assert ask_summary.failed_retrieval_logs == 1
    assert ask_summary.endpoint_counts == {
        "ask": 2,
    }

    it_summary = agent_ops_service.get_retrieval_metrics_summary(
        tenant_id="tenant_demo",
        category="it",
    )

    assert it_summary.total_retrieval_logs == 2
    assert it_summary.category_counts == {
        "it": 2,
    }


def test_get_retrieval_source_metrics(agent_ops_test_engine):
    agent_ops_service.create_retrieval_log(
        RetrievalLogCreate(
            tenant_id="tenant_demo",
            endpoint="search",
            query_text="VPN 连不上怎么办？",
            top_k=3,
            category="it",
            retrieval_status="ok",
            total_hits=2,
            source_documents_json=(
                "["
                '{"document_id":"doc_vpn","title":"VPN 手册",'
                '"source_path":"docs/it/vpn.md","distance":0.3},'
                '{"document_id":"doc_email","title":"邮箱手册",'
                '"source_path":"docs/it/email.md","distance":0.6}'
                "]"
            ),
            latency_ms=100,
        )
    )

    agent_ops_service.create_retrieval_log(
        RetrievalLogCreate(
            tenant_id="tenant_demo",
            endpoint="ask",
            query_text="VPN 配置方法？",
            top_k=3,
            category="it",
            retrieval_status="ok",
            total_hits=1,
            source_documents_json=(
                "["
                '{"document_id":"doc_vpn","title":"VPN 手册",'
                '"source_path":"docs/it/vpn.md","distance":0.4}'
                "]"
            ),
            latency_ms=120,
        )
    )

    agent_ops_service.create_retrieval_log(
        RetrievalLogCreate(
            tenant_id="other_tenant",
            endpoint="search",
            query_text="其他租户 VPN 问题",
            top_k=3,
            category="it",
            retrieval_status="ok",
            total_hits=1,
            source_documents_json=(
                "["
                '{"document_id":"doc_vpn","title":"VPN 手册",'
                '"source_path":"docs/it/vpn.md","distance":0.1}'
                "]"
            ),
            latency_ms=80,
        )
    )

    metrics = agent_ops_service.get_retrieval_source_metrics(
        tenant_id="tenant_demo",
        limit=10,
    )

    assert len(metrics) == 2
    assert metrics[0].document_id == "doc_vpn"
    assert metrics[0].title == "VPN 手册"
    assert metrics[0].source_path == "docs/it/vpn.md"
    assert metrics[0].retrieval_count == 2
    assert metrics[0].average_distance == 0.35

    assert metrics[1].document_id == "doc_email"
    assert metrics[1].retrieval_count == 1
    assert metrics[1].average_distance == 0.6

    ask_metrics = agent_ops_service.get_retrieval_source_metrics(
        tenant_id="tenant_demo",
        endpoint="ask",
        limit=10,
    )

    assert len(ask_metrics) == 1
    assert ask_metrics[0].document_id == "doc_vpn"
    assert ask_metrics[0].retrieval_count == 1
    assert ask_metrics[0].average_distance == 0.4

    limited_metrics = agent_ops_service.get_retrieval_source_metrics(
        tenant_id="tenant_demo",
        limit=1,
    )

    assert len(limited_metrics) == 1
    assert limited_metrics[0].document_id == "doc_vpn"


def test_get_retrieval_no_context_query_metrics(agent_ops_test_engine):
    agent_ops_service.create_retrieval_log(
        RetrievalLogCreate(
            tenant_id="tenant_demo",
            endpoint="ask",
            query_text="VPN 怎么配置？",
            top_k=3,
            category="it",
            retrieval_status="no_context",
            total_hits=0,
            latency_ms=100,
        )
    )

    agent_ops_service.create_retrieval_log(
        RetrievalLogCreate(
            tenant_id="tenant_demo",
            endpoint="ask",
            query_text="VPN 怎么配置？",
            top_k=3,
            category="it",
            retrieval_status="no_context",
            total_hits=0,
            latency_ms=200,
        )
    )

    agent_ops_service.create_retrieval_log(
        RetrievalLogCreate(
            tenant_id="tenant_demo",
            endpoint="search",
            query_text="打印机驱动怎么下载？",
            top_k=3,
            category="it",
            retrieval_status="no_context",
            total_hits=0,
            latency_ms=150,
        )
    )

    agent_ops_service.create_retrieval_log(
        RetrievalLogCreate(
            tenant_id="tenant_demo",
            endpoint="ask",
            query_text="VPN 怎么配置？",
            top_k=3,
            category="it",
            retrieval_status="ok",
            total_hits=1,
            latency_ms=80,
        )
    )

    agent_ops_service.create_retrieval_log(
        RetrievalLogCreate(
            tenant_id="other_tenant",
            endpoint="ask",
            query_text="VPN 怎么配置？",
            top_k=3,
            category="it",
            retrieval_status="no_context",
            total_hits=0,
            latency_ms=50,
        )
    )

    metrics = agent_ops_service.get_retrieval_no_context_query_metrics(
        tenant_id="tenant_demo",
        limit=10,
    )

    assert len(metrics) == 2

    assert metrics[0].query_text == "VPN 怎么配置？"
    assert metrics[0].endpoint == "ask"
    assert metrics[0].category == "it"
    assert metrics[0].no_context_count == 2
    assert metrics[0].latest_latency_ms == 200

    assert metrics[1].query_text == "打印机驱动怎么下载？"
    assert metrics[1].endpoint == "search"
    assert metrics[1].category == "it"
    assert metrics[1].no_context_count == 1
    assert metrics[1].latest_latency_ms == 150

    ask_metrics = agent_ops_service.get_retrieval_no_context_query_metrics(
        tenant_id="tenant_demo",
        endpoint="ask",
        limit=10,
    )

    assert len(ask_metrics) == 1
    assert ask_metrics[0].query_text == "VPN 怎么配置？"
    assert ask_metrics[0].endpoint == "ask"

    limited_metrics = agent_ops_service.get_retrieval_no_context_query_metrics(
        tenant_id="tenant_demo",
        limit=1,
    )

    assert len(limited_metrics) == 1
    assert limited_metrics[0].query_text == "VPN 怎么配置？"


def test_get_retrieval_failure_metrics(agent_ops_test_engine):
    agent_ops_service.create_retrieval_log(
        RetrievalLogCreate(
            tenant_id="tenant_demo",
            endpoint="ask",
            query_text="VPN 怎么配置？",
            top_k=3,
            category="it",
            retrieval_status="failed",
            total_hits=0,
            latency_ms=100,
            error_message="LLM is unavailable",
        )
    )

    agent_ops_service.create_retrieval_log(
        RetrievalLogCreate(
            tenant_id="tenant_demo",
            endpoint="ask",
            query_text="邮箱怎么登录？",
            top_k=3,
            category="it",
            retrieval_status="failed",
            total_hits=0,
            latency_ms=200,
            error_message="LLM is unavailable",
        )
    )

    agent_ops_service.create_retrieval_log(
        RetrievalLogCreate(
            tenant_id="tenant_demo",
            endpoint="search",
            query_text="打印机驱动怎么下载？",
            top_k=3,
            category="it",
            retrieval_status="failed",
            total_hits=0,
            latency_ms=150,
            error_message="Chroma search failed",
        )
    )

    agent_ops_service.create_retrieval_log(
        RetrievalLogCreate(
            tenant_id="tenant_demo",
            endpoint="ask",
            query_text="VPN 怎么配置？",
            top_k=3,
            category="it",
            retrieval_status="no_context",
            total_hits=0,
            latency_ms=80,
        )
    )

    agent_ops_service.create_retrieval_log(
        RetrievalLogCreate(
            tenant_id="other_tenant",
            endpoint="ask",
            query_text="VPN 怎么配置？",
            top_k=3,
            category="it",
            retrieval_status="failed",
            total_hits=0,
            latency_ms=50,
            error_message="LLM is unavailable",
        )
    )

    metrics = agent_ops_service.get_retrieval_failure_metrics(
        tenant_id="tenant_demo",
        limit=10,
    )

    assert len(metrics) == 2

    assert metrics[0].error_message == "LLM is unavailable"
    assert metrics[0].endpoint == "ask"
    assert metrics[0].category == "it"
    assert metrics[0].failed_count == 2
    assert metrics[0].latest_latency_ms == 200

    assert metrics[1].error_message == "Chroma search failed"
    assert metrics[1].endpoint == "search"
    assert metrics[1].category == "it"
    assert metrics[1].failed_count == 1
    assert metrics[1].latest_latency_ms == 150

    ask_metrics = agent_ops_service.get_retrieval_failure_metrics(
        tenant_id="tenant_demo",
        endpoint="ask",
        limit=10,
    )

    assert len(ask_metrics) == 1
    assert ask_metrics[0].error_message == "LLM is unavailable"
    assert ask_metrics[0].endpoint == "ask"

    limited_metrics = agent_ops_service.get_retrieval_failure_metrics(
        tenant_id="tenant_demo",
        limit=1,
    )

    assert len(limited_metrics) == 1
    assert limited_metrics[0].error_message == "LLM is unavailable"