from datetime import datetime
from types import SimpleNamespace
import json
import pytest
from fastapi import HTTPException

from schemas.agent_ticket import (
    TicketAgentConfirmRequest,
    TicketAgentPreviewRequest,
    TicketDraft,
)
from services.ticket_agent_service import (
    confirm_ticket,
    infer_priority,
    normalize_rag_category,
    preview_ticket,
)


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


def make_ticket_draft(
    title: str = "VPN 连不上",
    description: str = "用户远程办公时 VPN 连不上。",
    category: str = "it",
    priority: str = "high",
) -> TicketDraft:
    return TicketDraft(
        title=title,
        description=description,
        category=category,
        priority=priority,
    )


def make_ticket_draft_json(**overrides) -> str:
    return json.dumps(
        make_ticket_draft(**overrides).model_dump(),
        ensure_ascii=False,
    )


def fake_create_agent_run(agent_run_create):
    return SimpleNamespace(
        id=1,
        tenant_id=agent_run_create.tenant_id,
        user_id=agent_run_create.user_id,
        agent_name=agent_run_create.agent_name,
        input_message=agent_run_create.input_message,
        category=agent_run_create.category,
        status=agent_run_create.status,
        result_summary=agent_run_create.result_summary,
        latency_ms=agent_run_create.latency_ms,
        retrieval_summary_json=agent_run_create.retrieval_summary_json,
    )

def fake_update_agent_run(agent_run_id, tenant_id, agent_run_update):
    return SimpleNamespace(
        id=agent_run_id,
        tenant_id=tenant_id,
        status=agent_run_update.status,
        result_summary=agent_run_update.result_summary,
        latency_ms=agent_run_update.latency_ms,
        retrieval_summary_json=agent_run_update.retrieval_summary_json,
    )

def fake_create_approval_request(approval_request_create):
    return SimpleNamespace(
        id=10,
        agent_run_id=approval_request_create.agent_run_id,
        tenant_id=approval_request_create.tenant_id,
        approval_type=approval_request_create.approval_type,
        status=approval_request_create.status,
        draft_json=approval_request_create.draft_json,
        approved_by=approval_request_create.approved_by,
    )

def fake_create_tool_call(tool_call_create):
    tool_call_ids = {
        "search_kb": 200,
        "classify_ticket": 201,
        "create_ticket": 300,
    }

    return SimpleNamespace(
        id=tool_call_ids.get(tool_call_create.tool_name, 999),
        agent_run_id=tool_call_create.agent_run_id,
        tenant_id=tool_call_create.tenant_id,
        tool_name=tool_call_create.tool_name,
        tool_input_json=tool_call_create.tool_input_json,
        status=tool_call_create.status,
    )


def fake_update_tool_call(tool_call_id, tenant_id, tool_call_update):
    return SimpleNamespace(
        id=tool_call_id,
        tenant_id=tenant_id,
        status=tool_call_update.status,
        tool_output_json=tool_call_update.tool_output_json,
        error_message=tool_call_update.error_message,
    )


def test_normalize_rag_category():
    """
    确认 other 不会被传给 RAG filter，而是直接返回 None，表示不使用 category 过滤 RAG 来源。
    """
    assert normalize_rag_category("it") == "it"
    assert normalize_rag_category("hr") == "hr"
    assert normalize_rag_category("finance") == "finance"
    assert normalize_rag_category("admin") == "admin"
    assert normalize_rag_category("security") == "security"
    assert normalize_rag_category("other") is None
    assert normalize_rag_category(None) is None


def test_infer_priority_client_keyword_should_not_be_urgent():
    """
    防止“客户端”误命中“客户”导致优先级被误判为 urgent。
    """
    priority = infer_priority("VPN 连不上，重启客户端也没用")

    assert priority == "high"


def test_infer_priority_production_incident_should_be_urgent():      
    """
    确认生产系统大面积无法访问的情况应被标记为 urgent。
    """
    priority = infer_priority("生产系统大面积无法访问，客户现场已经阻塞")

    assert priority == "urgent"


def test_preview_ticket_returns_high_priority_for_common_vpn_issue(monkeypatch):
    """
    普通 VPN 故障应是 high，并记录 search_kb / classify_ticket 两个 preview tool_call。
    """
    calls = {}

    def fake_update_agent_run(agent_run_id, tenant_id, agent_run_update):
        calls["updated_agent_run_id"] = agent_run_id
        calls["updated_agent_tenant_id"] = tenant_id
        calls["agent_run_status"] = agent_run_update.status
        calls["agent_result_summary"] = agent_run_update.result_summary
        calls["latency_ms"] = agent_run_update.latency_ms
        calls["retrieval_summary_json"] = agent_run_update.retrieval_summary_json

        return SimpleNamespace(
            id=agent_run_id,
            tenant_id=tenant_id,
            status=agent_run_update.status,
            result_summary=agent_run_update.result_summary,
            latency_ms=agent_run_update.latency_ms,
            retrieval_summary_json=agent_run_update.retrieval_summary_json,
        )

    def fake_create_tool_call_for_preview(tool_call_create):
        if tool_call_create.tool_name == "search_kb":
            calls["search_tool_agent_run_id"] = tool_call_create.agent_run_id
            calls["search_tool_tenant_id"] = tool_call_create.tenant_id
            calls["search_tool_name"] = tool_call_create.tool_name
            calls["search_tool_input_json"] = tool_call_create.tool_input_json
            calls["search_tool_status"] = tool_call_create.status
            tool_call_id = 200

        elif tool_call_create.tool_name == "classify_ticket":
            calls["classify_tool_agent_run_id"] = tool_call_create.agent_run_id
            calls["classify_tool_tenant_id"] = tool_call_create.tenant_id
            calls["classify_tool_name"] = tool_call_create.tool_name
            calls["classify_tool_input_json"] = tool_call_create.tool_input_json
            calls["classify_tool_status"] = tool_call_create.status
            tool_call_id = 201

        else:
            tool_call_id = 999

        return SimpleNamespace(
            id=tool_call_id,
            agent_run_id=tool_call_create.agent_run_id,
            tenant_id=tool_call_create.tenant_id,
            tool_name=tool_call_create.tool_name,
            tool_input_json=tool_call_create.tool_input_json,
            status=tool_call_create.status,
        )

    def fake_update_tool_call_for_preview(tool_call_id, tenant_id, tool_call_update):
        if tool_call_id == 200:
            calls["updated_search_tool_call_id"] = tool_call_id
            calls["updated_search_tool_tenant_id"] = tenant_id
            calls["updated_search_tool_status"] = tool_call_update.status
            calls["search_tool_output_json"] = tool_call_update.tool_output_json

        elif tool_call_id == 201:
            calls["updated_classify_tool_call_id"] = tool_call_id
            calls["updated_classify_tool_tenant_id"] = tenant_id
            calls["updated_classify_tool_status"] = tool_call_update.status
            calls["classify_tool_output_json"] = tool_call_update.tool_output_json

        return SimpleNamespace(
            id=tool_call_id,
            tenant_id=tenant_id,
            status=tool_call_update.status,
            tool_output_json=tool_call_update.tool_output_json,
            error_message=tool_call_update.error_message,
        )

    def fake_search_chroma(query: str, top_k: int, tenant_id: str, category: str | None):
        calls["query"] = query
        calls["top_k"] = top_k
        calls["tenant_id"] = tenant_id
        calls["category"] = category
        return [
            make_search_result(
                document_id="doc_vpn_guide",
                chunk_id="doc_vpn_guide_chunk_1",
                title="vpn_guide",
                category="it",
            )
        ]

    monkeypatch.setattr(
        "services.ticket_agent_service.search_chroma",
        fake_search_chroma,
    )

    monkeypatch.setattr(
        "services.ticket_agent_service.update_agent_run",
        fake_update_agent_run,
    )

    monkeypatch.setattr(
        "services.ticket_agent_service.create_agent_run",
        fake_create_agent_run,
    )

    monkeypatch.setattr(
        "services.ticket_agent_service.create_approval_request",
        fake_create_approval_request,
    )

    monkeypatch.setattr(
        "services.ticket_agent_service.create_tool_call",
        fake_create_tool_call_for_preview,
    )

    monkeypatch.setattr(
        "services.ticket_agent_service.update_tool_call",
        fake_update_tool_call_for_preview,
    )

    response = preview_ticket(
        request=TicketAgentPreviewRequest(
            message="VPN 连不上，重启客户端也没用",
            category="it",
        ),
        tenant_id="tenant_demo",
    )

    assert response.should_create_ticket is True
    assert response.draft is not None
    assert response.draft.category == "it"
    assert response.draft.priority == "high"
    assert response.sources[0].document_id == "doc_vpn_guide"

    assert calls["query"] == "VPN 连不上，重启客户端也没用"
    assert calls["tenant_id"] == "tenant_demo"
    assert calls["category"] == "it"

    assert calls["agent_run_status"] == "completed"
    assert calls["latency_ms"] is not None
    assert calls["latency_ms"] >= 0

    retrieval_summary = json.loads(calls["retrieval_summary_json"])

    assert retrieval_summary["top_k"] == 3
    assert retrieval_summary["request_category"] == "it"
    assert retrieval_summary["rag_category"] == "it"
    assert retrieval_summary["sources_count"] == 1
    assert retrieval_summary["top_distance"] == 0.52
    assert retrieval_summary["source_document_ids"] == ["doc_vpn_guide"]

    assert calls["search_tool_agent_run_id"] == 1
    assert calls["search_tool_tenant_id"] == "tenant_demo"
    assert calls["search_tool_name"] == "search_kb"
    assert calls["search_tool_status"] == "pending"
    assert calls["updated_search_tool_call_id"] == 200
    assert calls["updated_search_tool_status"] == "success"

    search_input = json.loads(calls["search_tool_input_json"])
    assert search_input["query"] == "VPN 连不上，重启客户端也没用"
    assert search_input["top_k"] == 3
    assert search_input["tenant_id"] == "tenant_demo"
    assert search_input["category"] == "it"

    search_output = json.loads(calls["search_tool_output_json"])
    assert search_output["results_count"] == 1
    assert search_output["document_ids"] == ["doc_vpn_guide"]
    assert search_output["top_distance"] == 0.52

    assert calls["classify_tool_agent_run_id"] == 1
    assert calls["classify_tool_tenant_id"] == "tenant_demo"
    assert calls["classify_tool_name"] == "classify_ticket"
    assert calls["classify_tool_status"] == "pending"
    assert calls["updated_classify_tool_call_id"] == 201
    assert calls["updated_classify_tool_status"] == "success"

    classify_input = json.loads(calls["classify_tool_input_json"])
    assert classify_input["message"] == "VPN 连不上，重启客户端也没用"
    assert classify_input["requested_category"] == "it"
    assert classify_input["sources_count"] == 1
    assert classify_input["source_categories"] == ["it"]
    assert classify_input["source_document_ids"] == ["doc_vpn_guide"]

    classify_output = json.loads(calls["classify_tool_output_json"])
    assert classify_output["should_create_ticket"] is True
    assert classify_output["category"] == "it"
    assert classify_output["priority"] == "high"
    assert classify_output["reason"]


def test_preview_ticket_with_other_category_does_not_filter_rag_by_other(monkeypatch):
    """
    当用户选择的 category 是 other 时，不将 other 传给 RAG 过滤器，而是直接不使用 category 过滤。
    """
    calls = {}

    def fake_search_chroma(query: str, top_k: int, tenant_id: str, category: str | None):
        calls["query"] = query
        calls["tenant_id"] = tenant_id
        calls["category"] = category
        return []

    monkeypatch.setattr(
        "services.ticket_agent_service.search_chroma",
        fake_search_chroma,
    )

    monkeypatch.setattr(
        "services.ticket_agent_service.create_agent_run",
        fake_create_agent_run,
    )

    monkeypatch.setattr(
        "services.ticket_agent_service.update_agent_run",
        fake_update_agent_run,
    )

    monkeypatch.setattr(
        "services.ticket_agent_service.create_approval_request",
        fake_create_approval_request,
    )

    monkeypatch.setattr(
        "services.ticket_agent_service.create_tool_call",
        fake_create_tool_call,
    )

    monkeypatch.setattr(
        "services.ticket_agent_service.update_tool_call",
        fake_update_tool_call,
    )

    response = preview_ticket(
        request=TicketAgentPreviewRequest(
            message="不知道这个问题应该找哪个部门处理",
            category="other",
        ),
        tenant_id="tenant_demo",
    )

    assert calls["category"] is None
    assert response.should_create_ticket is True
    assert response.draft is not None
    assert response.draft.category == "other"
    assert response.sources == []


def test_preview_ticket_infers_category_from_top_source_when_category_missing(monkeypatch):
    """
    用户不传 category 时可从 source 推断
    """
    def fake_search_chroma(query: str, top_k: int, tenant_id: str, category: str | None):
        return [
            make_search_result(
                document_id="doc_data_access_policy",
                chunk_id="doc_data_access_policy_chunk_1",
                title="data_access_policy",
                category="security",
                content="数据访问权限申请说明。",
            )
        ]

    monkeypatch.setattr(
        "services.ticket_agent_service.search_chroma",
        fake_search_chroma,
    )

    monkeypatch.setattr(
        "services.ticket_agent_service.create_agent_run",
        fake_create_agent_run,
    )

    monkeypatch.setattr(
        "services.ticket_agent_service.update_agent_run",
        fake_update_agent_run,
    )

    monkeypatch.setattr(
        "services.ticket_agent_service.create_approval_request",
        fake_create_approval_request,
    )

    monkeypatch.setattr(
        "services.ticket_agent_service.create_tool_call",
        fake_create_tool_call,
    )

    monkeypatch.setattr(
        "services.ticket_agent_service.update_tool_call",
        fake_update_tool_call,
    )

    response = preview_ticket(
        request=TicketAgentPreviewRequest(
            message="申请客户数据访问权限失败了",
            category=None,
        ),
        tenant_id="tenant_demo",
    )

    assert response.should_create_ticket is True
    assert response.draft is not None
    assert response.draft.category == "security"
    assert response.draft.priority == "high"
    assert response.sources[0].category == "security"


def test_preview_ticket_can_return_no_ticket_for_knowledge_question(monkeypatch):
    """
    知识咨询不创建工单
    """
    def fake_search_chroma(query: str, top_k: int, tenant_id: str, category: str | None):
        return [
            make_search_result(
                document_id="doc_leave_policy",
                chunk_id="doc_leave_policy_chunk_1",
                title="leave_policy",
                category="hr",
                content="请假需要在系统中提交申请。",
            )
        ]

    monkeypatch.setattr(
        "services.ticket_agent_service.search_chroma",
        fake_search_chroma,
    )

    monkeypatch.setattr(
        "services.ticket_agent_service.create_agent_run",
        fake_create_agent_run,
    )

    monkeypatch.setattr(
        "services.ticket_agent_service.update_agent_run",
        fake_update_agent_run,
    )

    monkeypatch.setattr(
        "services.ticket_agent_service.create_approval_request",
        fake_create_approval_request,
    )

    monkeypatch.setattr(
        "services.ticket_agent_service.create_tool_call",
        fake_create_tool_call,
    )

    monkeypatch.setattr(
        "services.ticket_agent_service.update_tool_call",
        fake_update_tool_call,
    )

    response = preview_ticket(
        request=TicketAgentPreviewRequest(
            message="请问请假怎么申请？",
            category="hr",
        ),
        tenant_id="tenant_demo",
    )

    assert response.should_create_ticket is False
    assert response.draft is None
    assert response.sources[0].document_id == "doc_leave_policy"


def test_confirm_ticket_calls_create_ticket_service(monkeypatch):
    """
    confirm 复用 Ticket Service 创建真实工单，并记录 AgentOps 轨迹
    """
    calls = {}

    def fake_get_approval_request(approval_request_id, tenant_id):
        calls["get_approval_request_id"] = approval_request_id
        calls["get_approval_tenant_id"] = tenant_id
        return SimpleNamespace(
            id=approval_request_id,
            agent_run_id=1,
            tenant_id=tenant_id,
            status="pending",
            draft_json=make_ticket_draft_json(),
        )

    def fake_update_approval_request(approval_request_id, tenant_id, approval_request_update):
        calls["approval_request_id"] = approval_request_id
        calls["approval_tenant_id"] = tenant_id
        calls["approval_status"] = approval_request_update.status
        calls["approved_by"] = approval_request_update.approved_by
        return SimpleNamespace(
            id=approval_request_id,
            tenant_id=tenant_id,
            status=approval_request_update.status,
            approved_by=approval_request_update.approved_by,
        )

    def fake_create_tool_call(tool_call_create):
        calls["tool_agent_run_id"] = tool_call_create.agent_run_id
        calls["tool_tenant_id"] = tool_call_create.tenant_id
        calls["tool_name"] = tool_call_create.tool_name
        calls["tool_input_json"] = tool_call_create.tool_input_json
        calls["tool_status"] = tool_call_create.status
        return SimpleNamespace(
            id=300,
            agent_run_id=tool_call_create.agent_run_id,
            tenant_id=tool_call_create.tenant_id,
            tool_name=tool_call_create.tool_name,
            status=tool_call_create.status,
        )

    def fake_update_tool_call(tool_call_id, tenant_id, tool_call_update):
        calls["updated_tool_call_id"] = tool_call_id
        calls["updated_tool_tenant_id"] = tenant_id
        calls["updated_tool_status"] = tool_call_update.status
        calls["tool_output_json"] = tool_call_update.tool_output_json
        return SimpleNamespace(
            id=tool_call_id,
            tenant_id=tenant_id,
            status=tool_call_update.status,
            tool_output_json=tool_call_update.tool_output_json,
        )

    def fake_update_agent_run(agent_run_id, tenant_id, agent_run_update):
        calls["updated_agent_run_id"] = agent_run_id
        calls["updated_agent_tenant_id"] = tenant_id
        calls["agent_run_status"] = agent_run_update.status
        calls["agent_result_summary"] = agent_run_update.result_summary
        return SimpleNamespace(
            id=agent_run_id,
            tenant_id=tenant_id,
            status=agent_run_update.status,
            result_summary=agent_run_update.result_summary,
        )

    def fake_create_ticket_service(ticket_create, tenant_id: str, created_by: str):
        calls["title"] = ticket_create.title
        calls["description"] = ticket_create.description
        calls["category"] = ticket_create.category
        calls["priority"] = ticket_create.priority
        calls["tenant_id"] = tenant_id
        calls["created_by"] = created_by

        return SimpleNamespace(
            id=100,
            tenant_id=tenant_id,
            created_by=created_by,
            title=ticket_create.title,
            description=ticket_create.description,
            category=ticket_create.category,
            priority=ticket_create.priority,
            status="open",
            created_at=datetime(2026, 1, 1, 10, 0, 0),
            updated_at=datetime(2026, 1, 1, 10, 0, 0),
        )
    
    monkeypatch.setattr(
        "services.ticket_agent_service.get_approval_request",
        fake_get_approval_request,
    )
    monkeypatch.setattr(
        "services.ticket_agent_service.update_approval_request",
        fake_update_approval_request,
    )
    monkeypatch.setattr(
        "services.ticket_agent_service.create_tool_call",
        fake_create_tool_call,
    )
    monkeypatch.setattr(
        "services.ticket_agent_service.update_tool_call",
        fake_update_tool_call,
    )
    monkeypatch.setattr(
        "services.ticket_agent_service.update_agent_run",
        fake_update_agent_run,
    )
    monkeypatch.setattr(
        "services.ticket_agent_service.create_ticket_service",
        fake_create_ticket_service,
    )

    response = confirm_ticket(
        request=TicketAgentConfirmRequest(
            agent_run_id=1,
            approval_request_id=10,
            draft=TicketDraft(
                title="VPN 连不上",
                description="用户远程办公时 VPN 连不上。",
                category="it",
                priority="high",
            ),
        ),
        tenant_id="tenant_demo",
        created_by="user_demo",
    )

    assert response.agent_run_id == 1
    assert response.approval_request_id == 10
    assert response.tool_call_id == 300
    assert response.ticket.id == 100
    assert response.ticket.tenant_id == "tenant_demo"
    assert response.ticket.created_by == "user_demo"
    assert response.ticket.title == "VPN 连不上"
    assert response.ticket.category == "it"
    assert response.ticket.priority == "high"
    assert response.ticket.status == "open"

    assert calls["get_approval_request_id"] == 10
    assert calls["get_approval_tenant_id"] == "tenant_demo"
    assert calls["approval_request_id"] == 10
    assert calls["approval_status"] == "approved"
    assert calls["approved_by"] == "user_demo"

    assert calls["tool_agent_run_id"] == 1
    assert calls["tool_name"] == "create_ticket"
    assert calls["tool_status"] == "pending"
    assert calls["updated_tool_call_id"] == 300
    assert calls["updated_tool_status"] == "success"

    assert calls["updated_agent_run_id"] == 1
    assert calls["agent_run_status"] == "completed"

    assert calls["title"] == "VPN 连不上"
    assert calls["description"] == "用户远程办公时 VPN 连不上。"
    assert calls["category"] == "it"
    assert calls["priority"] == "high"
    assert calls["tenant_id"] == "tenant_demo"
    assert calls["created_by"] == "user_demo"

def test_confirm_ticket_rejects_mismatched_approval_request(monkeypatch):
    calls = {}

    def fake_get_approval_request(approval_request_id, tenant_id):
        calls["get_approval_request_id"] = approval_request_id
        calls["get_approval_tenant_id"] = tenant_id
        return SimpleNamespace(
            id=approval_request_id,
            agent_run_id=999,
            tenant_id=tenant_id,
            status="pending",
            draft_json=make_ticket_draft_json(),
        )

    def fake_update_approval_request(*args, **kwargs):
        calls["update_approval_request_called"] = True

    def fake_create_tool_call(*args, **kwargs):
        calls["create_tool_call_called"] = True

    def fake_create_ticket_service(*args, **kwargs):
        calls["create_ticket_service_called"] = True

    monkeypatch.setattr(
        "services.ticket_agent_service.get_approval_request",
        fake_get_approval_request,
    )
    monkeypatch.setattr(
        "services.ticket_agent_service.update_approval_request",
        fake_update_approval_request,
    )
    monkeypatch.setattr(
        "services.ticket_agent_service.create_tool_call",
        fake_create_tool_call,
    )
    monkeypatch.setattr(
        "services.ticket_agent_service.create_ticket_service",
        fake_create_ticket_service,
    )

    with pytest.raises(HTTPException) as exc_info:
        confirm_ticket(
            request=TicketAgentConfirmRequest(
                agent_run_id=1,
                approval_request_id=10,
                draft=TicketDraft(
                    title="VPN 连不上",
                    description="用户远程办公时 VPN 连不上。",
                    category="it",
                    priority="high",
                ),
            ),
            tenant_id="tenant_demo",
            created_by="user_demo",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Approval request does not belong to agent run"

    assert calls["get_approval_request_id"] == 10
    assert calls["get_approval_tenant_id"] == "tenant_demo"
    assert "update_approval_request_called" not in calls
    assert "create_tool_call_called" not in calls
    assert "create_ticket_service_called" not in calls

@pytest.mark.parametrize("approval_status", ["rejected", "cancelled", "approved"])
def test_confirm_ticket_rejects_non_pending_approval_request(
    monkeypatch,
    approval_status,
):
    calls = {}

    def fake_get_approval_request(approval_request_id, tenant_id):
        return SimpleNamespace(
            id=approval_request_id,
            agent_run_id=1,
            tenant_id=tenant_id,
            status=approval_status,
            draft_json=make_ticket_draft_json(),
        )

    def fake_update_approval_request(*args, **kwargs):
        calls["update_approval_request_called"] = True

    def fake_create_tool_call(*args, **kwargs):
        calls["create_tool_call_called"] = True

    def fake_create_ticket_service(*args, **kwargs):
        calls["create_ticket_service_called"] = True

    monkeypatch.setattr(
        "services.ticket_agent_service.get_approval_request",
        fake_get_approval_request,
    )
    monkeypatch.setattr(
        "services.ticket_agent_service.update_approval_request",
        fake_update_approval_request,
    )
    monkeypatch.setattr(
        "services.ticket_agent_service.create_tool_call",
        fake_create_tool_call,
    )
    monkeypatch.setattr(
        "services.ticket_agent_service.create_ticket_service",
        fake_create_ticket_service,
    )

    with pytest.raises(HTTPException) as exc_info:
        confirm_ticket(
            request=TicketAgentConfirmRequest(
                agent_run_id=1,
                approval_request_id=10,
                draft=make_ticket_draft(),
            ),
            tenant_id="tenant_demo",
            created_by="user_demo",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Approval request is not pending"
    assert "update_approval_request_called" not in calls
    assert "create_tool_call_called" not in calls
    assert "create_ticket_service_called" not in calls

def test_confirm_ticket_rejects_tampered_draft(monkeypatch):
    calls = {}

    def fake_get_approval_request(approval_request_id, tenant_id):
        return SimpleNamespace(
            id=approval_request_id,
            agent_run_id=1,
            tenant_id=tenant_id,
            status="pending",
            draft_json=make_ticket_draft_json(
                title="VPN 连不上",
                category="it",
                priority="high",
            ),
        )

    def fake_update_approval_request(*args, **kwargs):
        calls["update_approval_request_called"] = True

    def fake_create_tool_call(*args, **kwargs):
        calls["create_tool_call_called"] = True

    def fake_create_ticket_service(*args, **kwargs):
        calls["create_ticket_service_called"] = True

    monkeypatch.setattr(
        "services.ticket_agent_service.get_approval_request",
        fake_get_approval_request,
    )
    monkeypatch.setattr(
        "services.ticket_agent_service.update_approval_request",
        fake_update_approval_request,
    )
    monkeypatch.setattr(
        "services.ticket_agent_service.create_tool_call",
        fake_create_tool_call,
    )
    monkeypatch.setattr(
        "services.ticket_agent_service.create_ticket_service",
        fake_create_ticket_service,
    )

    with pytest.raises(HTTPException) as exc_info:
        confirm_ticket(
            request=TicketAgentConfirmRequest(
                agent_run_id=1,
                approval_request_id=10,
                draft=make_ticket_draft(title="被篡改的标题"),
            ),
            tenant_id="tenant_demo",
            created_by="user_demo",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Confirm draft does not match approval draft"
    assert "update_approval_request_called" not in calls
    assert "create_tool_call_called" not in calls
    assert "create_ticket_service_called" not in calls


def test_preview_ticket_marks_search_kb_tool_call_failed_when_search_fails(monkeypatch):
    calls = {}

    def fake_search_chroma(query: str, top_k: int, tenant_id: str, category: str | None):
        raise RuntimeError("chroma unavailable")

    def fake_create_tool_call_for_search(tool_call_create):
        calls["search_tool_name"] = tool_call_create.tool_name
        calls["search_tool_status"] = tool_call_create.status
        calls["search_tool_input_json"] = tool_call_create.tool_input_json

        return SimpleNamespace(
            id=200,
            agent_run_id=tool_call_create.agent_run_id,
            tenant_id=tool_call_create.tenant_id,
            tool_name=tool_call_create.tool_name,
            status=tool_call_create.status,
        )

    def fake_update_tool_call_for_search(tool_call_id, tenant_id, tool_call_update):
        calls["updated_search_tool_call_id"] = tool_call_id
        calls["updated_search_tool_status"] = tool_call_update.status
        calls["search_tool_error_message"] = tool_call_update.error_message

        return SimpleNamespace(
            id=tool_call_id,
            tenant_id=tenant_id,
            status=tool_call_update.status,
            error_message=tool_call_update.error_message,
        )

    def fake_update_agent_run_for_failure(agent_run_id, tenant_id, agent_run_update):
        calls["failed_agent_run_id"] = agent_run_id
        calls["failed_agent_run_status"] = agent_run_update.status
        calls["failed_agent_result_summary"] = agent_run_update.result_summary

        return SimpleNamespace(
            id=agent_run_id,
            tenant_id=tenant_id,
            status=agent_run_update.status,
            result_summary=agent_run_update.result_summary,
            latency_ms=agent_run_update.latency_ms,
            retrieval_summary_json=agent_run_update.retrieval_summary_json,
        )

    monkeypatch.setattr(
        "services.ticket_agent_service.search_chroma",
        fake_search_chroma,
    )
    monkeypatch.setattr(
        "services.ticket_agent_service.create_agent_run",
        fake_create_agent_run,
    )
    monkeypatch.setattr(
        "services.ticket_agent_service.update_agent_run",
        fake_update_agent_run_for_failure,
    )
    monkeypatch.setattr(
        "services.ticket_agent_service.create_tool_call",
        fake_create_tool_call_for_search,
    )
    monkeypatch.setattr(
        "services.ticket_agent_service.update_tool_call",
        fake_update_tool_call_for_search,
    )

    with pytest.raises(RuntimeError) as exc_info:
        preview_ticket(
            request=TicketAgentPreviewRequest(
                message="VPN 连不上",
                category="it",
            ),
            tenant_id="tenant_demo",
        )

    assert str(exc_info.value) == "chroma unavailable"

    assert calls["search_tool_name"] == "search_kb"
    assert calls["search_tool_status"] == "pending"
    assert calls["updated_search_tool_call_id"] == 200
    assert calls["updated_search_tool_status"] == "failed"
    assert calls["search_tool_error_message"] == "chroma unavailable"

    assert calls["failed_agent_run_id"] == 1
    assert calls["failed_agent_run_status"] == "failed"
    assert calls["failed_agent_result_summary"] == "search_kb failed: chroma unavailable"