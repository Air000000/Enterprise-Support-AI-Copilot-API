from __future__ import annotations
from typing import cast
import json
import time

from fastapi import HTTPException
from experiments.rag_local.query_chroma import search_chroma
from schemas.agent_ticket import (
    TicketAgentConfirmRequest,
    TicketAgentConfirmResponse,
    TicketAgentPreviewRequest,
    TicketAgentPreviewResponse,
    TicketAgentSource,
    TicketDraft,
)
from schemas.ticket import TicketCategory, TicketCreate, TicketPriority, TicketResponse
from services.ticket_service import create_ticket as create_ticket_service
from schemas.agent_ops import (
    AgentRunCreate,
    AgentRunUpdate,
    ApprovalRequestCreate,
    ApprovalRequestUpdate,
    ToolCallCreate,
    ToolCallUpdate,
)
from services.agent_ops_service import (
    create_agent_run,
    create_approval_request,
    create_tool_call,
    get_approval_request,
    update_agent_run,
    update_approval_request,
    update_tool_call,
)


SEARCHABLE_CATEGORIES = {
    "it",
    "hr",
    "finance",
    "admin",
    "security",
}

SUPPORT_NEEDED_KEYWORDS = [
    "无法",
    "不能",
    "失败",
    "报错",
    "异常",
    "打不开",
    "连接不上",
    "连不上",
    "收不到",
    "锁定",
    "忘记密码",
    "退回",
    "损坏",
    "丢失",
    "过期",
    "开通",
    "审批",
    "权限",
]

URGENT_KEYWORDS = [
    "紧急",
    "生产系统",
    "全员",
    "大面积",
    "客户现场",
    "客户阻塞",
    "无法办公",
    "数据泄露",
    "安全事件",
]

LOW_PRIORITY_KEYWORDS = [
    "咨询",
    "了解",
    "请问",
    "怎么",
    "如何",
    "是否",
]


def normalize_rag_category(
    category: TicketCategory | None,
) -> str | None:
    """
    将用户传入的 ticket category 转换为 RAG 检索 category。

    - it/hr/finance/admin/security: 用于 Chroma category filter
    - other/None: 不使用 category filter，避免搜索不存在的 other 文档目录
    """
    if category is None:
        return None

    if category == "other":
        return None

    return category


def make_preview(text: str, max_length: int = 240) -> str:
    cleaned = " ".join(text.split())

    if len(cleaned) <= max_length:
        return cleaned

    return cleaned[:max_length] + "..."


def to_agent_source(search_result) -> TicketAgentSource:
    content = getattr(search_result, "content", "")

    return TicketAgentSource(
        document_id=search_result.document_id,
        chunk_id=search_result.chunk_id,
        title=search_result.title,
        source_path=search_result.source_path,
        distance=round(search_result.distance, 4),
        preview=make_preview(content),
        category=getattr(search_result, "category", None),
    )


def calculate_latency_ms(start_time: float) -> int:
    return int((time.perf_counter() - start_time) * 1000)


def build_retrieval_summary_json(
    top_k: int,
    request_category: TicketCategory | None,
    rag_category: str | None,
    sources: list[TicketAgentSource],
) -> str:
    top_distance = None

    if sources:
        top_distance = sources[0].distance

    summary = {
        "top_k": top_k,
        "request_category": request_category,
        "rag_category": rag_category,
        "sources_count": len(sources),
        "top_distance": top_distance,
        "source_document_ids": [
            source.document_id
            for source in sources
        ],
    }

    return json.dumps(summary, ensure_ascii=False)


def infer_ticket_category(
    request_category: TicketCategory | None,
    sources: list[TicketAgentSource],
) -> TicketCategory:
    """
    推断 TicketDraft 的 category。

    优先级：
    1. 用户显式传入 category
    2. 使用 top source 的 category
    3. fallback 到 other
    """
    if request_category is not None:
        return request_category

    for source in sources:
        if source.category in SEARCHABLE_CATEGORIES:
            return cast(TicketCategory, source.category)

    return "other"


def infer_priority(message: str) -> TicketPriority:
    if any(keyword in message for keyword in URGENT_KEYWORDS):
        return "urgent"

    if any(keyword in message for keyword in SUPPORT_NEEDED_KEYWORDS):
        return "high"

    if any(keyword in message for keyword in LOW_PRIORITY_KEYWORDS):
        return "low"

    return "medium"


def should_create_ticket(
    message: str,
    sources: list[TicketAgentSource],
) -> bool:
    """
    第一版使用规则判断是否建议创建工单。

    规则：
    - 没有检索到知识来源：建议创建工单
    - 用户描述包含故障、异常、申请、权限等支持类关键词：建议创建工单
    - 普通知识咨询：暂不建议创建工单
    """
    if not sources:
        return True

    return any(keyword in message for keyword in SUPPORT_NEEDED_KEYWORDS)


def build_ticket_title(message: str, max_length: int = 80) -> str:
    cleaned = " ".join(message.split())

    if len(cleaned) <= max_length:
        return cleaned

    return cleaned[:max_length] + "..."


def build_ticket_description(
    message: str,
    sources: list[TicketAgentSource],
) -> str:
    source_lines = []

    for index, source in enumerate(sources, start=1):
        source_lines.append(
            f"{index}. {source.title} ({source.document_id}, {source.chunk_id})"
        )

    if source_lines:
        sources_text = "\n".join(source_lines)
    else:
        sources_text = "未检索到明确知识库来源。"

    return (
        f"用户问题：{message}\n\n"
        "系统根据用户描述生成工单草稿，建议支持人员进一步确认和处理。\n\n"
        f"相关知识库来源：\n{sources_text}"
    )


def build_reason(
    should_create: bool,
    sources: list[TicketAgentSource],
) -> str:
    if not sources:
        return "知识库中未检索到明确依据，建议创建工单由人工支持人员处理。"

    if should_create:
        return "用户描述包含故障、异常、申请或权限类信号，建议生成工单草稿，用户确认后创建工单。"

    return "知识库中已检索到相关资料，当前更适合作为知识咨询，暂不建议创建工单。"


def load_ticket_draft_from_approval(approval_request) -> TicketDraft:
    """
    从 approval_request.draft_json 还原 preview 阶段服务端保存的工单草稿。

    confirm 阶段不能直接信任客户端重新传来的 draft。
    真正被审批的对象，应该是 preview 阶段保存下来的 draft_json。
    """
    try:
        draft_payload = json.loads(approval_request.draft_json)
        return TicketDraft.model_validate(draft_payload)
    except (TypeError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Stored approval draft is invalid",
        ) from exc


def ensure_approval_request_is_pending(approval_request) -> None:
    """
    rejected / cancelled / approved 的审批请求不能再次执行。
    """
    if approval_request.status != "pending":
        raise HTTPException(
            status_code=400,
            detail="Approval request is not pending",
        )


def ensure_confirm_draft_matches_approval_draft(
    request_draft: TicketDraft,
    approval_draft: TicketDraft,
) -> None:
    """
    confirm 请求里的 draft 必须和 preview 阶段保存的 draft 一致。

    后续如果要支持用户编辑 draft，要单独设计 edit / override 流程，
    不能在 confirm 阶段静默替换。
    """
    if request_draft.model_dump() != approval_draft.model_dump():
        raise HTTPException(
            status_code=400,
            detail="Confirm draft does not match approval draft",
        )


def preview_ticket(
    request: TicketAgentPreviewRequest,
    tenant_id: str,
    user_id: str = "user_demo",
    top_k: int = 3,
) -> TicketAgentPreviewResponse:
    start_time = time.perf_counter()

    agent_run = create_agent_run(
        AgentRunCreate(
            tenant_id=tenant_id,
            user_id=user_id,
            agent_name="ticket_agent",
            input_message=request.message,
            category=request.category,
            status="running",
        )
    )

    rag_category = normalize_rag_category(request.category)

    search_tool_call = create_tool_call(
        ToolCallCreate(
            agent_run_id=agent_run.id,
            tenant_id=tenant_id,
            tool_name="search_kb",
            tool_input_json=json.dumps(
                {
                    "query": request.message,
                    "top_k": top_k,
                    "tenant_id": tenant_id,
                    "category": rag_category,
                },
                ensure_ascii=False,
            ),
            status="pending",
        )
    )

    try:
        search_results = search_chroma(
            query=request.message,
            top_k=top_k,
            tenant_id=tenant_id,
            category=rag_category,
        )

        update_tool_call(
            tool_call_id=search_tool_call.id,
            tenant_id=tenant_id,
            tool_call_update=ToolCallUpdate(
                status="success",
                tool_output_json=json.dumps(
                    {
                        "results_count": len(search_results),
                        "document_ids": [
                            getattr(result, "document_id", None)
                            for result in search_results
                        ],
                        "top_distance": (
                            getattr(search_results[0], "distance", None)
                            if search_results
                            else None
                        ),
                    },
                    ensure_ascii=False,
                ),
            ),
        )

    except Exception as exc:
        update_tool_call(
            tool_call_id=search_tool_call.id,
            tenant_id=tenant_id,
            tool_call_update=ToolCallUpdate(
                status="failed",
                error_type="search_kb_failed",
                error_message=str(exc),
            ),
        )

        update_agent_run(
            agent_run_id=agent_run.id,
            tenant_id=tenant_id,
            agent_run_update=AgentRunUpdate(
                status="failed",
                result_summary=f"search_kb failed: {exc}",
                latency_ms=calculate_latency_ms(start_time),
            ),
        )

        raise

    sources = [
        to_agent_source(search_result)
        for search_result in search_results
    ]

    classify_tool_call = create_tool_call(
        ToolCallCreate(
            agent_run_id=agent_run.id,
            tenant_id=tenant_id,
            tool_name="classify_ticket",
            tool_input_json=json.dumps(
                {
                    "message": request.message,
                    "requested_category": request.category,
                    "sources_count": len(sources),
                    "source_categories": [
                        source.category
                        for source in sources
                    ],
                    "source_document_ids": [
                        source.document_id
                        for source in sources
                    ],
                },
                ensure_ascii=False,
            ),
            status="pending",
        )
    )

    try:
        retrieval_summary_json = build_retrieval_summary_json(
            top_k=top_k,
            request_category=request.category,
            rag_category=rag_category,
            sources=sources,
        )

        should_create = should_create_ticket(
            message=request.message,
            sources=sources,
        )

        reason = build_reason(
            should_create=should_create,
            sources=sources,
        )

        ticket_category: TicketCategory | None = None
        priority: TicketPriority | None = None
        draft: TicketDraft | None = None

        if should_create:
            ticket_category = infer_ticket_category(
                request_category=request.category,
                sources=sources,
            )

            priority = infer_priority(request.message)

            draft = TicketDraft(
                title=build_ticket_title(request.message),
                description=build_ticket_description(
                    message=request.message,
                    sources=sources,
                ),
                category=ticket_category,
                priority=priority,
            )

        update_tool_call(
            tool_call_id=classify_tool_call.id,
            tenant_id=tenant_id,
            tool_call_update=ToolCallUpdate(
                status="success",
                tool_output_json=json.dumps(
                    {
                        "should_create_ticket": should_create,
                        "category": ticket_category,
                        "priority": priority,
                        "reason": reason,
                    },
                    ensure_ascii=False,
                ),
            ),
        )

    except Exception as exc:
        update_tool_call(
            tool_call_id=classify_tool_call.id,
            tenant_id=tenant_id,
            tool_call_update=ToolCallUpdate(
                status="failed",
                error_type="classify_ticket_failed",
                error_message=str(exc),
            ),
        )

        update_agent_run(
            agent_run_id=agent_run.id,
            tenant_id=tenant_id,
            agent_run_update=AgentRunUpdate(
                status="failed",
                result_summary=f"classify_ticket failed: {exc}",
                latency_ms=calculate_latency_ms(start_time),
            ),
        )

        raise

    approval_request_id: int | None = None

    if should_create and draft is not None:
        approval_request = create_approval_request(
            ApprovalRequestCreate(
                agent_run_id=agent_run.id,
                tenant_id=tenant_id,
                approval_type="ticket_creation",
                status="pending",
                draft_json=json.dumps(
                    draft.model_dump(),
                    ensure_ascii=False,
                ),
            )
        )
        approval_request_id = approval_request.id

    update_agent_run(
        agent_run_id=agent_run.id,
        tenant_id=tenant_id,
        agent_run_update=AgentRunUpdate(
            status="completed",
            result_summary=reason,
            latency_ms=calculate_latency_ms(start_time),
            retrieval_summary_json=retrieval_summary_json,
        ),
    )

    return TicketAgentPreviewResponse(
        agent_run_id=agent_run.id,
        approval_request_id=approval_request_id,
        should_create_ticket=should_create,
        reason=reason,
        draft=draft,
        sources=sources,
    )


def confirm_ticket(
    request: TicketAgentConfirmRequest,
    tenant_id: str,
    created_by: str,
) -> TicketAgentConfirmResponse:
    approval_request = get_approval_request(
        approval_request_id=request.approval_request_id,
        tenant_id=tenant_id,
    )

    if approval_request.agent_run_id != request.agent_run_id:
        raise HTTPException(
            status_code=400,
            detail="Approval request does not belong to agent run",
        )

    ensure_approval_request_is_pending(approval_request)

    approval_draft = load_ticket_draft_from_approval(approval_request)

    ensure_confirm_draft_matches_approval_draft(
        request_draft=request.draft,
        approval_draft=approval_draft,
    )

    update_approval_request(
        approval_request_id=request.approval_request_id,
        tenant_id=tenant_id,
        approval_request_update=ApprovalRequestUpdate(
            status="approved",
            approved_by=created_by,
        ),
    )

    ticket_create = TicketCreate(
        title=approval_draft.title,
        description=approval_draft.description,
        category=approval_draft.category,
        priority=approval_draft.priority,
    )

    tool_call = create_tool_call(
        ToolCallCreate(
            agent_run_id=request.agent_run_id,
            tenant_id=tenant_id,
            tool_name="create_ticket",
            tool_input_json=json.dumps(
                ticket_create.model_dump(),
                ensure_ascii=False,
            ),
            status="pending",
        )
    )

    try:
        ticket = create_ticket_service(
            ticket_create=ticket_create,
            tenant_id=tenant_id,
            created_by=created_by,
        )

        update_tool_call(
            tool_call_id=tool_call.id,
            tenant_id=tenant_id,
            tool_call_update=ToolCallUpdate(
                status="success",
                tool_output_json=json.dumps(
                    {
                        "ticket_id": ticket.id,
                        "status": ticket.status,
                    },
                    ensure_ascii=False,
                ),
            ),
        )

        update_agent_run(
            agent_run_id=request.agent_run_id,
            tenant_id=tenant_id,
            agent_run_update=AgentRunUpdate(
                status="completed",
                result_summary=f"Ticket created: {ticket.id}",
            ),
        )

        return TicketAgentConfirmResponse(
            agent_run_id=request.agent_run_id,
            approval_request_id=request.approval_request_id,
            tool_call_id=tool_call.id,
            ticket=TicketResponse.model_validate(ticket),
        )

    except Exception as exc:
        update_tool_call(
            tool_call_id=tool_call.id,
            tenant_id=tenant_id,
            tool_call_update=ToolCallUpdate(
                status="failed",
                error_type="create_ticket_failed",
                error_message=str(exc),
            ),
        )

        update_agent_run(
            agent_run_id=request.agent_run_id,
            tenant_id=tenant_id,
            agent_run_update=AgentRunUpdate(
                status="failed",
                result_summary=str(exc),
            ),
        )

        raise