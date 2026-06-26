from __future__ import annotations

from datetime import datetime, timezone

from fastapi import HTTPException
from sqlmodel import Session, select

from database import engine
from models.agent_ops import AgentRun, ApprovalRequest, RetrievalLog, ToolCall
from schemas.agent_ops import (
    AgentOpsMetricsSummaryResponse,
    AgentRunCreate,
    AgentRunUpdate,
    ApprovalRequestCreate,
    ApprovalRequestUpdate,
    RetrievalLogCreate,
    RetrievalMetricsSummaryResponse,
    ToolCallCreate,
    ToolCallUpdate,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def create_agent_run(agent_run_create: AgentRunCreate) -> AgentRun:
    """
    创建一个新的 agent 运行记录。
    """
    agent_run = AgentRun(
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

    with Session(engine) as session:
        session.add(agent_run)
        session.commit()
        session.refresh(agent_run)
        return agent_run


def get_agent_run(
    agent_run_id: int,
    tenant_id: str,
) -> AgentRun:
    """
    根据 ID 获取 agent 运行记录，确保它属于指定的 tenant。
    """
    with Session(engine) as session:
        agent_run = session.get(AgentRun, agent_run_id)

        if agent_run is None or agent_run.tenant_id != tenant_id:
            raise HTTPException(status_code=404, detail="Agent run not found")

        return agent_run


def list_agent_runs(
    tenant_id: str,
    status: str | None = None,
    agent_name: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[AgentRun]:
    with Session(engine) as session:
        statement = select(AgentRun).where(AgentRun.tenant_id == tenant_id)

        if status is not None:
            statement = statement.where(AgentRun.status == status)

        if agent_name is not None:
            statement = statement.where(AgentRun.agent_name == agent_name)

        statement = (
            statement
            .order_by(AgentRun.id.desc())
            .offset(offset)
            .limit(limit)
        )
        agent_runs = session.exec(statement).all()
        return list(agent_runs)


def update_agent_run(
    agent_run_id: int,
    tenant_id: str,
    agent_run_update: AgentRunUpdate,
) -> AgentRun:
    with Session(engine) as session:
        agent_run = session.get(AgentRun, agent_run_id)

        if agent_run is None or agent_run.tenant_id != tenant_id:
            raise HTTPException(status_code=404, detail="Agent run not found")

        update_data = agent_run_update.model_dump(exclude_unset=True)

        for field_name, field_value in update_data.items():
            setattr(agent_run, field_name, field_value)

        agent_run.updated_at = utc_now()

        session.add(agent_run)
        session.commit()
        session.refresh(agent_run)

        return agent_run


def create_tool_call(tool_call_create: ToolCallCreate) -> ToolCall:
    # 确保 tool_call 关联的 agent_run 属于同一个 tenant。
    get_agent_run(
        agent_run_id=tool_call_create.agent_run_id,
        tenant_id=tool_call_create.tenant_id,
    )

    tool_call = ToolCall(
        agent_run_id=tool_call_create.agent_run_id,
        tenant_id=tool_call_create.tenant_id,
        tool_name=tool_call_create.tool_name,
        tool_input_json=tool_call_create.tool_input_json,
        tool_output_json=tool_call_create.tool_output_json,
        status=tool_call_create.status,
        error_type=tool_call_create.error_type,
        error_message=tool_call_create.error_message,
    )

    with Session(engine) as session:
        session.add(tool_call)
        session.commit()
        session.refresh(tool_call)
        return tool_call


def get_tool_call(
    tool_call_id: int,
    tenant_id: str,
) -> ToolCall:
    with Session(engine) as session:
        tool_call = session.get(ToolCall, tool_call_id)

        if tool_call is None or tool_call.tenant_id != tenant_id:
            raise HTTPException(status_code=404, detail="Tool call not found")

        return tool_call


def list_tool_calls_by_run(
    agent_run_id: int,
    tenant_id: str,
    status: str | None = None,
    tool_name: str | None = None,
    error_type: str | None = None,
) -> list[ToolCall]:
    # 确保 run 存在且属于当前 tenant。
    get_agent_run(
        agent_run_id=agent_run_id,
        tenant_id=tenant_id,
    )

    with Session(engine) as session:
        statement = (
            select(ToolCall)
            .where(ToolCall.tenant_id == tenant_id)
            .where(ToolCall.agent_run_id == agent_run_id)
        )

        if status is not None:
            statement = statement.where(ToolCall.status == status)

        if tool_name is not None:
            statement = statement.where(ToolCall.tool_name == tool_name)

        if error_type is not None:
            statement = statement.where(ToolCall.error_type == error_type)

        tool_calls = session.exec(statement).all()
        return list(tool_calls)


def list_tool_calls(
    tenant_id: str,
    agent_run_id: int | None = None,
    status: str | None = None,
    tool_name: str | None = None,
    error_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[ToolCall]:
    with Session(engine) as session:
        statement = select(ToolCall).where(ToolCall.tenant_id == tenant_id)

        if agent_run_id is not None:
            statement = statement.where(ToolCall.agent_run_id == agent_run_id)

        if status is not None:
            statement = statement.where(ToolCall.status == status)

        if tool_name is not None:
            statement = statement.where(ToolCall.tool_name == tool_name)

        if error_type is not None:
            statement = statement.where(ToolCall.error_type == error_type)

        statement = (
            statement
            .order_by(ToolCall.id.desc())
            .offset(offset)
            .limit(limit)
        )

        tool_calls = session.exec(statement).all()
        return list(tool_calls)


def update_tool_call(
    tool_call_id: int,
    tenant_id: str,
    tool_call_update: ToolCallUpdate,
) -> ToolCall:
    with Session(engine) as session:
        tool_call = session.get(ToolCall, tool_call_id)

        if tool_call is None or tool_call.tenant_id != tenant_id:
            raise HTTPException(status_code=404, detail="Tool call not found")

        update_data = tool_call_update.model_dump(exclude_unset=True)

        for field_name, field_value in update_data.items():
            setattr(tool_call, field_name, field_value)

        if tool_call.status in {"success", "failed"}:
            tool_call.finished_at = utc_now()

        session.add(tool_call)
        session.commit()
        session.refresh(tool_call)

        return tool_call


def create_approval_request(
    approval_request_create: ApprovalRequestCreate,
) -> ApprovalRequest:
    # 确保 approval_request 关联的 agent_run 属于同一个 tenant。
    get_agent_run(
        agent_run_id=approval_request_create.agent_run_id,
        tenant_id=approval_request_create.tenant_id,
    )

    approval_request = ApprovalRequest(
        agent_run_id=approval_request_create.agent_run_id,
        tenant_id=approval_request_create.tenant_id,
        approval_type=approval_request_create.approval_type,
        status=approval_request_create.status,
        draft_json=approval_request_create.draft_json,
        approved_by=approval_request_create.approved_by,
    )

    with Session(engine) as session:
        session.add(approval_request)
        session.commit()
        session.refresh(approval_request)
        return approval_request


def get_approval_request(
    approval_request_id: int,
    tenant_id: str,
) -> ApprovalRequest:
    with Session(engine) as session:
        approval_request = session.get(ApprovalRequest, approval_request_id)

        if approval_request is None or approval_request.tenant_id != tenant_id:
            raise HTTPException(
                status_code=404,
                detail="Approval request not found",
            )

        return approval_request


def list_approval_requests_by_run(
    agent_run_id: int,
    tenant_id: str,
) -> list[ApprovalRequest]:
    # 确保 run 存在且属于当前 tenant。
    get_agent_run(
        agent_run_id=agent_run_id,
        tenant_id=tenant_id,
    )

    with Session(engine) as session:
        statement = (
            select(ApprovalRequest)
            .where(ApprovalRequest.tenant_id == tenant_id)
            .where(ApprovalRequest.agent_run_id == agent_run_id)
        )

        approval_requests = session.exec(statement).all()
        return list(approval_requests)


def get_agent_run_trace(
    agent_run_id: int,
    tenant_id: str,
) -> tuple[AgentRun, list[ToolCall], list[ApprovalRequest]]:
    """
    获取指定 agent_run 的完整 trace，包括 agent_run 本身、相关的 tool_calls 和 approval_requests。
    """
    agent_run = get_agent_run(
        agent_run_id=agent_run_id,
        tenant_id=tenant_id,
    )

    tool_calls = list_tool_calls_by_run(
        agent_run_id=agent_run_id,
        tenant_id=tenant_id,
    )

    approval_requests = list_approval_requests_by_run(
        agent_run_id=agent_run_id,
        tenant_id=tenant_id,
    )

    return agent_run, tool_calls, approval_requests


def list_approval_requests(
    tenant_id: str,
    agent_run_id: int | None = None,
    status: str | None = None,
    approval_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[ApprovalRequest]:
    with Session(engine) as session:
        statement = select(ApprovalRequest).where(
            ApprovalRequest.tenant_id == tenant_id
        )

        if agent_run_id is not None:
            statement = statement.where(
                ApprovalRequest.agent_run_id == agent_run_id
            )

        if status is not None:
            statement = statement.where(ApprovalRequest.status == status)

        if approval_type is not None:
            statement = statement.where(
                ApprovalRequest.approval_type == approval_type
            )

        statement = (
            statement
            .order_by(ApprovalRequest.id.desc())
            .offset(offset)
            .limit(limit)
        )
        
        approval_requests = session.exec(statement).all()
        return list(approval_requests)


def update_approval_request(
    approval_request_id: int,
    tenant_id: str,
    approval_request_update: ApprovalRequestUpdate,
) -> ApprovalRequest:
    with Session(engine) as session:
        approval_request = session.get(ApprovalRequest, approval_request_id)

        if approval_request is None or approval_request.tenant_id != tenant_id:
            raise HTTPException(
                status_code=404,
                detail="Approval request not found",
            )

        update_data = approval_request_update.model_dump(exclude_unset=True)

        for field_name, field_value in update_data.items():
            setattr(approval_request, field_name, field_value)

        if approval_request.status in {"approved", "rejected", "cancelled"}:
            approval_request.decided_at = utc_now()

        session.add(approval_request)
        session.commit()
        session.refresh(approval_request)

        return approval_request


def create_retrieval_log(
    retrieval_log_create: RetrievalLogCreate,
) -> RetrievalLog:
    retrieval_log = RetrievalLog(
        tenant_id=retrieval_log_create.tenant_id,
        user_id=retrieval_log_create.user_id,
        endpoint=retrieval_log_create.endpoint,
        query_text=retrieval_log_create.query_text,
        top_k=retrieval_log_create.top_k,
        category=retrieval_log_create.category,
        retrieval_status=retrieval_log_create.retrieval_status,
        total_hits=retrieval_log_create.total_hits,
        top_distance=retrieval_log_create.top_distance,
        source_documents_json=retrieval_log_create.source_documents_json,
        scores_json=retrieval_log_create.scores_json,
        latency_ms=retrieval_log_create.latency_ms,
        error_message=retrieval_log_create.error_message,
    )

    with Session(engine) as session:
        session.add(retrieval_log)
        session.commit()
        session.refresh(retrieval_log)
        return retrieval_log


def list_retrieval_logs(
    tenant_id: str,
    endpoint: str | None = None,
    retrieval_status: str | None = None,
    category: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[RetrievalLog]:
    with Session(engine) as session:
        statement = select(RetrievalLog).where(
            RetrievalLog.tenant_id == tenant_id
        )

        if endpoint is not None:
            statement = statement.where(RetrievalLog.endpoint == endpoint)

        if retrieval_status is not None:
            statement = statement.where(
                RetrievalLog.retrieval_status == retrieval_status
            )

        if category is not None:
            statement = statement.where(RetrievalLog.category == category)

        statement = (
            statement
            .order_by(RetrievalLog.id.desc())
            .offset(offset)
            .limit(limit)
        )

        retrieval_logs = session.exec(statement).all()
        return list(retrieval_logs)

 
def count_status(items, status: str) -> int:
    return sum(1 for item in items if item.status == status)

def count_status(
    items,
    status: str,
    field_name: str = "status",
) -> int:
    return sum(
        1
        for item in items
        if getattr(item, field_name) == status
    )

def count_field_values(items, field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}

    for item in items:
        field_value = getattr(item, field_name, None)

        if field_value is None:
            continue

        counts[field_value] = counts.get(field_value, 0) + 1

    return counts


def average_optional_numbers(values: list[float | int | None]) -> float | None:
    valid_values = [
        value
        for value in values
        if value is not None
    ]

    if not valid_values:
        return None

    return round(sum(valid_values) / len(valid_values), 4)


def count_tool_call_error_types(tool_calls: list[ToolCall]) -> dict[str, int]:
    error_types: dict[str, int] = {}

    for tool_call in tool_calls:
        if tool_call.status != "failed":
            continue

        if not tool_call.error_type:
            continue

        error_types[tool_call.error_type] = (
            error_types.get(tool_call.error_type, 0) + 1
        )

    return error_types


def get_agent_ops_metrics_summary(
    tenant_id: str,
) -> AgentOpsMetricsSummaryResponse:
    """
    汇总当前 tenant 下的 AgentOps 指标。

    第一版使用简单的按 tenant 查询 + Python 内存计数。
    当前数据量很小，优先保持实现清晰。
    后续如果数据量增大，再改为数据库 count 聚合。
    """
    with Session(engine) as session:
        agent_runs = list(
            session.exec(
                select(AgentRun).where(AgentRun.tenant_id == tenant_id)
            ).all()
        )

        tool_calls = list(
            session.exec(
                select(ToolCall).where(ToolCall.tenant_id == tenant_id)
            ).all()
        )

        approval_requests = list(
            session.exec(
                select(ApprovalRequest).where(
                    ApprovalRequest.tenant_id == tenant_id
                )
            ).all()
        )

    return AgentOpsMetricsSummaryResponse(
        total_agent_runs=len(agent_runs),
        running_agent_runs=count_status(agent_runs, "running"),
        completed_agent_runs=count_status(agent_runs, "completed"),
        failed_agent_runs=count_status(agent_runs, "failed"),
        cancelled_agent_runs=count_status(agent_runs, "cancelled"),
        total_tool_calls=len(tool_calls),
        pending_tool_calls=count_status(tool_calls, "pending"),
        successful_tool_calls=count_status(tool_calls, "success"),
        failed_tool_calls=count_status(tool_calls, "failed"),
        tool_call_error_types=count_tool_call_error_types(tool_calls),
        total_approval_requests=len(approval_requests),
        pending_approval_requests=count_status(approval_requests, "pending"),
        approved_approval_requests=count_status(approval_requests, "approved"),
        rejected_approval_requests=count_status(approval_requests, "rejected"),
        cancelled_approval_requests=count_status(approval_requests, "cancelled"),
    )


def get_retrieval_metrics_summary(
    tenant_id: str,
    endpoint: str | None = None,
    category: str | None = None,
) -> RetrievalMetricsSummaryResponse:
    with Session(engine) as session:
        statement = select(RetrievalLog).where(
            RetrievalLog.tenant_id == tenant_id
        )

        if endpoint is not None:
            statement = statement.where(RetrievalLog.endpoint == endpoint)

        if category is not None:
            statement = statement.where(RetrievalLog.category == category)

        retrieval_logs = list(session.exec(statement).all())

    return RetrievalMetricsSummaryResponse(
        total_retrieval_logs=len(retrieval_logs),
        ok_retrieval_logs=count_status(
            retrieval_logs,
            "ok",
            field_name="retrieval_status",
        ),
        no_context_retrieval_logs=count_status(
            retrieval_logs,
            "no_context",
            field_name="retrieval_status",
        ),
        failed_retrieval_logs=count_status(
            retrieval_logs,
            "failed",
            field_name="retrieval_status",
        ),
        average_latency_ms=average_optional_numbers(
            [
                retrieval_log.latency_ms
                for retrieval_log in retrieval_logs
            ]
        ),
        average_top_distance=average_optional_numbers(
            [
                retrieval_log.top_distance
                for retrieval_log in retrieval_logs
            ]
        ),
        endpoint_counts=count_field_values(
            retrieval_logs,
            "endpoint",
        ),
        category_counts=count_field_values(
            retrieval_logs,
            "category",
        ),
    )