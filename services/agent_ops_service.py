from __future__ import annotations

from datetime import datetime, timezone

from fastapi import HTTPException
from sqlmodel import Session, select

from database import engine
from models.agent_ops import AgentRun, ApprovalRequest, ToolCall
from schemas.agent_ops import (
    AgentRunCreate,
    AgentRunUpdate,
    ApprovalRequestCreate,
    ApprovalRequestUpdate,
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
    from __future__ import annotations

from datetime import datetime, timezone

from fastapi import HTTPException
from sqlmodel import Session, select

from database import engine
from models.agent_ops import AgentRun, ApprovalRequest, ToolCall
from schemas.agent_ops import (
    AgentRunCreate,
    AgentRunUpdate,
    ApprovalRequestCreate,
    ApprovalRequestUpdate,
    ToolCallCreate,
    ToolCallUpdate,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def create_agent_run(agent_run_create: AgentRunCreate) -> AgentRun:
    agent_run = AgentRun(
        tenant_id=agent_run_create.tenant_id,
        user_id=agent_run_create.user_id,
        agent_name=agent_run_create.agent_name,
        input_message=agent_run_create.input_message,
        category=agent_run_create.category,
        status=agent_run_create.status,
        result_summary=agent_run_create.result_summary,
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
    with Session(engine) as session:
        agent_run = session.get(AgentRun, agent_run_id)

        if agent_run is None or agent_run.tenant_id != tenant_id:
            raise HTTPException(status_code=404, detail="Agent run not found")

        return agent_run


def list_agent_runs(
    tenant_id: str,
    status: str | None = None,
    agent_name: str | None = None,
) -> list[AgentRun]:
    with Session(engine) as session:
        statement = select(AgentRun).where(AgentRun.tenant_id == tenant_id)

        if status is not None:
            statement = statement.where(AgentRun.status == status)

        if agent_name is not None:
            statement = statement.where(AgentRun.agent_name == agent_name)

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
) -> list[AgentRun]:
    with Session(engine) as session:
        statement = select(AgentRun).where(AgentRun.tenant_id == tenant_id)

        if status is not None:
            statement = statement.where(AgentRun.status == status)

        if agent_name is not None:
            statement = statement.where(AgentRun.agent_name == agent_name)

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