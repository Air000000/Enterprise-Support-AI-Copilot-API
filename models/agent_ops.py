from __future__ import annotations

from datetime import datetime, timezone
from typing import ClassVar, Optional

from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class AgentRun(SQLModel, table=True):
    __tablename__: ClassVar[str] = "agent_runs"

    id: Optional[int] = Field(default=None, primary_key=True)

    tenant_id: str = Field(index=True)
    user_id: str = Field(index=True)

    agent_name: str = Field(default="ticket_agent", index=True)
    input_message: str = Field(max_length=4000)
    category: Optional[str] = Field(default=None, index=True)

    status: str = Field(default="running", index=True)
    result_summary: Optional[str] = Field(default=None, max_length=2000)

    latency_ms: Optional[int] = Field(default=None)
    retrieval_summary_json: Optional[str] = Field(default=None)

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ToolCall(SQLModel, table=True):
    __tablename__: ClassVar[str] = "tool_calls"

    id: Optional[int] = Field(default=None, primary_key=True)

    agent_run_id: int = Field(foreign_key="agent_runs.id", index=True)
    tenant_id: str = Field(index=True)

    tool_name: str = Field(index=True)
    tool_input_json: str = Field(default="{}")
    tool_output_json: Optional[str] = None

    status: str = Field(default="pending", index=True)
    error_message: Optional[str] = Field(default=None, max_length=2000)

    created_at: datetime = Field(default_factory=utc_now)
    finished_at: Optional[datetime] = None


class ApprovalRequest(SQLModel, table=True):
    __tablename__: ClassVar[str] = "approval_requests"

    id: Optional[int] = Field(default=None, primary_key=True)

    agent_run_id: int = Field(foreign_key="agent_runs.id", index=True)
    tenant_id: str = Field(index=True)

    approval_type: str = Field(default="ticket_creation", index=True)
    status: str = Field(default="pending", index=True)

    draft_json: str
    approved_by: Optional[str] = Field(default=None, index=True)
    decision_reason: Optional[str] = Field(default=None, max_length=2000)

    created_at: datetime = Field(default_factory=utc_now)
    decided_at: Optional[datetime] = None