from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


AgentRunStatus = Literal[
    "running",
    "completed",
    "failed",
    "cancelled",
]

ToolCallStatus = Literal[
    "pending",
    "success",
    "failed",
]

ApprovalRequestStatus = Literal[
    "pending",
    "approved",
    "rejected",
    "cancelled",
]

ApprovalType = Literal[
    "ticket_creation",
]


class AgentRunCreate(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    agent_name: str = Field(default="ticket_agent", min_length=1, max_length=100)
    input_message: str = Field(..., min_length=1, max_length=4000)
    category: str | None = Field(default=None, max_length=100)
    status: AgentRunStatus = "running"
    result_summary: str | None = Field(default=None, max_length=2000)


class AgentRunUpdate(BaseModel):
    status: AgentRunStatus | None = None
    result_summary: str | None = Field(default=None, max_length=2000)


class AgentRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    tenant_id: str
    user_id: str
    agent_name: str
    input_message: str
    category: str | None
    status: str
    result_summary: str | None
    created_at: datetime
    updated_at: datetime


class ToolCallCreate(BaseModel):
    agent_run_id: int
    tenant_id: str = Field(..., min_length=1)
    tool_name: str = Field(..., min_length=1, max_length=100)
    tool_input_json: str = "{}"
    tool_output_json: str | None = None
    status: ToolCallStatus = "pending"
    error_message: str | None = Field(default=None, max_length=2000)


class ToolCallUpdate(BaseModel):
    tool_output_json: str | None = None
    status: ToolCallStatus | None = None
    error_message: str | None = Field(default=None, max_length=2000)


class ToolCallResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    agent_run_id: int
    tenant_id: str
    tool_name: str
    tool_input_json: str
    tool_output_json: str | None
    status: str
    error_message: str | None
    created_at: datetime
    finished_at: datetime | None


class ApprovalRequestCreate(BaseModel):
    agent_run_id: int
    tenant_id: str = Field(..., min_length=1)
    approval_type: ApprovalType = "ticket_creation"
    status: ApprovalRequestStatus = "pending"
    draft_json: str = Field(..., min_length=1)
    approved_by: str | None = Field(default=None, max_length=100)


class ApprovalRequestUpdate(BaseModel):
    status: ApprovalRequestStatus
    approved_by: str | None = Field(default=None, max_length=100)


class ApprovalRequestResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    agent_run_id: int
    tenant_id: str
    approval_type: str
    status: str
    draft_json: str
    approved_by: str | None
    created_at: datetime
    decided_at: datetime | None