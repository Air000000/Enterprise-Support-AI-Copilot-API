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
    latency_ms: int | None = Field(default=None, ge=0)
    retrieval_summary_json: str | None = None


class AgentRunUpdate(BaseModel):
    status: AgentRunStatus | None = None
    result_summary: str | None = Field(default=None, max_length=2000)
    latency_ms: int | None = Field(default=None, ge=0)
    retrieval_summary_json: str | None = None


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
    latency_ms: int | None
    retrieval_summary_json: str | None
    created_at: datetime
    updated_at: datetime


class ToolCallCreate(BaseModel):
    agent_run_id: int
    tenant_id: str = Field(..., min_length=1)
    tool_name: str = Field(..., min_length=1, max_length=100)
    tool_input_json: str = "{}"
    tool_output_json: str | None = None
    status: ToolCallStatus = "pending"
    error_type: str | None = Field(default=None, max_length=100)
    error_message: str | None = Field(default=None, max_length=2000)


class ToolCallUpdate(BaseModel):
    tool_output_json: str | None = None
    status: ToolCallStatus | None = None
    error_type: str | None = Field(default=None, max_length=100)
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
    error_type: str | None
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
    decision_reason: str | None = Field(default=None, max_length=2000)


class ApprovalDecisionRequest(BaseModel):
    reason: str = Field(..., min_length=1, max_length=2000)


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
    decision_reason: str | None


class RetrievalLogCreate(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    user_id: str | None = Field(default=None, max_length=100)
    endpoint: str = Field(..., min_length=1, max_length=100)
    query_text: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(..., ge=1)
    category: str | None = Field(default=None, max_length=100)
    retrieval_status: str = Field(default="ok", max_length=100)
    total_hits: int = Field(default=0, ge=0)
    top_distance: float | None = None
    source_documents_json: str | None = None
    scores_json: str | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    error_message: str | None = Field(default=None, max_length=2000)


class RetrievalLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    tenant_id: str
    user_id: str | None
    endpoint: str
    query_text: str
    top_k: int
    category: str | None
    retrieval_status: str
    total_hits: int
    top_distance: float | None
    source_documents_json: str | None
    scores_json: str | None
    latency_ms: int | None
    error_message: str | None
    created_at: datetime


class RetrievalMetricsSummaryResponse(BaseModel):
    total_retrieval_logs: int
    ok_retrieval_logs: int
    no_context_retrieval_logs: int
    failed_retrieval_logs: int
    average_latency_ms: float | None = None
    average_top_distance: float | None = None
    endpoint_counts: dict[str, int]
    category_counts: dict[str, int]


class RetrievalSourceMetricResponse(BaseModel):
    document_id: str
    title: str | None = None
    source_path: str | None = None
    retrieval_count: int
    average_distance: float | None = None


class RetrievalNoContextQueryMetricResponse(BaseModel):
    query_text: str
    endpoint: str
    category: str | None = None
    no_context_count: int
    latest_latency_ms: int | None = None


class RetrievalFailureMetricResponse(BaseModel):
    error_message: str
    endpoint: str
    category: str | None = None
    failed_count: int
    latest_latency_ms: int | None = None


class AgentOpsMetricsSummaryResponse(BaseModel):
    total_agent_runs: int
    running_agent_runs: int
    completed_agent_runs: int
    failed_agent_runs: int
    cancelled_agent_runs: int

    total_tool_calls: int
    pending_tool_calls: int
    successful_tool_calls: int
    failed_tool_calls: int
    tool_call_error_types: dict[str, int] = Field(default_factory=dict)

    total_approval_requests: int
    pending_approval_requests: int
    approved_approval_requests: int
    rejected_approval_requests: int
    cancelled_approval_requests: int


class AgentRunTraceResponse(BaseModel):
    agent_run: AgentRunResponse
    tool_calls: list[ToolCallResponse]
    approval_requests: list[ApprovalRequestResponse]