from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.ticket import TicketCategory, TicketPriority, TicketResponse


class TicketAgentPreviewRequest(BaseModel):
    """
    /agent/ticket/preview的请求体
    """
    message: str = Field(..., min_length=1, max_length=2000)
    category: TicketCategory | None = None


class TicketDraft(BaseModel):
    """
    工单草稿
    """
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=2000)
    category: TicketCategory
    priority: TicketPriority = "medium"


class TicketAgentSource(BaseModel):
    """
    工单知识库来源(sources)
    """
    document_id: str
    chunk_id: str
    title: str
    source_path: str
    distance: float
    preview: str
    category: str | None = None


class TicketAgentPreviewResponse(BaseModel):
    """
    工单预览响应体
    """
    agent_run_id: int
    approval_request_id: int | None = None
    should_create_ticket: bool
    reason: str
    draft: TicketDraft | None = None
    sources: list[TicketAgentSource] = Field(default_factory=list)


class TicketAgentConfirmRequest(BaseModel):
    """
    /agent/ticket/confirm 的请求体
    """
    agent_run_id: int
    approval_request_id: int
    draft: TicketDraft


class TicketAgentConfirmResponse(BaseModel):
    """
    confirm 成功后的响应体
    """
    agent_run_id: int
    approval_request_id: int
    tool_call_id: int
    ticket: TicketResponse