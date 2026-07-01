from __future__ import annotations

from fastapi import APIRouter

from schemas.agent_ticket import (
    TicketAgentConfirmRequest,
    TicketAgentConfirmResponse,
    TicketAgentPreviewRequest,
    TicketAgentPreviewResponse,
)
from services.ticket_agent_service import (
    confirm_ticket as confirm_ticket_service,
    preview_ticket as preview_ticket_service,
)
from mock_context import MOCK_TENANT_ID, MOCK_USER_ID


router = APIRouter(
    prefix="/agent/ticket",
    tags=["agent-ticket"],
)



@router.post("/preview", response_model=TicketAgentPreviewResponse)
def preview_ticket(
    request: TicketAgentPreviewRequest,
) -> TicketAgentPreviewResponse:
    return preview_ticket_service(
        request=request,
        tenant_id=MOCK_TENANT_ID,
    )


@router.post("/confirm", response_model=TicketAgentConfirmResponse, status_code=201)
def confirm_ticket(
    request: TicketAgentConfirmRequest,
) -> TicketAgentConfirmResponse:
    return confirm_ticket_service(
        request=request,
        tenant_id=MOCK_TENANT_ID,
        created_by=MOCK_USER_ID,
    )