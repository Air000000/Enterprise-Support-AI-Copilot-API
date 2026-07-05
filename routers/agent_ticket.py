from __future__ import annotations

from fastapi import APIRouter, Depends

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
from auth import CurrentUser, get_current_user


router = APIRouter(
    prefix="/agent/ticket",
    tags=["agent-ticket"],
)



@router.post("/preview", response_model=TicketAgentPreviewResponse)
def preview_ticket(
    request: TicketAgentPreviewRequest,
    user: CurrentUser = Depends(get_current_user),
) -> TicketAgentPreviewResponse:
    return preview_ticket_service(
        request=request,
        tenant_id=user.tenant_id,
    )


@router.post("/confirm", response_model=TicketAgentConfirmResponse, status_code=201)
def confirm_ticket(
    request: TicketAgentConfirmRequest,
    user: CurrentUser = Depends(get_current_user),
) -> TicketAgentConfirmResponse:
    return confirm_ticket_service(
        request=request,
        tenant_id=user.tenant_id,
        created_by=user.user_id,
    )
