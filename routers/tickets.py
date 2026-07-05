from __future__ import annotations

from fastapi import APIRouter, Depends

from schemas.ticket import (
    TicketCategory,
    TicketCreate,
    TicketResponse,
    TicketStatus,
    TicketUpdate,
)
from services.ticket_service import (
    create_ticket as create_ticket_service,
    get_ticket as get_ticket_service,
    list_tickets as list_tickets_service,
    update_ticket as update_ticket_service,
)
from auth import CurrentUser, get_current_user


router = APIRouter(
    prefix="/tickets",
    tags=["tickets"],
)



@router.post("", response_model=TicketResponse, status_code=201)
def create_ticket(
    request: TicketCreate,
    user: CurrentUser = Depends(get_current_user),
) -> TicketResponse:
    ticket = create_ticket_service(
        ticket_create=request,
        tenant_id=user.tenant_id,
        created_by=user.user_id,
    )

    return TicketResponse.model_validate(ticket)


@router.get("", response_model=list[TicketResponse])
def list_tickets(
    status: TicketStatus | None = None,
    category: TicketCategory | None = None,
    user: CurrentUser = Depends(get_current_user),
) -> list[TicketResponse]:
    tickets = list_tickets_service(
        tenant_id=user.tenant_id,
        status=status,
        category=category,
    )

    return [
        TicketResponse.model_validate(ticket)
        for ticket in tickets
    ]


@router.get("/{ticket_id}", response_model=TicketResponse)
def get_ticket(
    ticket_id: int,
    user: CurrentUser = Depends(get_current_user),
) -> TicketResponse:
    ticket = get_ticket_service(
        ticket_id=ticket_id,
        tenant_id=user.tenant_id,
    )

    return TicketResponse.model_validate(ticket)


@router.patch("/{ticket_id}", response_model=TicketResponse)
def update_ticket(
    ticket_id: int,
    request: TicketUpdate,
    user: CurrentUser = Depends(get_current_user),
) -> TicketResponse:
    ticket = update_ticket_service(
        ticket_id=ticket_id,
        ticket_update=request,
        tenant_id=user.tenant_id,
    )

    return TicketResponse.model_validate(ticket)
