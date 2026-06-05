from __future__ import annotations

from fastapi import APIRouter

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


router = APIRouter(
    prefix="/tickets",
    tags=["tickets"],
)

MOCK_TENANT_ID = "tenant_demo"
MOCK_USER_ID = "user_demo"


@router.post("", response_model=TicketResponse, status_code=201)
def create_ticket(request: TicketCreate) -> TicketResponse:
    ticket = create_ticket_service(
        ticket_create=request,
        tenant_id=MOCK_TENANT_ID,
        created_by=MOCK_USER_ID,
    )

    return TicketResponse.model_validate(ticket)


@router.get("", response_model=list[TicketResponse])
def list_tickets(
    status: TicketStatus | None = None,
    category: TicketCategory | None = None,
) -> list[TicketResponse]:
    tickets = list_tickets_service(
        tenant_id=MOCK_TENANT_ID,
        status=status,
        category=category,
    )

    return [
        TicketResponse.model_validate(ticket)
        for ticket in tickets
    ]


@router.get("/{ticket_id}", response_model=TicketResponse)
def get_ticket(ticket_id: int) -> TicketResponse:
    ticket = get_ticket_service(
        ticket_id=ticket_id,
        tenant_id=MOCK_TENANT_ID,
    )

    return TicketResponse.model_validate(ticket)


@router.patch("/{ticket_id}", response_model=TicketResponse)
def update_ticket(
    ticket_id: int,
    request: TicketUpdate,
) -> TicketResponse:
    ticket = update_ticket_service(
        ticket_id=ticket_id,
        ticket_update=request,
        tenant_id=MOCK_TENANT_ID,
    )

    return TicketResponse.model_validate(ticket)