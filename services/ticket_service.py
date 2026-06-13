from __future__ import annotations

from datetime import datetime, timezone

from fastapi import HTTPException
from sqlmodel import Session, select

from database import engine
from models.ticket import Ticket
from schemas.ticket import TicketCreate, TicketUpdate


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def create_ticket(
    ticket_create: TicketCreate,
    tenant_id: str,
    created_by: str,
) -> Ticket:
    ticket = Ticket(
        tenant_id=tenant_id,
        created_by=created_by,
        title=ticket_create.title,
        description=ticket_create.description,
        category=ticket_create.category,
        priority=ticket_create.priority,
        status="open",
    )

    with Session(engine) as session:
        session.add(ticket)
        session.commit()
        session.refresh(ticket)
        return ticket


def list_tickets(
    tenant_id: str,
    status: str | None = None,
    category: str | None = None,
) -> list[Ticket]:
    with Session(engine) as session:
        statement = select(Ticket).where(Ticket.tenant_id == tenant_id)

        if status is not None:
            statement = statement.where(Ticket.status == status)

        if category is not None:
            statement = statement.where(Ticket.category == category)

        tickets = session.exec(statement).all()
        return list(tickets)


def get_ticket(
    ticket_id: int,
    tenant_id: str, 
) -> Ticket:
    with Session(engine) as session:
        ticket = session.get(Ticket, ticket_id)

        if ticket is None or ticket.tenant_id != tenant_id:
            raise HTTPException(status_code=404, detail="Ticket not found")

        return ticket


def update_ticket(
    ticket_id: int,
    ticket_update: TicketUpdate,
    tenant_id: str,
) -> Ticket:
    with Session(engine) as session:
        ticket = session.get(Ticket, ticket_id)

        if ticket is None or ticket.tenant_id != tenant_id:
            raise HTTPException(status_code=404, detail="Ticket not found")

        update_data = ticket_update.model_dump(exclude_unset=True)

        for field_name, field_value in update_data.items():
            setattr(ticket, field_name, field_value)

        ticket.updated_at = utc_now()

        session.add(ticket)
        session.commit()
        session.refresh(ticket)

        return ticket