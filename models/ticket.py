from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Ticket(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    tenant_id: str = Field(index=True)
    created_by: str = Field(index=True)

    title: str = Field(max_length=200)
    description: str = Field(max_length=2000)

    category: str = Field(index=True)
    priority: str = Field(default="medium", index=True)
    status: str = Field(default="open", index=True)

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)