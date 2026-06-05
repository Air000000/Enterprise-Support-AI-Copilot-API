from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


TicketCategory = Literal[
    "it",
    "hr",
    "finance",
    "admin",
    "security",
    "other",
]

TicketPriority = Literal[
    "low",
    "medium",
    "high",
    "urgent",
]

TicketStatus = Literal[
    "open",
    "in_progress",
    "resolved",
    "closed",
]


class TicketCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=2000)
    category: TicketCategory
    priority: TicketPriority = "medium"


class TicketUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, min_length=1, max_length=2000)
    category: TicketCategory | None = None
    priority: TicketPriority | None = None
    status: TicketStatus | None = None


class TicketResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    tenant_id: str
    created_by: str
    title: str
    description: str
    category: str
    priority: str
    status: str
    created_at: datetime
    updated_at: datetime