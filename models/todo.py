from __future__ import annotations

from typing import Optional

from sqlmodel import Field, SQLModel


class Todo(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    # ponytail: unowned legacy rows stay hidden; never guess a tenant during migration.
    tenant_id: Optional[str] = Field(default=None, index=True)
    title: str
    completed: bool = False
    due_time: Optional[str] = None
