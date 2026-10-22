"""Isolate legacy Todo API without guessing ownership of existing records."""

from alembic import op
import sqlalchemy as sa

revision = "20261022_0002"
down_revision = "20260705_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("todo", sa.Column("tenant_id", sa.String(), nullable=True))
    op.create_index("ix_todo_tenant_id", "todo", ["tenant_id"])


def downgrade() -> None:
    # ponytail: refuse to erase known ownership; rollback needs an explicit data plan.
    owned = op.get_bind().execute(sa.text("SELECT count(*) FROM todo WHERE tenant_id IS NOT NULL")).scalar_one()
    if owned:
        raise RuntimeError("Cannot remove Todo tenant ownership while owned rows exist")
    op.drop_index("ix_todo_tenant_id", table_name="todo")
    op.drop_column("todo", "tenant_id")
