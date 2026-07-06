from __future__ import annotations

from alembic import op
import sqlalchemy as sa
import sqlmodel


revision = "20260705_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tenant_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("user_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("agent_name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("input_message", sqlmodel.sql.sqltypes.AutoString(length=4000), nullable=False),
        sa.Column("category", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("status", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("result_summary", sqlmodel.sql.sqltypes.AutoString(length=2000), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("retrieval_summary_json", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_runs_agent_name", "agent_runs", ["agent_name"])
    op.create_index("ix_agent_runs_category", "agent_runs", ["category"])
    op.create_index("ix_agent_runs_status", "agent_runs", ["status"])
    op.create_index("ix_agent_runs_tenant_id", "agent_runs", ["tenant_id"])
    op.create_index("ix_agent_runs_user_id", "agent_runs", ["user_id"])

    op.create_table(
        "documents",
        sa.Column("id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("tenant_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("uploaded_by", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("filename", sqlmodel.sql.sqltypes.AutoString(length=255), nullable=False),
        sa.Column("file_type", sqlmodel.sql.sqltypes.AutoString(length=20), nullable=False),
        sa.Column("category", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("source_path", sqlmodel.sql.sqltypes.AutoString(length=1000), nullable=False),
        sa.Column("status", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("checksum", sqlmodel.sql.sqltypes.AutoString(length=128), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("error_message", sqlmodel.sql.sqltypes.AutoString(length=2000), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_documents_category", "documents", ["category"])
    op.create_index("ix_documents_status", "documents", ["status"])
    op.create_index("ix_documents_tenant_id", "documents", ["tenant_id"])
    op.create_index("ix_documents_uploaded_by", "documents", ["uploaded_by"])

    op.create_table(
        "ticket",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tenant_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("created_by", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("title", sqlmodel.sql.sqltypes.AutoString(length=200), nullable=False),
        sa.Column("description", sqlmodel.sql.sqltypes.AutoString(length=2000), nullable=False),
        sa.Column("category", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("priority", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("status", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ticket_category", "ticket", ["category"])
    op.create_index("ix_ticket_created_by", "ticket", ["created_by"])
    op.create_index("ix_ticket_priority", "ticket", ["priority"])
    op.create_index("ix_ticket_status", "ticket", ["status"])
    op.create_index("ix_ticket_tenant_id", "ticket", ["tenant_id"])

    op.create_table(
        "todo",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("title", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("completed", sa.Boolean(), nullable=False),
        sa.Column("due_time", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "approval_requests",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("agent_run_id", sa.Integer(), nullable=False),
        sa.Column("tenant_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("approval_type", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("status", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("draft_json", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("approved_by", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("decision_reason", sqlmodel.sql.sqltypes.AutoString(length=2000), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("decided_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["agent_run_id"], ["agent_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_approval_requests_agent_run_id", "approval_requests", ["agent_run_id"])
    op.create_index("ix_approval_requests_approval_type", "approval_requests", ["approval_type"])
    op.create_index("ix_approval_requests_approved_by", "approval_requests", ["approved_by"])
    op.create_index("ix_approval_requests_status", "approval_requests", ["status"])
    op.create_index("ix_approval_requests_tenant_id", "approval_requests", ["tenant_id"])

    op.create_table(
        "document_chunks",
        sa.Column("id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("tenant_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("document_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sqlmodel.sql.sqltypes.AutoString(length=8000), nullable=False),
        sa.Column("category", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("metadata_json", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("embedding_id", sqlmodel.sql.sqltypes.AutoString(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_document_chunks_category", "document_chunks", ["category"])
    op.create_index("ix_document_chunks_chunk_index", "document_chunks", ["chunk_index"])
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])
    op.create_index("ix_document_chunks_embedding_id", "document_chunks", ["embedding_id"])
    op.create_index("ix_document_chunks_tenant_id", "document_chunks", ["tenant_id"])

    op.create_table(
        "retrieval_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tenant_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("user_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("endpoint", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("query_text", sqlmodel.sql.sqltypes.AutoString(length=4000), nullable=False),
        sa.Column("top_k", sa.Integer(), nullable=False),
        sa.Column("category", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("retrieval_status", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("total_hits", sa.Integer(), nullable=False),
        sa.Column("top_distance", sa.Float(), nullable=True),
        sa.Column("source_documents_json", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("scores_json", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("error_message", sqlmodel.sql.sqltypes.AutoString(length=2000), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_retrieval_logs_category", "retrieval_logs", ["category"])
    op.create_index("ix_retrieval_logs_endpoint", "retrieval_logs", ["endpoint"])
    op.create_index("ix_retrieval_logs_retrieval_status", "retrieval_logs", ["retrieval_status"])
    op.create_index("ix_retrieval_logs_tenant_id", "retrieval_logs", ["tenant_id"])
    op.create_index("ix_retrieval_logs_user_id", "retrieval_logs", ["user_id"])

    op.create_table(
        "tool_calls",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("agent_run_id", sa.Integer(), nullable=False),
        sa.Column("tenant_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("tool_name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("tool_input_json", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("tool_output_json", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("status", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("error_type", sqlmodel.sql.sqltypes.AutoString(length=100), nullable=True),
        sa.Column("error_message", sqlmodel.sql.sqltypes.AutoString(length=2000), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["agent_run_id"], ["agent_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_tool_calls_agent_run_id", "tool_calls", ["agent_run_id"])
    op.create_index("ix_tool_calls_error_type", "tool_calls", ["error_type"])
    op.create_index("ix_tool_calls_status", "tool_calls", ["status"])
    op.create_index("ix_tool_calls_tenant_id", "tool_calls", ["tenant_id"])
    op.create_index("ix_tool_calls_tool_name", "tool_calls", ["tool_name"])


def downgrade() -> None:
    op.drop_index("ix_tool_calls_tool_name", table_name="tool_calls")
    op.drop_index("ix_tool_calls_tenant_id", table_name="tool_calls")
    op.drop_index("ix_tool_calls_status", table_name="tool_calls")
    op.drop_index("ix_tool_calls_error_type", table_name="tool_calls")
    op.drop_index("ix_tool_calls_agent_run_id", table_name="tool_calls")
    op.drop_table("tool_calls")

    op.drop_index("ix_retrieval_logs_user_id", table_name="retrieval_logs")
    op.drop_index("ix_retrieval_logs_tenant_id", table_name="retrieval_logs")
    op.drop_index("ix_retrieval_logs_retrieval_status", table_name="retrieval_logs")
    op.drop_index("ix_retrieval_logs_endpoint", table_name="retrieval_logs")
    op.drop_index("ix_retrieval_logs_category", table_name="retrieval_logs")
    op.drop_table("retrieval_logs")

    op.drop_index("ix_document_chunks_tenant_id", table_name="document_chunks")
    op.drop_index("ix_document_chunks_embedding_id", table_name="document_chunks")
    op.drop_index("ix_document_chunks_document_id", table_name="document_chunks")
    op.drop_index("ix_document_chunks_chunk_index", table_name="document_chunks")
    op.drop_index("ix_document_chunks_category", table_name="document_chunks")
    op.drop_table("document_chunks")

    op.drop_index("ix_approval_requests_tenant_id", table_name="approval_requests")
    op.drop_index("ix_approval_requests_status", table_name="approval_requests")
    op.drop_index("ix_approval_requests_approved_by", table_name="approval_requests")
    op.drop_index("ix_approval_requests_approval_type", table_name="approval_requests")
    op.drop_index("ix_approval_requests_agent_run_id", table_name="approval_requests")
    op.drop_table("approval_requests")

    op.drop_table("todo")

    op.drop_index("ix_ticket_tenant_id", table_name="ticket")
    op.drop_index("ix_ticket_status", table_name="ticket")
    op.drop_index("ix_ticket_priority", table_name="ticket")
    op.drop_index("ix_ticket_created_by", table_name="ticket")
    op.drop_index("ix_ticket_category", table_name="ticket")
    op.drop_table("ticket")

    op.drop_index("ix_documents_uploaded_by", table_name="documents")
    op.drop_index("ix_documents_tenant_id", table_name="documents")
    op.drop_index("ix_documents_status", table_name="documents")
    op.drop_index("ix_documents_category", table_name="documents")
    op.drop_table("documents")

    op.drop_index("ix_agent_runs_user_id", table_name="agent_runs")
    op.drop_index("ix_agent_runs_tenant_id", table_name="agent_runs")
    op.drop_index("ix_agent_runs_status", table_name="agent_runs")
    op.drop_index("ix_agent_runs_category", table_name="agent_runs")
    op.drop_index("ix_agent_runs_agent_name", table_name="agent_runs")
    op.drop_table("agent_runs")
