"""Asynchronous jobs (POST /v1/jobs): one row per (workspace, idempotency key).

Revision ID: 0002_jobs
Revises: 0001_initial
"""

import sqlalchemy as sa
from alembic import op

revision = "0002_jobs"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "job",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("type", sa.String(40), nullable=False),
        sa.Column("idempotency_key", sa.String(200), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_job"),
        sa.UniqueConstraint("workspace_id", "idempotency_key", name="uq_job_workspace_id"),
    )
    op.create_index("ix_job_workspace_id", "job", ["workspace_id"])


def downgrade() -> None:
    op.drop_index("ix_job_workspace_id", table_name="job")
    op.drop_table("job")
