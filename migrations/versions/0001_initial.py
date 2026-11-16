"""Library entry index, embedding cache and result cache (passage vectors are in Qdrant).

Revision ID: 0001_initial
Revises:
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "library_entry_index",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("entry_id", sa.String(64), nullable=False),
        sa.Column("client", sa.String(200), nullable=False),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("industry", sa.String(60), nullable=False),
        sa.Column("outcome", sa.String(10), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("indexed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("workspace_id", "entry_id", name="pk_library_entry_index"),
    )
    op.create_table(
        "embedding_cache",
        sa.Column("key", sa.String(64), nullable=False),
        sa.Column("embedding", postgresql.ARRAY(sa.REAL()), nullable=False),
        sa.PrimaryKeyConstraint("key", name="pk_embedding_cache"),
    )
    op.create_table(
        "result_cache",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("key", sa.String(64), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("workspace_id", "key", name="pk_result_cache"),
    )


def downgrade() -> None:
    op.drop_table("result_cache")
    op.drop_table("embedding_cache")
    op.drop_table("library_entry_index")
