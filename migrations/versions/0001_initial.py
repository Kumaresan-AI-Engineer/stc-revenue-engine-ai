"""Library index (pgvector), embedding cache and result cache.

Revision ID: 0001_initial
Revises:
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

# Frozen: BAAI/bge-small-en-v1.5 (fastembed). Another size needs a new migration and a re-index.
DIMENSIONS = 384


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
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
        "library_passage",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("id", sa.String(32), nullable=False),
        sa.Column("entry_id", sa.String(64), nullable=False),
        sa.Column("section", sa.String(16), nullable=False),
        sa.Column("page", sa.Integer(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(DIMENSIONS), nullable=False),
        sa.PrimaryKeyConstraint("workspace_id", "id", name="pk_library_passage"),
    )
    op.create_index("ix_library_passage_entry_id", "library_passage", ["entry_id"])
    op.create_table(
        "embedding_cache",
        sa.Column("key", sa.String(64), nullable=False),
        sa.Column("embedding", Vector(DIMENSIONS), nullable=False),
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
    op.drop_index("ix_library_passage_entry_id", table_name="library_passage")
    op.drop_table("library_passage")
    op.drop_table("library_entry_index")
