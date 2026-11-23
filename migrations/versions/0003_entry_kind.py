"""What an indexed entry is: a past proposal, a case study or another document.

Revision ID: 0003_entry_kind
Revises: 0002_jobs
"""

import sqlalchemy as sa
from alembic import op

revision = "0003_entry_kind"
down_revision = "0002_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "library_entry_index",
        sa.Column("kind", sa.String(16), server_default="proposal", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("library_entry_index", "kind")
