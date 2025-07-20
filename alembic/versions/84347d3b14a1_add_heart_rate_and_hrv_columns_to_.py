"""Add heart rate and HRV columns to baseline aggregates

Revision ID: 84347d3b14a1
Revises: 
Create Date: 2025-07-20 00:28:47.229960

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '84347d3b14a1'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add heart_rate_mean, heart_rate_std, hrv_mean, hrv_std to user_baseline_30d table
    op.add_column('user_baseline_30d', sa.Column('heart_rate_mean', sa.Float(), nullable=True))
    op.add_column('user_baseline_30d', sa.Column('heart_rate_std', sa.Float(), nullable=True))
    op.add_column('user_baseline_30d', sa.Column('hrv_mean', sa.Float(), nullable=True))
    op.add_column('user_baseline_30d', sa.Column('hrv_std', sa.Float(), nullable=True))


def downgrade() -> None:
    # Remove HR/HRV columns
    op.drop_column('user_baseline_30d', 'hrv_std')
    op.drop_column('user_baseline_30d', 'hrv_mean')
    op.drop_column('user_baseline_30d', 'heart_rate_std')
    op.drop_column('user_baseline_30d', 'heart_rate_mean')
