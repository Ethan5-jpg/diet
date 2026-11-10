"""The validated V10 joint-day context, unchanged input projection and eligibility."""

from bridge import ContextCore

KEY, DAY, FOOD = ContextCore.KEY, ContextCore.DAY, ContextCore.FOOD
LIFE, ALCOHOL, NUTRIENTS = ContextCore.LIFE, ContextCore.ALCOHOL, ContextCore.NUTRIENTS
aggregate = ContextCore.aggregate
read_context = ContextCore.read_context
projected_read, pq = ContextCore.projected_read, ContextCore.pq
