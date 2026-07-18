from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


CATEGORY_HALF_LIVES = {
    "identity": math.inf,
    "policy": math.inf,
    "heartbeat": math.inf,
    "ob_permanent": math.inf,
    "relationship_moment": 180.0,
    "core": 120.0,
    "fragments": 90.0,
    "episode": 90.0,
    "diary": 60.0,
    "worklog": 45.0,
    "knowledge": 30.0,
    "tasks": 21.0,
    "ob_dynamic": 45.0,
    "conversation": 14.0,
}

# Recall should be relevant without letting an old literal match permanently
# eclipse a recent continuation. The three components are returned to callers
# for auditability rather than hidden in one opaque score.
RECALL_LEXICAL_WEIGHT = 0.45
RECALL_VITALITY_WEIGHT = 0.30
RECALL_RECENCY_WEIGHT = 0.25
RECENCY_HALF_LIFE_DAYS = 30.0


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def vitality(memory: dict[str, Any], now: datetime | None = None) -> float:
    """Ombre-inspired vitality score with category-aware decay.

    The canonical deployment uses valence in [-1, 1] and arousal in [0, 1].
    Legacy Ombre values are preserved because their observed 0..1 subset is
    already valid on that scale.
    """
    if memory.get("protected"):
        return 999.0
    now = now or datetime.now(timezone.utc)
    reference = _as_datetime(memory.get("last_hit")) or _as_datetime(memory.get("created_at"))
    age_days = max(0.0, (now - reference).total_seconds() / 86400) if reference else 30.0
    weight = max(0.1, float(memory.get("weight") or 1.0))
    importance = max(1.0, min(10.0, weight * 3.3))
    activation = min(30, max(1, int(memory.get("hit_count") or 0)))
    arousal = max(0.0, min(1.0, float(memory.get("arousal") or 0.3)))
    half_life = CATEGORY_HALF_LIVES.get(str(memory.get("category") or ""), 45.0)
    decay = 1.0 if math.isinf(half_life) else math.exp(-math.log(2) * age_days / half_life)
    time_weight = 1.0 if age_days <= 1 else max(0.3, math.exp(-0.069 * (age_days - 1)))
    score = time_weight * importance * activation**0.3 * decay * (1.0 + arousal * 0.8)
    if memory.get("resolved"):
        score *= 0.05
    if memory.get("digested"):
        score *= 0.3
    return round(score, 4)


def normalized_vitality(memory: dict[str, Any], now: datetime | None = None) -> float:
    """Map unbounded vitality into [0, 1] without a hard saturation cliff."""
    live = vitality(memory, now=now)
    return round(1.0 - math.exp(-max(0.0, live) / 10.0), 4)


def recency_score(
    memory: dict[str, Any],
    now: datetime | None = None,
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
) -> float:
    """Explicit event-time freshness used independently from activation.

    `last_hit` is deliberately ignored here: recalling an old event should
    increase its activation, but should not rewrite when that event happened.
    """
    now = now or datetime.now(timezone.utc)
    created = _as_datetime(memory.get("created_at")) or _as_datetime(memory.get("valid_at"))
    if created is None:
        return 0.0
    age_days = max(0.0, (now - created).total_seconds() / 86400)
    return round(math.exp(-math.log(2) * age_days / max(1.0, half_life_days)), 4)


def recall_score(
    memory: dict[str, Any],
    lexical_score: float,
    now: datetime | None = None,
) -> tuple[float, dict[str, float]]:
    """Blend text relevance, Ombre vitality, and explicit event recency."""
    lexical = max(0.0, min(1.0, float(lexical_score)))
    vitality_component = normalized_vitality(memory, now=now)
    recency = recency_score(memory, now=now)
    total = (
        lexical * RECALL_LEXICAL_WEIGHT
        + vitality_component * RECALL_VITALITY_WEIGHT
        + recency * RECALL_RECENCY_WEIGHT
    )
    breakdown = {
        "lexical": round(lexical, 4),
        "vitality": vitality_component,
        "recency": recency,
        "lexical_weight": RECALL_LEXICAL_WEIGHT,
        "vitality_weight": RECALL_VITALITY_WEIGHT,
        "recency_weight": RECALL_RECENCY_WEIGHT,
    }
    return round(total, 4), breakdown
