from __future__ import annotations

from datetime import datetime, timedelta, timezone

from lmc5_web.scoring import recall_score, recency_score, vitality


def test_recent_activated_memory_outranks_old_unhit_memory():
    now = datetime.now(timezone.utc)
    base = {
        "weight": 2.0,
        "category": "ob_dynamic",
        "protected": False,
        "activation_boost": 0,
        "valence": 0.5,
        "arousal": 0.5,
    }
    recent = {**base, "created_at": now - timedelta(days=2), "hit_count": 4, "last_hit": now}
    old = {**base, "created_at": now - timedelta(days=180), "hit_count": 0, "last_hit": None}
    assert vitality(recent, now=now) > vitality(old, now=now)


def test_protected_memory_does_not_time_decay():
    now = datetime.now(timezone.utc)
    memory = {
        "weight": 2.0,
        "category": "ob_permanent",
        "protected": True,
        "created_at": now - timedelta(days=1000),
        "hit_count": 0,
        "activation_boost": 0,
        "valence": 0,
        "arousal": 0,
    }
    assert vitality(memory, now=now) >= 2.0


def test_explicit_recency_can_outweigh_a_modest_old_lexical_advantage():
    now = datetime.now(timezone.utc)
    base = {
        "weight": 2.0,
        "category": "episode",
        "protected": False,
        "hit_count": 1,
        "arousal": 0.4,
    }
    day8 = {**base, "created_at": now - timedelta(days=24)}
    day32 = {**base, "created_at": now}
    old_score, old_breakdown = recall_score(day8, lexical_score=0.82, now=now)
    recent_score, recent_breakdown = recall_score(day32, lexical_score=0.68, now=now)
    assert recent_score > old_score
    assert recent_breakdown["recency"] > old_breakdown["recency"]
    assert recent_breakdown["lexical_weight"] == 0.45
    assert recent_breakdown["vitality_weight"] == 0.30
    assert recent_breakdown["recency_weight"] == 0.25


def test_recency_uses_event_time_not_last_recall_time():
    now = datetime.now(timezone.utc)
    old = {"created_at": now - timedelta(days=60), "last_hit": now}
    assert recency_score(old, now=now) == 0.25
