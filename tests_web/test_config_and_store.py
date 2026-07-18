from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

from lmc5_web.config import Settings
from lmc5_web.store import MemoryStore


def test_settings_accepts_zeabur_postgres_alias(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("POSTGRES_CONNECTION_STRING", "postgresql://db/lmc5")
    monkeypatch.setenv("LMC5_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LMC5_PUBLIC_BASE_URL", "https://memory.example/")
    settings = Settings.from_env()
    assert settings.database_url == "postgresql://db/lmc5"
    assert settings.server_base_url == "https://memory.example"
    assert settings.oauth_resource_url == "https://memory.example/mcp"


@pytest.mark.parametrize(
    ("field", "value"),
    [("category", "unknown"), ("privacy_scope", "everybody"), ("arousal", 1.5)],
)
def test_memory_validation_rejects_unsupported_values(field, value):
    memory = {"title": "Title", "content": "Body", field: value}
    with pytest.raises(ValueError):
        MemoryStore._validate_memory(memory)


def test_recall_reserves_space_for_linked_memory_and_backfills():
    now = datetime.now(timezone.utc)
    lexical = [
        {
            "id": memory_id,
            "title": f"Lexical {memory_id}",
            "content": "matching content",
            "category": "episode",
            "thread": "timeline",
            "tags": [],
            "protected": False,
            "privacy_scope": "personal",
            "created_at": now,
            "weight": 2,
            "hit_count": 1,
            "lexical_score": score,
        }
        for memory_id, score in ((1, 0.9), (2, 0.8), (3, 0.7))
    ]
    linked = {
        "id": 9,
        "title": "Linked OB memory",
        "content": "felt perspective",
        "category": "fragments",
        "thread": "relationship",
        "tags": [],
        "protected": False,
        "privacy_scope": "personal",
        "created_at": now,
        "weight": 2,
        "hit_count": 1,
        "graph_score": 0.8,
        "depth": 1,
        "relation_type": "emotional_link",
    }

    class FakeConnection:
        def execute(self, sql, params=None, **kwargs):
            class Result:
                def __init__(self, rows):
                    self.rows = rows

                def fetchall(self):
                    return self.rows

            if "AS lexical_score" in sql:
                return Result(lexical)
            if "WITH RECURSIVE" in sql:
                return Result([linked])
            if sql.lstrip().startswith("UPDATE"):
                return Result([])
            raise AssertionError(sql)

    class FakeStore(MemoryStore):
        @contextmanager
        def connect(self):
            yield FakeConnection()

    recalled = FakeStore("unused").recall("matching", limit=3)
    assert [item["id"] for item in recalled] == [1, 9, 2]
    assert recalled[1]["channels"] == [
        "graph:emotional_link:hop1",
        "vitality",
        "recency",
    ]


def test_reimport_can_refresh_legacy_classification_without_duplicate():
    calls = []

    class Result:
        def __init__(self, row=None):
            self.row = row

        def fetchone(self):
            return self.row

    class FakeConnection:
        def execute(self, sql, params=None, **kwargs):
            calls.append((sql, params))
            if sql.lstrip().startswith("SELECT id"):
                return Result({"id": 7})
            return Result()

    memory_id, created = MemoryStore("unused")._upsert_memory(
        FakeConnection(),
        {
            "legacy_source": "ltm",
            "legacy_id": "same-section",
            "source": "legacy_ltm",
            "category": "relationship_moment",
            "title": "助手状态更新",
            "content": "关系内容中也可能提到技术。",
            "metadata": {"classification_version": "structure-v2"},
        },
    )
    assert (memory_id, created) == (7, False)
    update_sql, update_params = calls[1]
    assert "UPDATE lmc5_curated_memories" in update_sql
    assert update_params[2] == "relationship_moment"
