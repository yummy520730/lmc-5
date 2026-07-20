from __future__ import annotations

import hashlib
import random
import re
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from importlib.resources import files
from typing import Any, Iterator, Sequence

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .scoring import normalized_vitality, recall_score, recency_score, vitality


ALLOWED_CATEGORIES = {
    "identity",
    "policy",
    "core",
    "relationship_state",
    "relationship_moment",
    "heartbeat",
    "fragments",
    "episode",
    "diary",
    "worklog",
    "knowledge",
    "tasks",
    "health",
    "legal",
    "ob_dynamic",
    "ob_permanent",
    "conversation",
    "note",
}
ALLOWED_PRIVACY = {"personal", "sensitive", "secret", "public"}
ALLOWED_ROLES = {"user", "assistant", "system", "tool", "note"}

QUERY_TERM_SYNONYMS = {
    "裁员": ("被裁", "辞退", "解雇", "优化", "n+1", "赔偿"),
    "工作": ("职业", "上班", "任职", "公司", "岗位", "合同", "续签", "工资"),
    "离婚": ("婚姻", "离婚协议", "起诉离婚", "抚养权", "律师"),
    "回家": ("回老家", "搬家", "返乡"),
}


def _query_concepts(query: str) -> list[tuple[str, ...]]:
    raw_terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.+\-]*|[\u4e00-\u9fff]{2,}", query)
    if not raw_terms:
        raw_terms = [query]
    concepts: list[tuple[str, ...]] = []
    for raw in raw_terms[:12]:
        alternatives = [raw]
        for key, synonyms in QUERY_TERM_SYNONYMS.items():
            if key in raw or raw in key:
                alternatives.extend((key, *synonyms))
        concepts.append(tuple(dict.fromkeys(item.casefold() for item in alternatives if item)))
    return concepts


def _query_coverage(memory: dict[str, Any], concepts: list[tuple[str, ...]]) -> float:
    if not concepts:
        return 0.0
    tags = " ".join(str(tag) for tag in memory.get("tags") or [])
    text = f"{memory.get('title') or ''}\n{memory.get('content') or ''}\n{tags}".casefold()
    matched = sum(any(term in text for term in alternatives) for alternatives in concepts)
    return matched / len(concepts)


def _select_recall_seeds(
    ranked: list[dict[str, Any]],
    *,
    lexical_slots: int,
    include_sensitive: bool,
) -> list[dict[str, Any]]:
    """Reserve explicit sensitive matches before graph expansion.

    Sensitive memories never receive a score penalty. When the caller explicitly
    opts in, up to half of the lexical slots are reserved for sensitive memories
    that matched a query term in title, body, or tags. The remaining slots keep
    the ordinary global score order.
    """
    if not include_sensitive:
        return ranked[:lexical_slots]

    reserve = max(1, (lexical_slots + 1) // 2)
    exact_sensitive = [
        item for item in ranked if item.get("_explicit_sensitive_match")
    ][:reserve]
    selected = list(exact_sensitive)
    seen = {int(item["id"]) for item in selected}
    for item in ranked:
        if len(selected) >= lexical_slots:
            break
        if int(item["id"]) not in seen:
            selected.append(item)
            seen.add(int(item["id"]))
    return selected


class StoreUnavailable(RuntimeError):
    pass


class MemoryStore:
    def __init__(self, database_url: str):
        self.database_url = database_url

    @contextmanager
    def connect(self) -> Iterator[psycopg.Connection]:
        if not self.database_url:
            raise StoreUnavailable("DATABASE_URL is not configured")
        with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
            yield conn

    def initialize(self) -> None:
        schema = files("lmc5_web").joinpath("schema.sql").read_text(encoding="utf-8")
        with self.connect() as conn:
            conn.execute(schema, prepare=False)

    def health(self) -> dict[str, Any]:
        if not self.database_url:
            return {"connected": False, "error": "DATABASE_URL is not configured"}
        try:
            with self.connect() as conn:
                counts = conn.execute(
                    """
                    SELECT
                      (SELECT count(*) FROM lmc5_curated_memories) AS memories,
                      (SELECT count(*) FROM lmc5_source_documents) AS documents,
                      (SELECT count(*) FROM lmc5_raw_events) AS events,
                      (SELECT count(*) FROM lmc5_memory_relations WHERE status='current') AS relations,
                      EXISTS (SELECT 1 FROM pg_extension WHERE extname='vector') AS pgvector
                    """
                ).fetchone()
            return {"connected": True, **dict(counts)}
        except Exception as exc:
            return {"connected": False, "error": str(exc)[:240]}

    @staticmethod
    def _validate_memory(data: dict[str, Any]) -> None:
        if not str(data.get("title") or "").strip():
            raise ValueError("title is required")
        if not str(data.get("content") or "").strip():
            raise ValueError("content is required")
        category = str(data.get("category") or "note")
        if category not in ALLOWED_CATEGORIES:
            raise ValueError(f"unsupported category: {category}")
        privacy = str(data.get("privacy_scope") or "personal")
        if privacy not in ALLOWED_PRIVACY:
            raise ValueError(f"unsupported privacy_scope: {privacy}")
        for key, low, high in (("valence", -1, 1), ("arousal", 0, 1), ("tension", 0, 1)):
            value = data.get(key)
            if value is not None and not low <= float(value) <= high:
                raise ValueError(f"{key} must be between {low} and {high}")

    def _upsert_memory(self, conn: psycopg.Connection, data: dict[str, Any]) -> tuple[int, bool]:
        self._validate_memory(data)
        legacy_source = data.get("legacy_source")
        legacy_id = data.get("legacy_id")
        created_at = data.get("created_at") or datetime.now(timezone.utc)
        if legacy_source and legacy_id:
            existing = conn.execute(
                "SELECT id FROM lmc5_curated_memories WHERE legacy_source=%s AND legacy_id=%s",
                (legacy_source, legacy_id),
            ).fetchone()
            if existing:
                memory_id = int(existing["id"])
                # Re-imports are idempotent but not frozen: parser improvements
                # may correct category/privacy metadata while runtime activation
                # and factual version history remain untouched.
                conn.execute(
                    """
                    UPDATE lmc5_curated_memories SET
                      source_document_id=COALESCE(%s,source_document_id),source=%s,category=%s,
                      title=%s,content=%s,thread=%s,tags=%s,metadata=%s,weight=%s,
                      original_importance=%s,valence=%s,arousal=%s,
                      protected=(protected OR %s),privacy_scope=%s,surface_allowed=%s,
                      resolved=(resolved OR %s),digested=(digested OR %s),version_status='current',
                      created_at=COALESCE(%s,created_at),updated_at=NOW()
                    WHERE id=%s
                    """,
                    (
                        data.get("source_document_id"),
                        data.get("source", "manual"),
                        data.get("category", "note"),
                        str(data["title"]).strip()[:500],
                        str(data["content"]).strip(),
                        data.get("thread", "other"),
                        list(data.get("tags") or []),
                        Jsonb(data.get("metadata") or {}),
                        float(data.get("weight") or 1.0),
                        data.get("original_importance"),
                        data.get("valence"),
                        data.get("arousal"),
                        bool(data.get("protected", False)),
                        data.get("privacy_scope", "personal"),
                        bool(data.get("surface_allowed", True)),
                        bool(data.get("resolved", False)),
                        bool(data.get("digested", False)),
                        data.get("created_at"),
                        memory_id,
                    ),
                )
                return memory_id, False

        row = conn.execute(
            """
            INSERT INTO lmc5_curated_memories (
                legacy_source, legacy_id, source_document_id, source, category,
                title, content, thread, tags, metadata, weight, original_importance,
                hit_count, last_hit, depth, activation_boost, valence, arousal,
                tension, response_tendency, growth_delta, version_status, fact_key,
                active_fact, protected, confidence, privacy_scope, surface_allowed,
                resolved, digested, created_at, updated_at, valid_at
            ) VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
            ) RETURNING id
            """,
            (
                legacy_source,
                legacy_id,
                data.get("source_document_id"),
                data.get("source", "manual"),
                data.get("category", "note"),
                str(data["title"]).strip()[:500],
                str(data["content"]).strip(),
                data.get("thread", "other"),
                list(data.get("tags") or []),
                Jsonb(data.get("metadata") or {}),
                float(data.get("weight") or 1.0),
                data.get("original_importance"),
                int(data.get("hit_count") or 0),
                data.get("last_hit"),
                data.get("depth"),
                float(data.get("activation_boost") or 0),
                data.get("valence"),
                data.get("arousal"),
                data.get("tension"),
                data.get("response_tendency", ""),
                data.get("growth_delta", ""),
                data.get("version_status", "current"),
                data.get("fact_key"),
                bool(data.get("active_fact", False)),
                bool(data.get("protected", False)),
                data.get("confidence"),
                data.get("privacy_scope", "personal"),
                bool(data.get("surface_allowed", True)),
                bool(data.get("resolved", False)),
                bool(data.get("digested", False)),
                created_at,
                data.get("updated_at") or created_at,
                data.get("valid_at") or created_at,
            ),
        ).fetchone()
        return int(row["id"]), True

    def remember(self, **data: Any) -> dict[str, Any]:
        with self.connect() as conn:
            memory_id, created = self._upsert_memory(conn, data)
        return {"memory_id": memory_id, "created": created}

    def record_event(
        self,
        role: str,
        content: str,
        *,
        session_id: str | None = None,
        channel: str = "claude_web",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if role not in ALLOWED_ROLES:
            raise ValueError(f"role must be one of {sorted(ALLOWED_ROLES)}")
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("content is required")
        content_hash = hashlib.sha256(clean_content.encode("utf-8")).hexdigest()
        with self.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO lmc5_raw_events(session_id, role, channel, content, content_hash, metadata)
                VALUES (%s,%s,%s,%s,%s,%s)
                ON CONFLICT (session_id, role, content_hash) DO NOTHING
                RETURNING id
                """,
                (session_id or "", role, channel, clean_content, content_hash, Jsonb(metadata or {})),
            ).fetchone()
        return {"event_id": int(row["id"]) if row else None, "created": bool(row)}

    def recall(self, query: str, *, limit: int = 8, include_sensitive: bool = False) -> list[dict[str, Any]]:
        query = query.strip()[:2000]
        if not query:
            return []
        privacy = ["personal", "public", "sensitive"] if include_sensitive else ["personal", "public"]
        concepts = _query_concepts(query)
        expanded_terms = list(dict.fromkeys(term for group in concepts for term in group))
        patterns = [f"%{term}%" for term in expanded_terms]
        lexical_candidate_limit = max(limit * (10 if include_sensitive else 6), 80 if include_sensitive else 40)
        recent_candidate_limit = max(limit * (6 if include_sensitive else 4), 40 if include_sensitive else 24)
        with self.connect() as conn:
            rows = conn.execute(
                """
                WITH eligible AS (
                  SELECT *, similarity(COALESCE(title,'') || ' ' || content, %s) AS lexical_score,
                    (SELECT count(*) FROM unnest(%s::text[]) AS term
                     WHERE (COALESCE(title,'') || ' ' || content) ILIKE '%%' || term || '%%'
                        OR array_to_string(tags,' ') ILIKE '%%' || term || '%%') AS term_hits
                  FROM lmc5_curated_memories
                  WHERE version_status='current'
                    AND privacy_scope = ANY(%s)
                    AND (
                      (COALESCE(title,'') || ' ' || content) ILIKE ANY(%s)
                      OR array_to_string(tags,' ') ILIKE ANY(%s)
                      OR similarity(COALESCE(title,'') || ' ' || content, %s) > 0.04
                    )
                ), candidates AS (
                  (SELECT * FROM eligible
                   ORDER BY term_hits DESC, lexical_score DESC, protected DESC, created_at DESC
                   LIMIT %s)
                  UNION ALL
                  (SELECT * FROM eligible
                   ORDER BY created_at DESC NULLS LAST, lexical_score DESC
                   LIMIT %s)
                )
                SELECT DISTINCT ON (id) * FROM candidates
                ORDER BY id, lexical_score DESC
                """,
                (
                    query,
                    expanded_terms,
                    privacy,
                    patterns,
                    patterns,
                    query,
                    lexical_candidate_limit,
                    recent_candidate_limit,
                ),
            ).fetchall()

            scored: list[tuple[float, dict[str, Any]]] = []
            for row in rows:
                item = dict(row)
                live = vitality(item)
                term_hits = int(item.pop("term_hits", 0) or 0)
                lexical = max(
                    0.0,
                    float(item.pop("lexical_score") or 0),
                    _query_coverage(item, concepts),
                )
                final, breakdown = recall_score(item, lexical)
                explicit_sensitive_match = bool(
                    include_sensitive
                    and item.get("privacy_scope") == "sensitive"
                    and term_hits > 0
                )
                item["_explicit_sensitive_match"] = explicit_sensitive_match
                if explicit_sensitive_match:
                    breakdown["privacy_multiplier"] = 1.0
                    breakdown["explicit_sensitive_match"] = 1.0
                item["score"] = final
                item["score_breakdown"] = breakdown
                item["vitality"] = live
                item["channels"] = ["lexical", "entity_terms", "vitality", "recency"]
                if explicit_sensitive_match:
                    item["channels"].insert(2, "explicit_sensitive")
                scored.append((final, item))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            ranked = [item for _, item in scored]
            # Keep a small, bounded space for linked OB/LTM memories. Without this,
            # a full lexical result page would make the relation graph invisible.
            graph_slots = min(2, max(0, limit - 1))
            lexical_slots = max(1, limit - graph_slots)
            selected = _select_recall_seeds(
                ranked,
                lexical_slots=lexical_slots,
                include_sensitive=include_sensitive,
            )

            seed_ids = [int(item["id"]) for item in selected]
            if seed_ids:
                graph_rows = conn.execute(
                    """
                    WITH RECURSIVE edges AS (
                      SELECT source_id AS a, target_id AS b, relation_type, strength
                      FROM lmc5_memory_relations
                      WHERE status='current' AND valid_until IS NULL
                      UNION ALL
                      SELECT target_id, source_id, relation_type, strength
                      FROM lmc5_memory_relations
                      WHERE status='current' AND valid_until IS NULL
                    ), walk AS (
                      SELECT a AS seed_id, b AS memory_id, relation_type,
                             strength::double precision AS graph_score,
                             ARRAY[a,b]::bigint[] AS path, 1 AS depth
                      FROM edges WHERE a = ANY(%s)
                      UNION ALL
                      SELECT w.seed_id, e.b, e.relation_type,
                             w.graph_score * e.strength * 0.7,
                             w.path || e.b, w.depth + 1
                      FROM walk w JOIN edges e ON e.a=w.memory_id
                      WHERE w.depth < 2 AND NOT e.b = ANY(w.path)
                    )
                    SELECT * FROM (
                      SELECT DISTINCT ON (m.id) m.*, w.graph_score, w.depth, w.relation_type
                      FROM walk w JOIN lmc5_curated_memories m ON m.id=w.memory_id
                      WHERE NOT (m.id = ANY(%s))
                        AND m.version_status='current'
                        AND m.privacy_scope = ANY(%s)
                      ORDER BY m.id, w.graph_score DESC
                    ) linked
                    ORDER BY graph_score DESC
                    LIMIT %s
                    """,
                    (seed_ids, seed_ids, privacy, limit),
                ).fetchall()
                seen = set(seed_ids)
                for row in sorted(graph_rows, key=lambda r: float(r["graph_score"]), reverse=True):
                    if int(row["id"]) in seen or len(selected) >= limit:
                        continue
                    item = dict(row)
                    graph_score = float(item.pop("graph_score"))
                    live = vitality(item)
                    vitality_component = normalized_vitality(item)
                    recency = recency_score(item)
                    item["score"] = round(
                        graph_score * 0.55 + vitality_component * 0.25 + recency * 0.20,
                        4,
                    )
                    item["score_breakdown"] = {
                        "graph": round(graph_score, 4),
                        "vitality": vitality_component,
                        "recency": recency,
                        "graph_weight": 0.55,
                        "vitality_weight": 0.25,
                        "recency_weight": 0.20,
                    }
                    item["vitality"] = live
                    item["channels"] = [
                        f"graph:{item.pop('relation_type')}:hop{item.pop('depth')}",
                        "vitality",
                        "recency",
                    ]
                    selected.append(item)
                    seen.add(int(item["id"]))

                # If fewer graph memories were eligible, backfill with the next
                # lexical candidates so recall still returns up to the requested limit.
                for item in ranked:
                    if len(selected) >= limit:
                        break
                    if int(item["id"]) not in seen:
                        selected.append(item)
                        seen.add(int(item["id"]))

                conn.execute(
                    "UPDATE lmc5_curated_memories SET hit_count=hit_count+1,last_hit=NOW(),updated_at=NOW() WHERE id=ANY(%s)",
                    ([int(item["id"]) for item in selected],),
                )
        return [self._public_memory(item) for item in selected]

    @staticmethod
    def _public_memory(item: dict[str, Any]) -> dict[str, Any]:
        allowed = (
            "id", "title", "content", "category", "thread", "tags", "fact_key",
            "protected", "privacy_scope", "created_at", "score", "score_breakdown",
            "vitality", "channels",
        )
        return {key: item.get(key) for key in allowed if key in item}

    def correct_fact(
        self,
        fact_key: str,
        title: str,
        content: str,
        *,
        reason: str,
        privacy_scope: str = "personal",
    ) -> dict[str, Any]:
        if not fact_key.strip() or not reason.strip():
            raise ValueError("fact_key and reason are required")
        with self.connect() as conn:
            previous = conn.execute(
                """
                SELECT id FROM lmc5_curated_memories
                WHERE fact_key=%s AND active_fact AND version_status='current'
                ORDER BY created_at DESC FOR UPDATE
                """,
                (fact_key,),
            ).fetchall()
            new_id, _ = self._upsert_memory(
                conn,
                {
                    "source": "claude_web_correction",
                    "category": "core",
                    "title": title,
                    "content": content,
                    "fact_key": fact_key,
                    "active_fact": True,
                    "protected": False,
                    "privacy_scope": privacy_scope,
                    "surface_allowed": privacy_scope not in {"sensitive", "secret"},
                    "weight": 2.4,
                    "confidence": 1.0,
                },
            )
            old_ids = [int(row["id"]) for row in previous]
            if old_ids:
                conn.execute(
                    """
                    UPDATE lmc5_curated_memories
                    SET version_status='superseded',active_fact=FALSE,superseded_by=%s,
                        invalid_at=NOW(),updated_at=NOW()
                    WHERE id=ANY(%s)
                    """,
                    (new_id, old_ids),
                )
                for old_id in old_ids:
                    conn.execute(
                        """
                        INSERT INTO lmc5_z_audit(stale_id,current_id,fact_key,reason,status,reviewed_at)
                        VALUES (%s,%s,%s,%s,'approved',NOW())
                        """,
                        (old_id, new_id, fact_key, reason),
                    )
        return {"memory_id": new_id, "superseded_ids": old_ids}

    def refresh_pulse(self, limit: int = 2) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM lmc5_curated_memories
                WHERE version_status='current' AND surface_allowed
                  AND privacy_scope IN ('personal','public')
                  AND category NOT IN ('health','legal','knowledge','tasks','conversation')
                ORDER BY created_at DESC LIMIT 500
                """
            ).fetchall()
            if not rows:
                conn.execute("DELETE FROM lmc5_perception_cache")
                return []
            candidates = [(vitality(dict(row)), dict(row)) for row in rows]
            candidates.sort(key=lambda pair: pair[0], reverse=True)
            high_pool = candidates[: min(60, len(candidates))]
            chosen: list[tuple[float, dict[str, Any], str]] = []
            if high_pool:
                weights = [max(0.01, score) for score, _ in high_pool]
                score, row = random.choices(high_pool, weights=weights, k=1)[0]
                chosen.append((score, row, "high_vitality"))
            if limit > 1:
                remaining = [(score, row) for score, row in candidates if not chosen or row["id"] != chosen[0][1]["id"]]
                if remaining:
                    score, row = random.choice(remaining)
                    chosen.append((score, row, "drift"))
            conn.execute("DELETE FROM lmc5_perception_cache")
            for score, row, via in chosen[:limit]:
                conn.execute(
                    """
                    INSERT INTO lmc5_perception_cache(memory_id,vitality,selected_via,expires_at)
                    VALUES (%s,%s,%s,NOW()+INTERVAL '12 hours')
                    """,
                    (row["id"], score, via),
                )
        return [
            {**self._public_memory(row), "vitality": score, "selected_via": via}
            for score, row, via in chosen[:limit]
        ]

    def pulse(self, limit: int = 2) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT m.*,p.vitality,p.selected_via
                FROM lmc5_perception_cache p JOIN lmc5_curated_memories m ON m.id=p.memory_id
                WHERE p.expires_at>NOW() ORDER BY p.generated_at DESC LIMIT %s
                """,
                (limit,),
            ).fetchall()
        if not rows:
            return self.refresh_pulse(limit)
        return [
            {**self._public_memory(dict(row)), "selected_via": row["selected_via"]}
            for row in rows
        ]

    def import_records(
        self,
        *,
        source_type: str,
        archive_sha256: str,
        documents: Sequence[dict[str, Any]],
        memories: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        created = reused = archived = same_document_relations = 0
        document_ids: dict[str, int] = {}
        memory_ids_by_document: dict[int, list[int]] = {}
        legacy_ids_by_document: dict[int, list[str]] = {}
        with self.connect() as conn:
            for document in documents:
                row = conn.execute(
                    """
                    INSERT INTO lmc5_source_documents(
                      source_type,source_name,original_filename,sha256,content,document_date,metadata
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (sha256) DO UPDATE SET source_name=EXCLUDED.source_name
                    RETURNING id
                    """,
                    (
                        source_type,
                        document.get("source_name", source_type),
                        document["filename"],
                        document["sha256"],
                        document["content"],
                        document.get("document_date"),
                        Jsonb(document.get("metadata") or {}),
                    ),
                ).fetchone()
                document_ids[document["key"]] = int(row["id"])

            for memory in memories:
                data = dict(memory)
                if key := data.pop("document_key", None):
                    data["source_document_id"] = document_ids.get(key)
                memory_id, is_new = self._upsert_memory(conn, data)
                document_id = data.get("source_document_id")
                if document_id:
                    memory_ids_by_document.setdefault(int(document_id), []).append(memory_id)
                    if data.get("legacy_id"):
                        legacy_ids_by_document.setdefault(int(document_id), []).append(
                            str(data["legacy_id"])
                        )
                created += int(is_new)
                reused += int(not is_new)

            if source_type == "ltm":
                # Parser upgrades can merge several old bullet-sized records into
                # one semantic section. Archive stale units from the same source
                # document so re-importing fixes an existing database in place.
                for document_id, legacy_ids in legacy_ids_by_document.items():
                    result = conn.execute(
                        """
                        UPDATE lmc5_curated_memories
                        SET version_status='archived',updated_at=NOW()
                        WHERE source_document_id=%s AND legacy_source='ltm'
                          AND version_status='current' AND legacy_id <> ALL(%s)
                        """,
                        (document_id, legacy_ids),
                    )
                    archived += max(0, int(result.rowcount or 0))

                # Sections from one Day patch share a decision/time context even
                # when classifiers place them in legal, worklog, and episode.
                for document_id, memory_ids in memory_ids_by_document.items():
                    unique_ids = sorted(set(memory_ids))
                    for index, source_id in enumerate(unique_ids):
                        for target_id in unique_ids[index + 1 :]:
                            inserted = conn.execute(
                                """
                                INSERT INTO lmc5_memory_relations(
                                  source_id,target_id,relation_type,strength,reason,status
                                ) VALUES (%s,%s,'same_ltm_patch',0.72,%s,'current')
                                ON CONFLICT (source_id,target_id,relation_type) DO NOTHING
                                RETURNING id
                                """,
                                (
                                    source_id,
                                    target_id,
                                    f"same imported LTM document {document_id}",
                                ),
                            ).fetchone()
                            same_document_relations += int(bool(inserted))
            conn.execute(
                """
                INSERT INTO lmc5_import_runs(
                  source_type,archive_sha256,dry_run,file_count,memory_count,created_count,reused_count,details
                ) VALUES (%s,%s,FALSE,%s,%s,%s,%s,%s)
                """,
                (
                    source_type,
                    archive_sha256,
                    len(documents),
                    len(memories),
                    created,
                    reused,
                    Jsonb(
                        {
                            "document_ids": list(document_ids.values()),
                            "archived_stale_memories": archived,
                            "same_document_relations_created": same_document_relations,
                        }
                    ),
                ),
            )
        return {
            "documents": len(documents),
            "memories": len(memories),
            "created": created,
            "reused": reused,
            "archived_stale_memories": archived,
            "same_document_relations_created": same_document_relations,
        }

    def build_cross_source_relations(self, auto_threshold: float, review_threshold: float) -> dict[str, int]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT o.id AS ob_id,l.id AS ltm_id,o.category,
                  similarity(COALESCE(o.title,'')||' '||o.content, COALESCE(l.title,'')||' '||l.content) AS score
                FROM lmc5_curated_memories o
                JOIN lmc5_curated_memories l ON l.legacy_source='ltm'
                WHERE o.legacy_source='ombre_brain'
                  AND similarity(COALESCE(o.title,'')||' '||o.content, COALESCE(l.title,'')||' '||l.content) >= %s
                  AND (o.created_at::date BETWEEN l.created_at::date-4 AND l.created_at::date+4
                       OR l.created_at IS NULL OR o.created_at IS NULL)
                ORDER BY score DESC
                """,
                (review_threshold,),
            ).fetchall()
            current = review = 0
            for row in rows:
                status = "current" if float(row["score"]) >= auto_threshold else "review"
                relation_type = "emotional_link" if row["category"] in {"fragments", "ob_permanent"} else "same_event"
                inserted = conn.execute(
                    """
                    INSERT INTO lmc5_memory_relations(source_id,target_id,relation_type,strength,reason,status)
                    VALUES (%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (source_id,target_id,relation_type) DO NOTHING RETURNING id
                    """,
                    (
                        min(row["ob_id"], row["ltm_id"]),
                        max(row["ob_id"], row["ltm_id"]),
                        relation_type,
                        min(1.0, float(row["score"]) * 1.5),
                        "legacy cross-source similarity with date gate",
                        status,
                    ),
                ).fetchone()
                if inserted:
                    current += int(status == "current")
                    review += int(status == "review")
        return {"current_relations_created": current, "review_relations_created": review}

    def stats(self) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                  count(*) AS memories,
                  count(*) FILTER (WHERE protected) AS protected,
                  count(*) FILTER (WHERE privacy_scope='sensitive') AS sensitive,
                  count(*) FILTER (WHERE version_status='review') AS memory_review,
                  count(*) FILTER (WHERE active_fact AND version_status='current') AS current_facts,
                  (SELECT count(*) FROM lmc5_source_documents) AS documents,
                  (SELECT count(*) FROM lmc5_raw_events) AS events,
                  (SELECT count(*) FROM lmc5_memory_relations WHERE status='current') AS relations,
                  (SELECT count(*) FROM lmc5_memory_relations WHERE status='review') AS relation_review
                FROM lmc5_curated_memories
                """
            ).fetchone()
            categories = conn.execute(
                "SELECT category,count(*) AS count FROM lmc5_curated_memories GROUP BY category ORDER BY count DESC"
            ).fetchall()
        return {**dict(row), "categories": [dict(item) for item in categories]}

    @staticmethod
    def _dashboard_source_sql() -> str:
        return """
            COALESCE(
              d.source_type,
              CASE
                WHEN m.legacy_source='ombre_brain' THEN 'ombre_brain'
                WHEN m.legacy_source='ltm' THEN 'ltm'
                ELSE 'manual'
              END
            )
        """

    @staticmethod
    def _dashboard_json_value(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Decimal):
            return float(value)
        return value

    @classmethod
    def _dashboard_row(cls, row: dict[str, Any]) -> dict[str, Any]:
        return {key: cls._dashboard_json_value(value) for key, value in dict(row).items()}

    def dashboard_stats(self) -> dict[str, Any]:
        """Return owner-only counts used by the small-home memory dashboard."""
        source_sql = self._dashboard_source_sql()
        with self.connect() as conn:
            totals = conn.execute(
                """
                SELECT
                  count(*) FILTER (WHERE version_status='current') AS memories,
                  count(*) FILTER (
                    WHERE version_status='current' AND privacy_scope IN ('sensitive','secret')
                  ) AS sensitive,
                  count(*) FILTER (WHERE version_status='current' AND protected) AS protected,
                  count(*) FILTER (WHERE version_status='review') AS review,
                  (SELECT count(*) FROM lmc5_source_documents) AS documents,
                  (SELECT count(*) FROM lmc5_raw_events) AS events,
                  (SELECT count(*) FROM lmc5_memory_relations WHERE status='current') AS relations
                FROM lmc5_curated_memories
                """
            ).fetchone()
            categories = conn.execute(
                """
                SELECT category,count(*) AS count
                FROM lmc5_curated_memories
                WHERE version_status='current'
                GROUP BY category ORDER BY count DESC,category
                """
            ).fetchall()
            sources = conn.execute(
                f"""
                SELECT {source_sql} AS source_type,count(*) AS count
                FROM lmc5_curated_memories m
                LEFT JOIN lmc5_source_documents d ON d.id=m.source_document_id
                WHERE m.version_status='current'
                GROUP BY 1 ORDER BY 2 DESC,1
                """
            ).fetchall()
            privacy = conn.execute(
                """
                SELECT privacy_scope,count(*) AS count
                FROM lmc5_curated_memories
                WHERE version_status='current'
                GROUP BY privacy_scope ORDER BY count DESC,privacy_scope
                """
            ).fetchall()
        return {
            **self._dashboard_row(totals),
            "categories": [self._dashboard_row(item) for item in categories],
            "sources": [self._dashboard_row(item) for item in sources],
            "privacy": [self._dashboard_row(item) for item in privacy],
        }

    def list_memories(
        self,
        *,
        query: str = "",
        source_type: str = "",
        category: str = "",
        include_sensitive: bool = False,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Browse curated memories without changing hit counts or recall vitality."""
        query = query.strip()[:500]
        source_type = source_type.strip().lower()
        category = category.strip().lower()
        if category and category not in ALLOWED_CATEGORIES:
            raise ValueError(f"unsupported category: {category}")
        if source_type and source_type not in {"ltm", "ombre_brain", "manual"}:
            raise ValueError("source_type must be ltm, ombre_brain, or manual")

        source_sql = self._dashboard_source_sql()
        conditions = ["m.version_status='current'"]
        params: list[Any] = []
        if not include_sensitive:
            conditions.append("m.privacy_scope IN ('personal','public')")
        if source_type:
            conditions.append(f"{source_sql}=%s")
            params.append(source_type)
        if category:
            conditions.append("m.category=%s")
            params.append(category)
        if query:
            pattern = f"%{query}%"
            conditions.append(
                "(m.title ILIKE %s OR m.content ILIKE %s OR "
                "COALESCE(d.original_filename,'') ILIKE %s OR %s=ANY(m.tags))"
            )
            params.extend((pattern, pattern, pattern, query))

        where_sql = " AND ".join(conditions)
        order_sql = "COALESCE(m.valid_at,m.created_at) DESC,m.id DESC"
        order_params: list[Any] = []
        if query:
            order_sql = (
                "similarity(COALESCE(m.title,'') || ' ' || m.content,%s) DESC," + order_sql
            )
            order_params.append(query)

        with self.connect() as conn:
            total = int(
                conn.execute(
                    f"""
                    SELECT count(*) AS count
                    FROM lmc5_curated_memories m
                    LEFT JOIN lmc5_source_documents d ON d.id=m.source_document_id
                    WHERE {where_sql}
                    """,
                    params,
                ).fetchone()["count"]
            )
            rows = conn.execute(
                f"""
                SELECT
                  m.id,{source_sql} AS source_type,m.source,m.category,m.title,m.content,
                  m.thread,m.tags,m.metadata,m.weight,m.original_importance,m.hit_count,
                  m.last_hit,m.depth,m.activation_boost,m.protected,m.privacy_scope,
                  m.surface_allowed,m.fact_key,m.active_fact,m.created_at,m.updated_at,
                  m.valid_at,d.id AS source_document_id,d.original_filename,d.document_date
                FROM lmc5_curated_memories m
                LEFT JOIN lmc5_source_documents d ON d.id=m.source_document_id
                WHERE {where_sql}
                ORDER BY {order_sql}
                LIMIT %s OFFSET %s
                """,
                [*params, *order_params, limit, offset],
            ).fetchall()

        items: list[dict[str, Any]] = []
        for row in rows:
            raw = dict(row)
            raw["vitality"] = round(float(vitality(raw)), 4)
            items.append(self._dashboard_row(raw))
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    def list_source_documents(
        self,
        *,
        query: str = "",
        source_type: str = "",
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List imported LTM/OB source files with bounded previews."""
        query = query.strip()[:500]
        source_type = source_type.strip().lower()
        if source_type and source_type not in {"ltm", "ombre_brain"}:
            raise ValueError("source_type must be ltm or ombre_brain")
        conditions = ["TRUE"]
        params: list[Any] = []
        if source_type:
            conditions.append("source_type=%s")
            params.append(source_type)
        if query:
            pattern = f"%{query}%"
            conditions.append(
                "(source_name ILIKE %s OR original_filename ILIKE %s OR content ILIKE %s)"
            )
            params.extend((pattern, pattern, pattern))
        where_sql = " AND ".join(conditions)
        with self.connect() as conn:
            total = int(
                conn.execute(
                    f"SELECT count(*) AS count FROM lmc5_source_documents WHERE {where_sql}",
                    params,
                ).fetchone()["count"]
            )
            rows = conn.execute(
                f"""
                SELECT id,source_type,source_name,original_filename,document_date,created_at,
                       metadata,char_length(content) AS characters,
                       left(content,480) AS preview
                FROM lmc5_source_documents
                WHERE {where_sql}
                ORDER BY COALESCE(document_date,created_at) DESC,id DESC
                LIMIT %s OFFSET %s
                """,
                [*params, limit, offset],
            ).fetchall()
        return {
            "items": [self._dashboard_row(row) for row in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def get_source_document(self, document_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id,source_type,source_name,original_filename,content,document_date,
                       metadata,created_at,char_length(content) AS characters
                FROM lmc5_source_documents WHERE id=%s
                """,
                (document_id,),
            ).fetchone()
        return self._dashboard_row(row) if row else None

    @staticmethod
    def content_sha256(content: bytes | str) -> str:
        raw = content.encode("utf-8") if isinstance(content, str) else content
        return hashlib.sha256(raw).hexdigest()
