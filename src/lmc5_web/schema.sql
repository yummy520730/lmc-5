CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS lmc5_source_documents (
    id BIGSERIAL PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_name TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL,
    document_date TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS lmc5_curated_memories (
    id BIGSERIAL PRIMARY KEY,
    legacy_source TEXT,
    legacy_id TEXT,
    source_document_id BIGINT REFERENCES lmc5_source_documents(id),
    source TEXT NOT NULL,
    category TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    thread TEXT NOT NULL DEFAULT 'other',
    tags TEXT[] NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    weight NUMERIC(7,3) NOT NULL DEFAULT 1.0,
    original_importance NUMERIC(7,3),
    hit_count INTEGER NOT NULL DEFAULT 0,
    last_hit TIMESTAMPTZ,
    depth INTEGER,
    activation_boost NUMERIC(7,3) NOT NULL DEFAULT 0,

    valence REAL,
    arousal REAL,
    tension REAL,
    response_tendency TEXT NOT NULL DEFAULT '',
    growth_delta TEXT NOT NULL DEFAULT '',

    version_status TEXT NOT NULL DEFAULT 'current',
    fact_key TEXT,
    active_fact BOOLEAN NOT NULL DEFAULT FALSE,
    protected BOOLEAN NOT NULL DEFAULT FALSE,
    superseded_by BIGINT REFERENCES lmc5_curated_memories(id),
    valid_at TIMESTAMPTZ,
    invalid_at TIMESTAMPTZ,

    confidence REAL,
    privacy_scope TEXT NOT NULL DEFAULT 'personal',
    surface_allowed BOOLEAN NOT NULL DEFAULT TRUE,
    resolved BOOLEAN NOT NULL DEFAULT FALSE,
    digested BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT lmc5_memory_status_check CHECK (
        version_status IN ('current', 'review', 'superseded', 'historical', 'archived', 'quarantine')
    ),
    CONSTRAINT lmc5_privacy_scope_check CHECK (
        privacy_scope IN ('personal', 'sensitive', 'secret', 'public')
    ),
    CONSTRAINT lmc5_valence_check CHECK (valence IS NULL OR valence BETWEEN -1 AND 1),
    CONSTRAINT lmc5_unit_arousal_check CHECK (arousal IS NULL OR arousal BETWEEN 0 AND 1),
    CONSTRAINT lmc5_unit_tension_check CHECK (tension IS NULL OR tension BETWEEN 0 AND 1),
    CONSTRAINT lmc5_depth_check CHECK (depth IS NULL OR depth BETWEEN 1 AND 5),
    UNIQUE (legacy_source, legacy_id)
);

CREATE INDEX IF NOT EXISTS lmc5_memory_status_idx ON lmc5_curated_memories(version_status);
CREATE INDEX IF NOT EXISTS lmc5_memory_category_idx ON lmc5_curated_memories(category);
CREATE INDEX IF NOT EXISTS lmc5_memory_thread_idx ON lmc5_curated_memories(thread);
CREATE INDEX IF NOT EXISTS lmc5_memory_fact_idx ON lmc5_curated_memories(fact_key)
    WHERE fact_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS lmc5_memory_created_idx ON lmc5_curated_memories(created_at DESC);
CREATE INDEX IF NOT EXISTS lmc5_memory_tags_idx ON lmc5_curated_memories USING GIN(tags);
CREATE INDEX IF NOT EXISTS lmc5_memory_trgm_idx ON lmc5_curated_memories USING GIN(
    ((COALESCE(title, '') || ' ' || content)) gin_trgm_ops
);

CREATE TABLE IF NOT EXISTS lmc5_memory_relations (
    id BIGSERIAL PRIMARY KEY,
    source_id BIGINT NOT NULL REFERENCES lmc5_curated_memories(id) ON DELETE CASCADE,
    target_id BIGINT NOT NULL REFERENCES lmc5_curated_memories(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    strength REAL NOT NULL DEFAULT 0.5,
    reason TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'current',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_until TIMESTAMPTZ,
    CONSTRAINT lmc5_relation_strength_check CHECK (strength BETWEEN 0 AND 1),
    CONSTRAINT lmc5_relation_status_check CHECK (status IN ('current', 'review', 'closed')),
    CONSTRAINT lmc5_relation_no_self CHECK (source_id <> target_id),
    UNIQUE (source_id, target_id, relation_type)
);

CREATE INDEX IF NOT EXISTS lmc5_relations_source_idx ON lmc5_memory_relations(source_id)
    WHERE status = 'current' AND valid_until IS NULL;
CREATE INDEX IF NOT EXISTS lmc5_relations_target_idx ON lmc5_memory_relations(target_id)
    WHERE status = 'current' AND valid_until IS NULL;

CREATE TABLE IF NOT EXISTS lmc5_raw_events (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL,
    channel TEXT NOT NULL DEFAULT 'claude_web',
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, role, content_hash)
);

CREATE INDEX IF NOT EXISTS lmc5_events_created_idx ON lmc5_raw_events(created_at DESC);
CREATE INDEX IF NOT EXISTS lmc5_events_trgm_idx ON lmc5_raw_events USING GIN(content gin_trgm_ops);

CREATE TABLE IF NOT EXISTS lmc5_perception_cache (
    memory_id BIGINT PRIMARY KEY REFERENCES lmc5_curated_memories(id) ON DELETE CASCADE,
    vitality REAL NOT NULL,
    selected_via TEXT NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS lmc5_z_audit (
    id BIGSERIAL PRIMARY KEY,
    stale_id BIGINT REFERENCES lmc5_curated_memories(id),
    current_id BIGINT REFERENCES lmc5_curated_memories(id),
    fact_key TEXT,
    reason TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reviewed_at TIMESTAMPTZ,
    CONSTRAINT lmc5_z_status_check CHECK (status IN ('pending', 'approved', 'rejected'))
);

CREATE TABLE IF NOT EXISTS lmc5_import_runs (
    id BIGSERIAL PRIMARY KEY,
    source_type TEXT NOT NULL,
    archive_sha256 TEXT NOT NULL,
    dry_run BOOLEAN NOT NULL,
    file_count INTEGER NOT NULL,
    memory_count INTEGER NOT NULL,
    created_count INTEGER NOT NULL DEFAULT 0,
    reused_count INTEGER NOT NULL DEFAULT 0,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

DO $$
BEGIN
    BEGIN
        CREATE EXTENSION IF NOT EXISTS vector;
        EXECUTE 'CREATE TABLE IF NOT EXISTS lmc5_vectors (
            memory_id BIGINT PRIMARY KEY REFERENCES lmc5_curated_memories(id) ON DELETE CASCADE,
            embedding vector,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )';
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'pgvector unavailable; lexical and graph recall remain enabled';
    END;
END $$;
