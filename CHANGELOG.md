# Changelog

## 0.3.5 - 2026-07-22

- Add an owner-only pending dream-candidate review list with title and source evidence.
- Allow explicit approve to promote a candidate into curated memory while keeping evidence IDs.
- Allow explicit reject to discard a candidate from the queue without deleting raw transcript evidence.
- Expose the pending count and add Cloudflare Worker routes for candidate review.

## 0.3.4 - 2026-07-21

- Turn dashboard archive into a visible, recoverable recycle bin without exposing
  importer-created cold archives.
- Allow protected non-current-fact memories to be explicitly unlocked so normal
  vitality decay resumes and the memory can later be archived.
- Replace an active current fact with an existing curated memory while preserving
  the old fact as a non-surfacing superseded historical version.
- Audit recycle-bin restore, protection changes, and current-fact replacement.

## 0.3.3 - 2026-07-21

- Import Claude account data exports as ordered raw events using stable conversation and message UUIDs.
- Keep only user-visible text; exclude thinking blocks, tool calls/results, and credential-pattern messages.
- Make repeated full exports incremental: update known UUIDs, create only new messages, and never infer deletion.
- Add raw-event lexical fallback plus a bounded five-message neighborhood in original conversation order.
- Keep raw transcript evidence separate from curated memory, graph associations, and cold archived fallback.
- Raise the default compressed import limit to 64 MB and the safe uncompressed ceiling to 150 MB.

## 0.3.2 - 2026-07-21

- Raise the default local-hash Nap relation threshold from 0.18 to 0.24 and expose it as
  `LMC5_NAP_RELATION_THRESHOLD`.
- Add `memory_bootstrap` so every new window receives a deterministic short history skeleton
  before recent and spontaneous memory.
- Add explicit layered recall roles for current authority, same-source navigation, graph
  association, and cold archived fallback.
- Add owner-only dashboard APIs for bounded weight edits and audited, recoverable archive.
- Block dashboard archive for protected memories and active current facts.
- Add Cloudflare Worker forwarding and Pages controls for weight changes and archive.

## 0.3.1 - 2026-07-20

- Add private local-hash vector recall that works without pgvector or an external API key.
- Add bounded hourly Nap backfill for changed memories and safe same-privacy orphan linking.
- Add evidence-preserving night-dream candidates with `dry_run` as the default mode.
- Add optional Gemini embedding and structured dream providers behind explicit configuration.
- Add a daily read-only metabolism patrol; it reports missing vectors, orphans and duplicates
  but never deletes memory.
- Add `memory_maintenance_status` for on-demand vector, dream-candidate and patrol diagnostics.
- Keep maintenance failures isolated so lexical, entity, graph and vitality recall stay available.

## 0.3.0 - 2026-07-18

- Add owner-token-protected REST endpoints for memory dashboard statistics.
- Add paginated browsing and search for curated LTM/Ombre memories.
- Add paginated source-document browsing with full LTM document details.
- Add the Cloudflare small-home Worker proxy and Memory-page integration guide.
- Keep all bullets under one semantic LTM subsection instead of creating one-line memories.
- Rewrite generic daily-patch titles and extract conservative entity/topic tags locally.
- Expand multi-term queries with concept coverage and common employment/legal synonyms.
- Keep explicitly requested sensitive memories at normal recall weight.
- Link sections from the same LTM Day patch and archive stale split records on re-import.
- Fix PostgreSQL source statistics grouping for the owner memory dashboard.
- Reserve explicit sensitive matches without applying a privacy score penalty.
- Rewrite weak daily-patch titles with deterministic local body hints.
- Make `memory_time` on-demand and change the reply header to `[YY/MM/DD｜HH:MM]`.

## Unreleased

### Claude Web 0.2.0
- Added a private startup-identity-kernel workflow so Project Instructions know
  who the assistant and person are before any memory lookup.
- Rebalanced web recall to 45% lexical relevance, 30% Ombre vitality, and 25%
  explicit event recency, with auditable score breakdowns and a recent-candidate pool.
- Replaced flat LTM keyword routing with structure-first classification,
  confidence metadata, reviewable ambiguity, and safe reclassification on re-import.
- Added the `memory_time` MCP tool for actual Beijing timestamps.

### Added
- **Refined Session Carryover / 精炼续窗.** New Claude Code helper
  `extras/claude_code/refined_session_carryover.py` plus
  `docs/REFINED_SESSION_CARRYOVER.md`. This replaces the old
  "keep the last 80k-100k tokens" transcript tail-cache recommendation with a
  filtered resume bridge: keep high-signal memory/state and a short natural
  tail, drop tool logs / tracebacks / hook dumps / paths / long JSON, and fail
  closed on recent AUP/policy poison.
- **Heartbeat detector.** New module
  `extras/pgvector_backend/heartbeat_detector.py` — automatically detects
  heartbeat moments (intimacy, physical reactions, nickname shifts) and
  emotional fragments (breakdown, crying, late-night emo, self-denial) from
  conversation chunks. Bilingual keyword gate (CN+EN) + optional LLM
  confirmation. Outputs `HeartbeatCandidate` / `EmotionCandidate` with
  `protected=True`, designed to feed into hippocampus as an extra candidate
  source alongside the LLM proposer. This closes the gap between "what the
  dream pass knows how to extract" (facts, events, preferences) and "what a
  persona deployment actually needs to preserve" (intimate moments and
  emotional peaks that define the relationship).
- **Query Expansion.** `RecallPipeline` now accepts an optional
  `query_expand` callable (Stage 0) that rewrites the user message into
  2–4 search angles before the three-tier cascade. Each expanded query
  feeds independently into vector/FTS/raw-events stages, with results
  merged by `source_id` keeping the highest score. Includes
  `query_expand_adapter()` helper for DeepSeek / any LLM.
- **Raw events FTS fallback.** Third tier in the recall cascade — searches
  the raw event journal when vector top score < 0.30. Catches keywords that
  only appeared in raw conversation turns, never promoted to curated memory.

### Changed
- **Recall fusion default: RRF.** `RecallPipeline` and the production hook now
  default to `fusion="rrf"` after a 726-real-trace A/B replay showed better top5
  composition than `minmax`. `minmax` remains selectable, with docs noting its
  tail-collapse trade-off.

### Fixed
- **E-axis trigger layer.** The 0.2.0 release shipped `EAxisScorer` (which
  decides *how* to score) but forgot the layer that decides *which memories
  should be scored at all*. Without it, `night_dream` never invokes the
  scorer on write, and the E columns stay NULL unless the caller wires it
  manually. New module `extras/pgvector_backend/e_axis_trigger.py`:
  - `should_score_e_axis(candidate)` — type / keyword / relation-hint gate.
    `relationship_moment`, `risk_boundary`, `preference` always fire; `fact`
    and `engineering_decision` skip unless emotion keywords or
    `emotional_link` hint say otherwise.
  - `EMOTION_TRIGGER_KEYWORDS` — bilingual (CN/EN) keyword dictionary across
    strong-emotion / relational / tension / physical-reaction categories.
  - `EAxisDispatcher` — chains gate → scorer → write-back with full
    exception isolation so a failed E score never blocks the underlying
    memory write.
  - `backfill_e_axis()` — nightly batch helper to score memories that
    landed in the last 24 hours but missed live scoring.
- **`NightDream` integration.** New optional `e_axis_dispatcher`
  constructor argument. When provided, `run(apply=True)` auto-fires the
  dispatcher for every successfully written candidate. Defaults to `None`
  (backward-compatible with 0.2.0).

## 0.2.0 — XYZEM completion: production reference impl + pipe layer

This release turns LMC-5 from a memory schema with one offline impl into a
complete memory **system** with two reference impls (minimal SQLite + production
PostgreSQL/pgvector), a documented store-to-conversation pipeline, and the
operational patterns needed for VPS 7×24 deployment.

**The minimal impl in `src/lmc5/` is unchanged.** This release adds the
production impl alongside it.

### New: production reference (`extras/pgvector_backend/`)

- `vector_pgvector.py` — PostgreSQL + pgvector halfvec + ivfflat ANN.
- `night_dream.py` — LLM-proposed hippocampus with 6-type classifier, safety
  gates (importance / risk / dedup / `max_promote` non-silent truncation),
  relation-hint-driven Y-axis expansion, semantic dedup via vector similarity
  before write to stop cross-batch synonyms.
- `narrative_timeline.py` — Weekly / monthly narrative reflection with
  injectable reflector and deterministic baseline.
- `ob_recall.py` — OB-style scoring with category half-life table, time
  ripple, Russell-distance emotional resonance.
- `e_axis_scorer.py` — Provider-agnostic LLM emotional scorer with categorized
  failure logs, exponential-backoff retry on retryable failures, `min_confidence`
  gate, `is_in_shadow_period(...)` helper.
- `perception.py` — Spontaneous-recall scheduler with high-vitality + drift
  ratios and time-of-day shaping; JSON cache for hook integration.
- `recall_pipeline.py` — Five-channel parallel recall (vector / FTS fallback /
  graph 2-hop / emotion / spontaneous) with merge, dedup, optional rerank.
- `embedders.py` — Adapters for Gemini embedding 2 (3072d), Voyage 1024d line,
  OpenAI text-embedding-3, local sentence-transformers; `get_embedder()`
  auto-picks the first available.
- `rerankers.py` — Adapters for DeepSeek, OpenAI, Voyage rerank-2;
  `get_reranker()` auto-picks.
- `config.py` — `LMC5Config` dataclass centralizes every threshold, batch size,
  top-K, retry knob. `LMC5Config.from_env()` loads from environment variables.
- `schema.sql` — Complete DDL for every table referenced anywhere in the codebase.
- `.env.example` — Environment template (PG / embedder / housekeeper LLM /
  frontend / ops).
- `hooks/{session_start,user_prompt_submit,session_end}.py` — Claude Code hook
  entrypoints.
- Construction-time `TypeError` for all injected callables (catches mistakes at
  `__init__` instead of inside a cron job).

### New documentation

- `docs/PERSONA_MODE.md` — Six policy switches for AI companion deployments
  (identity protected / Z manual gate / E shadow period / category half-lives /
  spontaneous recall / relationship moments protected). Each switch has a
  concrete configuration example.
- `docs/DEEPSEEK_INTEGRATION.md` — Housekeeper LLM role across dream / Z / Y /
  M / E / narrative. Provider-agnostic; swap rationale included.
- `docs/VECTOR_BACKENDS.md` — SQLite vs pgvector trade-offs, Gemini / Voyage
  embedder recommendations, migration notes.
- `docs/DEPLOYMENT.md` — VPS 7×24 shape, cron and systemd timer examples,
  three frontend options (Telegram / WeChat bot / self-hosted), self-grooming
  value proposition.
- `docs/FORGE_AND_SWAP.md` — Forge (session continuity from durable memory) and
  Swap (snapshot-based rollback) reference patterns.
- `docs/HOOKS_AND_RECALL.md` — Complete pipeline from store to conversation:
  three lifecycle hooks, five recall channels, perception scheduling, semantic
  dedup wiring, composed diagram.

### Other

- New cover illustration replacing the previous storybook image.
- `docs/credits.md` — XYZEM-origin note added (deliberately abstract; the
  engineering shape is what is published, not the private deployment).
- `tests/test_extras_import.py` — 19 smoke tests covering import paths, config
  defaults, ob_score basics, perception config shape, and construction-time
  callable validation across all five injectable classes.

### Status

The `extras/pgvector_backend/` subpackage is **alpha**. See
[extras/pgvector_backend/README.md](extras/pgvector_backend/README.md) for
what works end-to-end vs what is deployment-specific (graph_expand and
emotion_resonate need your relation/candidate-pool SQL) vs what is not yet
covered (integration tests against a real PG, performance benchmarks,
automated embedder migration).

## 0.1.0

- Initial LMC-5 reference implementation.
- Added SQLite storage with FTS5 recall.
- Added lightweight SQLite vector index with cosine search.
- Added one-hop relation-expanded recall.
- Added raw event journal and mixed memory/event surfacing.
- Added fact-key supersession.
- Added redacted CLI output.
- Added read-only metabolism patrol checks.
- Added JSONL import/export, demo, tests, and CI.
