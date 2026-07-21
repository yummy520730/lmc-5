from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo


def _int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    value = default if not raw else int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    value = default if not raw else float(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _choice(name: str, default: str, allowed: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}")
    return value


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    value = _int(name, default, minimum=minimum)
    if value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    value = _float(name, default, minimum=minimum)
    if value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str
    access_token: str
    public_base_url: str
    data_dir: Path
    port: int
    mcp_auth_mode: str
    timezone: str
    max_import_bytes: int
    recall_limit: int
    pulse_size: int
    relation_auto_threshold: float
    relation_review_threshold: float
    embed_provider: str
    embed_model: str
    embed_dimension: int
    gemini_api_key: str
    dream_provider: str
    dream_model: str
    dream_mode: str
    dream_hour: int
    dream_min_importance: float
    nap_interval_minutes: int
    nap_batch_size: int

    @classmethod
    def from_env(cls) -> "Settings":
        auth_mode = os.getenv("LMC5_MCP_AUTH_MODE", "oauth").strip().lower()
        if auth_mode not in {"oauth", "none"}:
            raise ValueError("LMC5_MCP_AUTH_MODE must be 'oauth' or 'none'")
        timezone = os.getenv("TZ", "Asia/Shanghai").strip()
        ZoneInfo(timezone)
        database_url = (
            os.getenv("DATABASE_URL")
            or os.getenv("POSTGRES_URL")
            or os.getenv("POSTGRES_CONNECTION_STRING")
            or ""
        ).strip()
        return cls(
            database_url=database_url,
            access_token=os.getenv("LMC5_ACCESS_TOKEN", "").strip(),
            public_base_url=os.getenv("LMC5_PUBLIC_BASE_URL", "").strip().rstrip("/"),
            data_dir=Path(os.getenv("LMC5_DATA_DIR", "/data")).expanduser(),
            port=_int("PORT", 8080),
            mcp_auth_mode=auth_mode,
            timezone=timezone,
            max_import_bytes=_int("LMC5_MAX_IMPORT_MB", 25) * 1024 * 1024,
            recall_limit=_int("LMC5_RECALL_LIMIT", 8),
            pulse_size=_int("LMC5_PULSE_SIZE", 2),
            relation_auto_threshold=_float("LMC5_RELATION_AUTO_THRESHOLD", 0.45),
            relation_review_threshold=_float("LMC5_RELATION_REVIEW_THRESHOLD", 0.25),
            embed_provider=_choice(
                "LMC5_EMBED_PROVIDER", "local_hash", {"local_hash", "gemini", "none"}
            ),
            embed_model=os.getenv("LMC5_EMBED_MODEL", "").strip(),
            embed_dimension=_int("LMC5_EMBED_DIM", 384, minimum=64),
            gemini_api_key=(
                os.getenv("LMC5_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
            ).strip(),
            dream_provider=_choice(
                "LMC5_DREAM_PROVIDER", "local_evidence", {"local_evidence", "gemini"}
            ),
            dream_model=os.getenv("LMC5_DREAM_MODEL", "gemini-2.5-flash").strip(),
            dream_mode=_choice("LMC5_DREAM_MODE", "dry_run", {"off", "dry_run", "apply"}),
            dream_hour=_bounded_int("LMC5_DREAM_HOUR", 4, 0, 23),
            dream_min_importance=_bounded_float(
                "LMC5_DREAM_MIN_IMPORTANCE", 7.0, 1.0, 10.0
            ),
            nap_interval_minutes=_int("LMC5_NAP_INTERVAL_MINUTES", 60, minimum=5),
            nap_batch_size=_int("LMC5_NAP_BATCH_SIZE", 40, minimum=1),
        )

    def prepare_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def server_base_url(self) -> str:
        return self.public_base_url or f"http://127.0.0.1:{self.port}"

    @property
    def oauth_resource_url(self) -> str:
        return f"{self.server_base_url}/mcp"
