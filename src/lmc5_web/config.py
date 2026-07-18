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
        )

    def prepare_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def server_base_url(self) -> str:
        return self.public_base_url or f"http://127.0.0.1:{self.port}"

    @property
    def oauth_resource_url(self) -> str:
        return f"{self.server_base_url}/mcp"

