from __future__ import annotations

import hashlib
import html
import json
import secrets
import sqlite3
import time
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    RegistrationError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken


OAUTH_SCOPES = ["lmc5", "mcp:tools", "claudeai"]
ACCESS_TOKEN_SECONDS = 60 * 60
REFRESH_TOKEN_SECONDS = 30 * 24 * 60 * 60
AUTHORIZATION_CODE_SECONDS = 5 * 60
PENDING_LOGIN_SECONDS = 10 * 60
MAX_LOGIN_ATTEMPTS = 5


class OAuthLoginError(ValueError):
    pass


class LMC5RefreshToken(RefreshToken):
    resource: str | None = None
    family_id: str


class LMC5OAuthProvider(
    OAuthAuthorizationServerProvider[AuthorizationCode, LMC5RefreshToken, AccessToken]
):
    """Persistent single-owner OAuth 2.1 provider for Claude remote MCP."""

    def __init__(self, database: Path, issuer_url: str, resource_url: str, owner_password: str):
        self.database = database
        self.issuer_url = issuer_url.rstrip("/")
        self.resource_url = resource_url.rstrip("/")
        self.owner_password = owner_password
        self._lock = Lock()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS oauth_clients (
                    client_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS oauth_pending (
                    request_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS oauth_codes (
                    token_hash TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS oauth_access_tokens (
                    token_hash TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    family_id TEXT NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS oauth_refresh_tokens (
                    token_hash TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    family_id TEXT NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS oauth_access_family_idx ON oauth_access_tokens(family_id);
                CREATE INDEX IF NOT EXISTS oauth_refresh_family_idx ON oauth_refresh_tokens(family_id);
                """
            )

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _password_matches(supplied: str, expected: str) -> bool:
        return bool(expected) and secrets.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))

    def _cleanup(self, db: sqlite3.Connection) -> None:
        now = time.time()
        db.execute("DELETE FROM oauth_pending WHERE expires_at <= ?", (now,))
        db.execute("DELETE FROM oauth_codes WHERE expires_at <= ?", (now,))
        db.execute("DELETE FROM oauth_access_tokens WHERE expires_at <= ?", (now,))
        db.execute("DELETE FROM oauth_refresh_tokens WHERE expires_at <= ?", (now,))

    @staticmethod
    def _redirect_allowed(uri: str) -> bool:
        parsed = urlparse(uri)
        if parsed.fragment or parsed.username or parsed.password:
            return False
        if parsed.scheme == "https" and parsed.hostname in {"claude.ai", "claude.com"}:
            return parsed.path == "/api/mcp/auth_callback"
        return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT payload_json FROM oauth_clients WHERE client_id = ?", (client_id,)).fetchone()
        return OAuthClientInformationFull.model_validate_json(row["payload_json"]) if row else None

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        if not client_info.client_id:
            raise RegistrationError("invalid_client_metadata", "client_id is required")
        redirects = [str(uri) for uri in client_info.redirect_uris or []]
        if not redirects or any(not self._redirect_allowed(uri) for uri in redirects):
            raise RegistrationError(
                "invalid_redirect_uri",
                "Only Claude's OAuth callback or a localhost callback is allowed",
            )

        # Claude has used several scope names across connector versions. This personal
        # server grants the same owner-level capability for each compatibility name.
        client_info.scope = " ".join(OAUTH_SCOPES)
        payload = client_info.model_dump_json()
        with self._lock, self._connect() as db:
            self._cleanup(db)
            db.execute(
                "INSERT OR REPLACE INTO oauth_clients VALUES (?, ?, ?)",
                (client_info.client_id, payload, time.time()),
            )

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        if not client.client_id:
            raise AuthorizeError("invalid_request", "client_id is required")
        if params.resource and params.resource.rstrip("/") != self.resource_url:
            raise AuthorizeError("invalid_request", "resource does not match this MCP server")

        request_id = secrets.token_urlsafe(32)
        pending = {
            "client_id": client.client_id,
            "client_name": client.client_name or "Claude",
            "state": params.state,
            "scopes": params.scopes or ["lmc5"],
            "code_challenge": params.code_challenge,
            "redirect_uri": str(params.redirect_uri),
            "redirect_uri_provided_explicitly": params.redirect_uri_provided_explicitly,
            "resource": params.resource or self.resource_url,
        }
        with self._lock, self._connect() as db:
            self._cleanup(db)
            db.execute(
                "INSERT INTO oauth_pending VALUES (?, ?, 0, ?)",
                (request_id, json.dumps(pending, ensure_ascii=False), time.time() + PENDING_LOGIN_SECONDS),
            )
        return f"{self.issuer_url}/oauth/login?request={request_id}"

    def get_pending_login(self, request_id: str) -> dict | None:
        with self._lock, self._connect() as db:
            self._cleanup(db)
            row = db.execute(
                "SELECT payload_json, attempts FROM oauth_pending WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        if not row:
            return None
        payload = json.loads(row["payload_json"])
        payload["attempts"] = int(row["attempts"])
        return payload

    def complete_authorization(self, request_id: str, password: str) -> str:
        with self._lock, self._connect() as db:
            self._cleanup(db)
            row = db.execute(
                "SELECT payload_json, attempts FROM oauth_pending WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if not row:
                raise OAuthLoginError("授权请求已过期，请回到 Claude 重新点击 Connect。")

            pending = json.loads(row["payload_json"])
            # A mobile browser can lose the first 303 while returning to Claude.
            # Keep the same short-lived callback so another tap can safely retry.
            if completed_redirect := pending.get("completed_redirect"):
                return str(completed_redirect)

            attempts = int(row["attempts"])
            if attempts >= MAX_LOGIN_ATTEMPTS:
                db.execute("DELETE FROM oauth_pending WHERE request_id = ?", (request_id,))
                raise OAuthLoginError("尝试次数过多，请回到 Claude 重新点击 Connect。")
            if not self._password_matches(password, self.owner_password):
                db.execute(
                    "UPDATE oauth_pending SET attempts = attempts + 1 WHERE request_id = ?",
                    (request_id,),
                )
                raise OAuthLoginError("密码不对，请输入 LMC5_ACCESS_TOKEN。")

            scopes = list(dict.fromkeys(["lmc5", *(pending.get("scopes") or [])]))
            raw_code = secrets.token_urlsafe(32)
            code = AuthorizationCode(
                code=raw_code,
                scopes=scopes,
                expires_at=time.time() + AUTHORIZATION_CODE_SECONDS,
                client_id=pending["client_id"],
                code_challenge=pending["code_challenge"],
                redirect_uri=pending["redirect_uri"],
                redirect_uri_provided_explicitly=bool(pending["redirect_uri_provided_explicitly"]),
                resource=pending.get("resource") or self.resource_url,
                subject="lmc5-owner",
            )
            db.execute(
                "INSERT INTO oauth_codes VALUES (?, ?, ?)",
                (
                    self._hash(raw_code),
                    code.model_dump_json(exclude={"code"}),
                    code.expires_at,
                ),
            )
            fields = {"code": raw_code}
            if pending.get("state") is not None:
                fields["state"] = pending["state"]
            completed_redirect = construct_redirect_uri(pending["redirect_uri"], **fields)
            pending["completed_redirect"] = completed_redirect
            pending["completed_at"] = time.time()
            db.execute(
                "UPDATE oauth_pending SET payload_json = ? WHERE request_id = ?",
                (json.dumps(pending, ensure_ascii=False), request_id),
            )

        return completed_redirect

    async def load_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: str,
    ) -> AuthorizationCode | None:
        with self._lock, self._connect() as db:
            self._cleanup(db)
            row = db.execute(
                "SELECT payload_json FROM oauth_codes WHERE token_hash = ?",
                (self._hash(authorization_code),),
            ).fetchone()
        if not row:
            return None
        payload = json.loads(row["payload_json"])
        return AuthorizationCode(code=authorization_code, **payload)

    def _store_token_pair(
        self,
        db: sqlite3.Connection,
        client_id: str,
        scopes: list[str],
        resource: str | None,
        subject: str | None,
    ) -> OAuthToken:
        now = int(time.time())
        family_id = secrets.token_urlsafe(24)
        raw_access = secrets.token_urlsafe(32)
        raw_refresh = secrets.token_urlsafe(48)
        access = AccessToken(
            token=raw_access,
            client_id=client_id,
            scopes=scopes,
            expires_at=now + ACCESS_TOKEN_SECONDS,
            resource=resource or self.resource_url,
            subject=subject or "lmc5-owner",
            claims={"iss": self.issuer_url, "family_id": family_id},
        )
        refresh = LMC5RefreshToken(
            token=raw_refresh,
            client_id=client_id,
            scopes=scopes,
            expires_at=now + REFRESH_TOKEN_SECONDS,
            subject=subject or "lmc5-owner",
            resource=resource or self.resource_url,
            family_id=family_id,
        )
        db.execute(
            "INSERT INTO oauth_access_tokens VALUES (?, ?, ?, ?)",
            (
                self._hash(raw_access),
                access.model_dump_json(exclude={"token"}),
                family_id,
                access.expires_at,
            ),
        )
        db.execute(
            "INSERT INTO oauth_refresh_tokens VALUES (?, ?, ?, ?)",
            (
                self._hash(raw_refresh),
                refresh.model_dump_json(exclude={"token"}),
                family_id,
                refresh.expires_at,
            ),
        )
        return OAuthToken(
            access_token=raw_access,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_SECONDS,
            scope=" ".join(scopes),
            refresh_token=raw_refresh,
        )

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: AuthorizationCode,
    ) -> OAuthToken:
        with self._lock, self._connect() as db:
            deleted = db.execute(
                "DELETE FROM oauth_codes WHERE token_hash = ?",
                (self._hash(authorization_code.code),),
            ).rowcount
            if deleted != 1:
                raise ValueError("authorization code was already used")
            return self._store_token_pair(
                db,
                authorization_code.client_id,
                authorization_code.scopes,
                authorization_code.resource,
                authorization_code.subject,
            )

    async def load_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: str,
    ) -> LMC5RefreshToken | None:
        with self._lock, self._connect() as db:
            self._cleanup(db)
            row = db.execute(
                "SELECT payload_json FROM oauth_refresh_tokens WHERE token_hash = ?",
                (self._hash(refresh_token),),
            ).fetchone()
        if not row:
            return None
        payload = json.loads(row["payload_json"])
        return LMC5RefreshToken(token=refresh_token, **payload)

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: LMC5RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        with self._lock, self._connect() as db:
            deleted = db.execute(
                "DELETE FROM oauth_refresh_tokens WHERE token_hash = ?",
                (self._hash(refresh_token.token),),
            ).rowcount
            if deleted != 1:
                raise ValueError("refresh token was already used")
            db.execute("DELETE FROM oauth_access_tokens WHERE family_id = ?", (refresh_token.family_id,))
            return self._store_token_pair(
                db,
                refresh_token.client_id,
                list(dict.fromkeys(["lmc5", *scopes])),
                refresh_token.resource,
                refresh_token.subject,
            )

    async def load_access_token(self, token: str) -> AccessToken | None:
        if self._password_matches(token, self.owner_password):
            return AccessToken(
                token=token,
                client_id="lmc5-static-owner",
                scopes=OAUTH_SCOPES,
                resource=self.resource_url,
                subject="lmc5-owner",
                claims={"iss": self.issuer_url},
            )

        with self._lock, self._connect() as db:
            self._cleanup(db)
            row = db.execute(
                "SELECT payload_json FROM oauth_access_tokens WHERE token_hash = ?",
                (self._hash(token),),
            ).fetchone()
        if not row:
            return None
        payload = json.loads(row["payload_json"])
        return AccessToken(token=token, **payload)

    async def revoke_token(self, token: AccessToken | LMC5RefreshToken) -> None:
        if self._password_matches(token.token, self.owner_password):
            return
        table = "oauth_refresh_tokens" if isinstance(token, RefreshToken) else "oauth_access_tokens"
        with self._lock, self._connect() as db:
            row = db.execute(
                f"SELECT family_id FROM {table} WHERE token_hash = ?",  # noqa: S608 - table is fixed above
                (self._hash(token.token),),
            ).fetchone()
            if not row:
                return
            db.execute("DELETE FROM oauth_access_tokens WHERE family_id = ?", (row["family_id"],))
            db.execute("DELETE FROM oauth_refresh_tokens WHERE family_id = ?", (row["family_id"],))


def render_oauth_login(request_id: str, pending: dict | None, error: str = "") -> str:
    if not pending:
        content = """
        <h1>这次授权已经失效</h1>
        <p>请回到 Claude，重新点击 <strong>Connect</strong>。</p>
        """
    elif completed_redirect := pending.get("completed_redirect"):
        safe_redirect = html.escape(str(completed_redirect), quote=True)
        content = f"""
        <span class="pill">OAuth 2.1 · PKCE</span>
        <h1>已经允许连接</h1>
        <p>如果浏览器没有自动回到 Claude，请点击下面的按钮继续。授权码只有几分钟有效。</p>
        <a class="continue" href="{safe_redirect}">继续回到 Claude</a>
        """
    else:
        client_name = html.escape(str(pending.get("client_name") or "Claude"))
        safe_request = html.escape(request_id, quote=True)
        error_html = f'<p class="error">{html.escape(error)}</p>' if error else ""
        content = f"""
        <span class="pill">OAuth 2.1 · PKCE</span>
        <h1>让 {client_name} 连接 LMC-5？</h1>
        <p>输入你在 Zeabur 设置的 <code>LMC5_ACCESS_TOKEN</code>。密码只交给你的记忆服务，不会发送给 Claude。</p>
        {error_html}
        <form method="post" action="/oauth/login">
          <input type="hidden" name="request" value="{safe_request}">
          <label for="password">LMC-5 访问密码</label>
          <input id="password" name="password" type="password" autocomplete="current-password" required autofocus>
          <button type="submit">允许连接</button>
        </form>
        """
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>授权连接 · LMC-5</title><style>
:root{{color-scheme:dark;--bg:#0d0d11;--panel:#191821;--line:#302c3b;--text:#f4f0f8;--muted:#b8afc4;--accent:#c8a8ff}}
*{{box-sizing:border-box}}body{{margin:0;min-height:100vh;display:grid;place-items:center;padding:20px;font-family:Inter,system-ui,-apple-system,"PingFang SC",sans-serif;background:radial-gradient(circle at 25% 0,#2b1b3d 0,transparent 42%),var(--bg);color:var(--text)}}
main{{width:min(480px,100%);padding:28px;background:var(--panel);border:1px solid var(--line);border-radius:24px;box-shadow:0 22px 70px #0008}}h1{{font-size:28px;margin:16px 0 10px}}p{{color:var(--muted);line-height:1.7}}code{{color:#e5d5ff}}label{{display:block;margin:22px 0 8px;color:var(--muted);font-size:13px}}input,button,.continue{{width:100%;padding:14px;border-radius:13px;font:inherit}}input{{background:#0f0e14;border:1px solid var(--line);color:var(--text)}}button,.continue{{margin-top:16px;border:0;background:var(--accent);color:#21152c;font-weight:800}}.continue{{display:block;text-align:center;text-decoration:none}}.pill{{font-size:12px;color:var(--muted);border:1px solid var(--line);padding:6px 10px;border-radius:999px}}.error{{color:#ffaca8}}
</style></head><body><main>{content}</main></body></html>"""
