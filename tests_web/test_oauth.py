import base64
import hashlib
import secrets
from urllib.parse import parse_qs, urlparse

import httpx

from lmc5_web.oauth import LMC5OAuthProvider, OAuthLoginError, OAUTH_SCOPES, render_oauth_login
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
from mcp.server.fastmcp import FastMCP


def _oauth_app(provider: LMC5OAuthProvider):
    server = FastMCP(
        "OAuth test",
        stateless_http=True,
        json_response=True,
        auth_server_provider=provider,
        auth=AuthSettings(
            issuer_url=provider.issuer_url,
            client_registration_options=ClientRegistrationOptions(
                enabled=True,
                valid_scopes=OAUTH_SCOPES,
                default_scopes=["lmc5"],
            ),
            revocation_options=RevocationOptions(enabled=True),
            required_scopes=["lmc5"],
            resource_server_url=provider.resource_url,
        ),
    )
    server.settings.streamable_http_path = "/mcp"
    return server.streamable_http_app()


async def test_oauth_dcr_pkce_and_refresh_flow(tmp_path):
    base_url = "https://memory.example"
    provider = LMC5OAuthProvider(
        tmp_path / "oauth.sqlite3",
        issuer_url=base_url,
        resource_url=f"{base_url}/mcp",
        owner_password="owner-password",
    )
    app = _oauth_app(provider)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url=base_url, follow_redirects=False) as client:
        metadata = await client.get("/.well-known/oauth-authorization-server")
        assert metadata.status_code == 200
        assert metadata.json()["registration_endpoint"] == f"{base_url}/register"

        protected = await client.get("/.well-known/oauth-protected-resource/mcp")
        assert protected.status_code == 200
        assert protected.json()["resource"] == f"{base_url}/mcp"

        rejected = await client.post(
            "/register",
            json={
                "redirect_uris": ["https://attacker.example/callback"],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
            },
        )
        assert rejected.status_code == 400

        registration = await client.post(
            "/register",
            json={
                "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"],
                "token_endpoint_auth_method": "client_secret_post",
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "client_name": "Claude",
            },
        )
        assert registration.status_code == 201
        client_info = registration.json()

        verifier = secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        authorize = await client.get(
            "/authorize",
            params={
                "client_id": client_info["client_id"],
                "redirect_uri": "https://claude.ai/api/mcp/auth_callback",
                "response_type": "code",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": "state-value",
                "scope": "mcp:tools",
                "resource": f"{base_url}/mcp",
            },
        )
        assert authorize.status_code == 302
        request_id = parse_qs(urlparse(authorize.headers["location"]).query)["request"][0]

        try:
            provider.complete_authorization(request_id, "wrong-password")
            raise AssertionError("wrong password should fail")
        except OAuthLoginError:
            pass

        callback = provider.complete_authorization(request_id, "owner-password")
        callback_params = parse_qs(urlparse(callback).query)
        assert callback_params["state"] == ["state-value"]

        # Losing the first browser redirect must not make a second tap expire.
        assert provider.complete_authorization(request_id, "irrelevant-on-retry") == callback
        completed = provider.get_pending_login(request_id)
        assert completed and completed["completed_redirect"] == callback
        assert "继续回到 Claude" in render_oauth_login(request_id, completed)

        token_response = await client.post(
            "/token",
            data={
                "grant_type": "authorization_code",
                "code": callback_params["code"][0],
                "redirect_uri": "https://claude.ai/api/mcp/auth_callback",
                "client_id": client_info["client_id"],
                "client_secret": client_info["client_secret"],
                "code_verifier": verifier,
                "resource": f"{base_url}/mcp",
            },
        )
        assert token_response.status_code == 200
        first_tokens = token_response.json()
        assert "lmc5" in first_tokens["scope"].split()
        assert await provider.load_access_token(first_tokens["access_token"])

        refresh_response = await client.post(
            "/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": first_tokens["refresh_token"],
                "client_id": client_info["client_id"],
                "client_secret": client_info["client_secret"],
            },
        )
        assert refresh_response.status_code == 200
        refreshed = refresh_response.json()
        assert refreshed["access_token"] != first_tokens["access_token"]
        assert await provider.load_access_token(first_tokens["access_token"]) is None
        assert await provider.load_access_token(refreshed["access_token"])

    # OAuth clients and tokens survive a server restart because they live on /data.
    reloaded = LMC5OAuthProvider(
        tmp_path / "oauth.sqlite3",
        issuer_url=base_url,
        resource_url=f"{base_url}/mcp",
        owner_password="owner-password",
    )
    assert await reloaded.get_client(client_info["client_id"])
    assert await reloaded.load_access_token(refreshed["access_token"])
    assert await reloaded.load_access_token("owner-password")
