from __future__ import annotations

import importlib

import httpx
import pytest


@pytest.mark.asyncio
async def test_public_surface_and_protected_import_api(monkeypatch, tmp_path):
    monkeypatch.setenv("LMC5_MCP_AUTH_MODE", "none")
    monkeypatch.setenv("LMC5_ACCESS_TOKEN", "test-owner-token")
    monkeypatch.setenv("LMC5_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LMC5_PUBLIC_BASE_URL", "http://testserver")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.delenv("POSTGRES_CONNECTION_STRING", raising=False)

    module = importlib.import_module("lmc5_web.app")
    transport = httpx.ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        homepage = await client.get("/")
        assert homepage.status_code == 200
        assert "让记忆留下来，也让它呼吸" in homepage.text

        health = await client.get("/healthz")
        assert health.status_code == 503
        assert health.json()["status"] == "degraded"

        rejected = await client.post("/api/import/nope")
        assert rejected.status_code == 401
        allowed = await client.post(
            "/api/import/nope", headers={"X-API-Key": "test-owner-token"}
        )
        assert allowed.status_code == 404

        dashboard_rejected = await client.get("/api/dashboard/stats")
        assert dashboard_rejected.status_code == 401

        monkeypatch.setattr(
            module.store,
            "dashboard_stats",
            lambda: {"memories": 12, "documents": 3, "categories": []},
        )
        dashboard = await client.get(
            "/api/dashboard/stats", headers={"Authorization": "Bearer test-owner-token"}
        )
        assert dashboard.status_code == 200
        assert dashboard.json()["memories"] == 12
        assert dashboard.headers["cache-control"] == "no-store"


@pytest.mark.asyncio
async def test_expected_mcp_tools_are_registered():
    module = importlib.import_module("lmc5_web.app")
    tools = await module.mcp.list_tools()
    names = {tool.name for tool in tools}
    assert names == {
        "memory_time",
        "memory_context",
        "memory_remember",
        "memory_checkpoint",
        "memory_correct",
        "memory_pulse",
        "memory_status",
    }
