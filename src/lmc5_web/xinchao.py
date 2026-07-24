from __future__ import annotations

import asyncio
from typing import Any

import httpx


class XinchaoClient:
    """Small fail-closed client for the optional dynamic-mind service."""

    def __init__(self, base_url: str, token: str, *, timeout_seconds: float = 6.0):
        self.base_url = base_url.strip().rstrip("/")
        self.token = token.strip()
        self.timeout_seconds = timeout_seconds

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.token)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        authenticated: bool = True,
    ) -> dict[str, Any]:
        if not self.configured:
            raise RuntimeError("xinchao is not configured")
        headers = {"Accept": "application/json"}
        if authenticated:
            headers["Authorization"] = f"Bearer {self.token}"
        if payload is not None:
            headers["Content-Type"] = "application/json"
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            follow_redirects=False,
        ) as client:
            response = await client.request(
                method,
                f"{self.base_url}{path}",
                headers=headers,
                json=payload,
            )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("xinchao returned an invalid response")
        return data

    async def health(self) -> dict[str, Any]:
        return await self._request("GET", "/health", authenticated=False)

    async def snapshot(self) -> dict[str, Any]:
        if not self.configured:
            return {"configured": False, "connected": False}
        try:
            health, state, intent, breath = await asyncio.gather(
                self.health(),
                self._request("GET", "/v1/state"),
                self._request("GET", "/v1/intent"),
                self._request("GET", "/v1/breath-context"),
            )
            return {
                "configured": True,
                "connected": True,
                "health": health,
                "state": state,
                "intent": intent.get("intent"),
                "top_drives": intent.get("topDrives") or [],
                "thought_pool": intent.get("thoughtPool") or {"flash": [], "obsessions": []},
                "fatigue": intent.get("fatigue", state.get("fatigue", 0)),
                "breath_context": breath,
            }
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            return {
                "configured": True,
                "connected": False,
                "error": f"{type(exc).__name__}: {str(exc)[:180]}",
            }

    async def conversation_event(
        self,
        *,
        satisfied_drives: list[str] | None = None,
        flash_thoughts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if satisfied_drives:
            payload["satisfiedDrives"] = satisfied_drives[:12]
        if flash_thoughts:
            payload["flashThoughts"] = flash_thoughts[:8]
        return await self._request("POST", "/v1/conversation-event", payload=payload)

