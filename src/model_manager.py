"""Centralized model lifecycle against Ollama.

Wraps `ollama.AsyncClient` so every code path uses the same semantics for:

* list / exists  (GET /api/tags)
* pull           (POST /api/pull)
* delete         (DELETE /api/delete)
* show           (GET /api/show)
* create_from_modelfile (POST /api/create with a raw Modelfile body)

The ollama python client does not expose a Modelfile-based create; we call the
raw endpoint directly through the same pooled HTTP client (no new client, no
new connection pool). No shell/subprocess is used — all calls are HTTP.
"""

from __future__ import annotations

from typing import Any, Dict, List

import ollama

from src.logging import logger


class ModelManager:
    """Single path for all model lifecycle operations.

    Not a singleton by design — callers hold one instance alongside their
    `GatewayState` (which owns the underlying AsyncClient).
    """

    def __init__(self, client: ollama.AsyncClient):
        self._client = client

    # ── read ─────────────────────────────────────────────────────
    async def list(self) -> List[str]:
        """Return model names from Ollama's /api/tags."""
        resp = await self._client.list()
        return [
            (getattr(m, "name", None) or getattr(m, "model", None) or str(m))
            for m in (resp.models or [])
        ]

    async def exists(self, name: str) -> bool:
        """True iff Ollama has a model with this exact name (including tag)."""
        names = await self.list()
        return name in names

    async def show(self, name: str) -> Dict[str, Any]:
        """Return the raw Ollama /api/show payload for `name`.

        Raises `ollama.ResponseError` on failure (e.g. model not found).
        """
        resp = await self._client.show(name)
        return resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)

    # ── write ────────────────────────────────────────────────────
    async def pull(self, name: str) -> None:
        """Pull (download) `name` from the Ollama registry."""
        logger.info(f"ModelManager.pull {name}")
        await self._client.pull(model=name, stream=False)

    async def delete(self, name: str) -> None:
        """Delete `name` via Ollama's /api/delete (Ollama manages storage)."""
        logger.info(f"ModelManager.delete {name}")
        await self._client.delete(model=name)

    async def create_from_modelfile(self, name: str, modelfile: str) -> None:
        """Create a model from a raw Modelfile (`FROM` / `RENDERER` / `PARSER` …).

        The ollama python client's `create()` does not accept a Modelfile;
        the Ollama HTTP API does. We reuse the same pooled httpx client via the
        client's raw request helper — no subprocess, no shell string, no new
        connection pool.
        """
        r = await self._client._request_raw(
            "POST", "/api/create",
            json={"name": name, "modelfile": modelfile, "stream": False},
        )
        payload = r.json()
        if isinstance(payload, dict) and payload.get("error"):
            raise ollama.ResponseError(payload["error"])
        logger.info(f"ModelManager.create_from_modelfile {name} ok")

    async def ensure_available(self, name: str) -> bool:
        """Check Ollama first; pull only if missing. Returns True if it was
        already present (no pull performed)."""
        if await self.exists(name):
            logger.info(f"ModelManager.ensure_available {name} already present — no pull")
            return True
        await self.pull(name)
        logger.info(f"ModelManager.ensure_available {name} pulled")
        return False
