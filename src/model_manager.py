"""Centralized model lifecycle against Ollama.

Wraps `ollama.AsyncClient` so every code path uses the same semantics for:

* list / exists  (GET /api/tags)
* pull           (POST /api/pull)
* delete         (DELETE /api/delete)
* show           (GET /api/show)
* create_from_modelfile (subprocess -> `ollama create <name> -f <Modelfile>`)

All HTTP-bound operations go through the same pooled `ollama.AsyncClient`.
`create_from_modelfile` is the one exception: Ollama's HTTP `/api/create`
accepts a *structured* body (model / from / parameters / ...) that does NOT
express the custom `RENDERER` / `PARSER` directives this build relies on. The
proven mechanism is the Ollama CLI:

    ollama create <derived> -f <Modelfile>

So we invoke the CLI via `subprocess` with an explicit argument list (no
shell, no string interpolation) and a temporary Modelfile that is always
removed -- on both success and failure.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

import ollama

from src.logging import logger


class CreateError(Exception):
    """Raised when `ollama create` fails (CLI not found, non-zero exit, ...).

    Carries the original exit code and stdout/stderr so the caller can log
    the real Ollama error rather than a generic failure.
    """

    def __init__(self, message: str, exit_code: Optional[int] = None,
                 stdout: Optional[str] = None, stderr: Optional[str] = None):
        super().__init__(message)
        self.exit_code = exit_code
        self.stdout = stdout or ""
        self.stderr = stderr or ""


class ModelManager:
    """Single path for all model lifecycle operations.

    Not a singleton by design -- callers hold one instance alongside their
    `GatewayState` (which owns the underlying AsyncClient).
    """

    def __init__(self, client: ollama.AsyncClient):
        self._client = client

    # -- read ----------------------------------------------------
    async def list(self) -> List[str]:
        """Return model names from Ollama's /api/tags."""
        resp = await self._client.list()
        return [
            (getattr(m, "name", None) or getattr(m, "model", None) or str(m))
            for m in (resp.models or [])
        ]

    async def exists(self, name: str) -> bool:
        """True iff Ollama has a model with this name.

        An exact `name:tag` matches as-is. A bare `name` also matches the
        implicit `name:latest` tag - Ollama's /api/tags always returns locally
        created models as `name:latest`, so a bare-name check (e.g. the repair
        ORIGINAL_NOT_FOUND guard, or the CLI install step) would otherwise
        wrongly report the model as missing and re-download it.
        """
        names = set(await self.list())
        if name in names:
            return True
        if ":" not in name:
            return f"{name}:latest" in names
        return False

    async def show(self, name: str) -> Dict[str, Any]:
        """Return the raw Ollama /api/show payload for `name`.

        Raises `ollama.ResponseError` on failure (e.g. model not found).
        """
        resp = await self._client.show(name)
        return resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)

    # -- write ---------------------------------------------------
    async def pull(self, name: str) -> None:
        """Pull (download) `name` from the Ollama registry."""
        logger.info(f"ModelManager.pull {name}")
        await self._client.pull(model=name, stream=False)

    async def delete(self, name: str) -> None:
        """Delete `name` via Ollama's /api/delete (Ollama manages storage)."""
        logger.info(f"ModelManager.delete {name}")
        await self._client.delete(model=name)

    async def create_from_modelfile(self, name: str, modelfile: str) -> None:
        """Create a model from a raw Modelfile via the Ollama CLI.

        The Ollama HTTP `/api/create` on this build uses the *structured*
        form (model / from / parameters / ...) and does not accept the
        `RENDERER` / `PARSER` directives this gateway relies on. The CLI
        (`ollama create <name> -f <Modelfile>`) does, and is the proven path.

        The Modelfile is written to a temporary file (system temp dir, not
        the repository), passed to `ollama create -f`, and removed in a
        `finally` block so it is cleaned up on both success and failure.
        `subprocess.run` is used with an explicit argument list -- no shell.

        Raises `CreateError` (with exit code + stdout + stderr) on failure
        so the caller can surface the real Ollama error.
        """
        if not name:
            raise CreateError("create_from_modelfile: name is required")
        if not modelfile:
            raise CreateError("create_from_modelfile: modelfile is required")

        logger.info(f"ModelManager.create_from_modelfile {name} via CLI")
        tmp_path: Optional[str] = None
        try:
            # delete=False keeps the file on disk until we explicitly unlink
            # it; the context manager only closes the handle.
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8",
                prefix="ragnarok-modelfile-", suffix=".txt",
                delete=False,
            ) as fh:
                fh.write(modelfile)
                fh.flush()
                tmp_path = fh.name
            proc = subprocess.run(
                ["ollama", "create", name, "-f", tmp_path],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            out = (proc.stdout or b"").decode("utf-8", errors="replace")
            err = (proc.stderr or b"").decode("utf-8", errors="replace")
            if proc.returncode != 0:
                logger.warning(
                    f"ModelManager.create_from_modelfile {name} "
                    f"CLI exit={proc.returncode} stderr={err!r} stdout={out!r}"
                )
                raise CreateError(
                    f"ollama create {name} failed (exit {proc.returncode}): {err.strip() or out.strip()}",
                    exit_code=proc.returncode, stdout=out, stderr=err,
                )
            logger.info(f"ModelManager.create_from_modelfile {name} ok")
            return None
        except FileNotFoundError:
            raise CreateError(
                "ollama CLI not found on PATH; the `ollama` binary must be "
                "installed and visible to the gateway process",
                exit_code=None, stdout="", stderr="",
            )
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError as e:
                    logger.warning(
                        f"create_from_modelfile: failed to remove temp file {tmp_path}: {e}"
                    )

    async def ensure_available(self, name: str) -> bool:
        """Check Ollama first; pull only if missing. Returns True if it was
        already present (no pull performed)."""
        if await self.exists(name):
            logger.info(f"ModelManager.ensure_available {name} already present -- no pull")
            return True
        await self.pull(name)
        logger.info(f"ModelManager.ensure_available {name} pulled")
        return False
