"""Gateway state management — http client, semaphore, warmup, active-stream registry."""

import asyncio
import time
from typing import Any, Dict, List, Optional

import httpx
import ollama

from fastapi import Request

from src.config import (
    MODEL_NAME,
    MAX_CONCURRENT,
    KEEP_ALIVE,
    OLLAMA_BASE_URL,
    HTTP_CONNECT_TIMEOUT,
    HTTP_READ_TIMEOUT,
    HTTP_WRITE_TIMEOUT,
    HTTP_POOL_TIMEOUT,
    MAX_CONNECTIONS,
    MAX_KEEPALIVE_CONNECTIONS,
    KEEPALIVE_EXPIRY,
    model_opts,
)
from src.logging import logger


class ActiveStream:
    """Handle for one in-flight streaming generation."""
    __slots__ = ("request_id", "model", "task", "started_at")

    def __init__(self, request_id: str, model: str, task: "asyncio.Task", started_at: float):
        self.request_id = request_id
        self.model = model
        self.task = task
        self.started_at = started_at


class GatewayState:
    __slots__ = (
        "http_client", "semaphore", "warmup_task", "is_warm", "warmup_ok",
        "active_streams",
    )

    def __init__(self):
        self.http_client = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self.warmup_task = None
        self.is_warm = False
        self.warmup_ok = False  # True if warmup succeeded, False if it failed
        # request_id -> ActiveStream ; owned by the streaming path
        self.active_streams: Dict[str, ActiveStream] = {}

    # ── active-stream registry ──
    def register_stream(self, request_id: str, model: str, task: "asyncio.Task") -> None:
        self.active_streams[request_id] = ActiveStream(request_id, model, task, time.monotonic())
        logger.info(f"[{request_id}] REGISTER active_stream model={model} "
                    f"inflight={len(self.active_streams)}")

    def unregister_stream(self, request_id: str) -> None:
        entry = self.active_streams.pop(request_id, None)
        if entry is not None:
            logger.info(f"[{request_id}] UNREGISTER active_stream "
                        f"inflight={len(self.active_streams)}")

    def get_stream(self, request_id: str) -> Optional[ActiveStream]:
        return self.active_streams.get(request_id)

    def streams_for_model(self, model: str) -> List[ActiveStream]:
        return [s for s in self.active_streams.values() if s.model == model]

    def _cancel_stream_task(self, entry: ActiveStream) -> None:
        """Request cancellation of the driving task. Caller awaits separately.

        We do NOT await here — this may be called from the streaming task's
        own finally (self-await would deadlock). The caller is responsible
        for awaiting the task after cancelling, when safe to do so.
        """
        task = entry.task
        if task is not None and not task.done():
            task.cancel()

    async def stop_stream(self, request_id: str) -> bool:
        """Cancel one in-flight generation (called from the stop endpoint).

        The generator's own finally will cancel the Ollama pusher and release
        the semaphore; here we just drive the task's unwinding to completion.
        Returns True if the stream was found and cancelled.
        """
        entry = self.active_streams.get(request_id)
        if entry is None:
            return False
        self._cancel_stream_task(entry)
        task = entry.task
        if task is not None and not task.done():
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self.active_streams.pop(request_id, None)
        logger.info(f"[{request_id}] STOPPED active_stream")
        return True

    async def stop_streams_for_model(self, model: str) -> int:
        """Cancel all in-flight generations for *model*."""
        entries = self.streams_for_model(model)
        for e in entries:
            self._cancel_stream_task(e)
        for e in entries:
            task = e.task
            if task is not None and not task.done():
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
            self.active_streams.pop(e.request_id, None)
        if entries:
            logger.info(f"STOPPED {len(entries)} stream(s) for model={model}")
        return len(entries)


async def ask_ollama_unload(http_client, model: str) -> None:
    """Ask Ollama to unload *model* (keep_alive=0). Best-effort.

    Tries /api/generate first (lighter than /api/chat), then falls back to chat.
    Both are non-streaming, single-token, and explicitly set keep_alive="0".
    """
    if http_client is None:
        return
    try:
        await http_client.generate(
            model=model,
            prompt=" ",
            options={"num_predict": 1},
            keep_alive="0",
        )
    except Exception as e1:
        try:
            await http_client.chat(
                model=model,
                messages=[{"role": "user", "content": " "}],
                keep_alive="0",
            )
        except Exception as e2:
            logger.warning(f"unload_model({model}) failed: gen={e1} chat={e2}")


def _get_state(request: Request) -> GatewayState:
    return request.app.state.gw


async def _warmup(state: GatewayState):
    try:
        await state.http_client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            keep_alive=KEEP_ALIVE,
            options=model_opts(MODEL_NAME, warmup=True),
        )
        state.warmup_ok = True
        state.is_warm = True
        logger.info(f"Model '{MODEL_NAME}' is warm and ready!")
    except Exception as e:
        state.warmup_ok = False
        state.is_warm = False
        logger.warning(f"Warm-up for '{MODEL_NAME}' failed: {e}")
        logger.warning("Server will return 503 until model loads successfully.")
