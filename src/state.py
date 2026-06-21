"""Gateway state management — http client, semaphore, warmup."""

import asyncio

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
    _OLLAMA_OPTS_WARMUP,
)
from src.logging import logger


class GatewayState:
    __slots__ = ("http_client", "semaphore", "warmup_task", "is_warm")

    def __init__(self):
        self.http_client = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self.warmup_task = None
        self.is_warm = False


def _get_state(request: Request) -> GatewayState:
    return request.app.state.gw


async def _warmup(state: GatewayState):
    try:
        await state.http_client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            keep_alive=KEEP_ALIVE,
            options=_OLLAMA_OPTS_WARMUP,
        )
        state.is_warm = True
        logger.info(f"Model '{MODEL_NAME}' is warm and ready!")
    except Exception as e:
        logger.warning(f"Warm-up skipped: {e}")
        state.is_warm = True


def _create_http_client() -> ollama.AsyncClient:
    return ollama.AsyncClient(
        host=OLLAMA_BASE_URL,
        timeout=httpx.Timeout(
            connect=HTTP_CONNECT_TIMEOUT,
            read=HTTP_READ_TIMEOUT,
            write=HTTP_WRITE_TIMEOUT,
            pool=HTTP_POOL_TIMEOUT,
        ),
        limits=httpx.Limits(
            max_connections=MAX_CONNECTIONS,
            max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
            keepalive_expiry=KEEPALIVE_EXPIRY,
        ),
        follow_redirects=True,
    )
