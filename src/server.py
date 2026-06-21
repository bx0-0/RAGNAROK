"""Slim app orchestrator — config, state, routes, lifespan, entry point."""

import asyncio
import contextlib

import httpx
import ollama
import uvloop
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import (
    MODEL_NAME,
    MAX_CONCURRENT,
    NUM_CTX,
    OLLAMA_BASE_URL,
    PORT,
    _MODEL_LIST,
    HTTP_CONNECT_TIMEOUT,
    HTTP_READ_TIMEOUT,
    HTTP_WRITE_TIMEOUT,
    HTTP_POOL_TIMEOUT,
    MAX_CONNECTIONS,
    MAX_KEEPALIVE_CONNECTIONS,
    KEEPALIVE_EXPIRY,
)
from src.state import GatewayState, _create_http_client, _warmup
from src.routes import register_routers
from src.logging import setup_logging, logger, _open_log_fh, _log_fh as _gw_log_fh


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    state = GatewayState()
    state.http_client = ollama.AsyncClient(
        host=OLLAMA_BASE_URL,
        timeout=httpx.Timeout(connect=HTTP_CONNECT_TIMEOUT, read=HTTP_READ_TIMEOUT, write=HTTP_WRITE_TIMEOUT, pool=HTTP_POOL_TIMEOUT),
        limits=httpx.Limits(
            max_connections=MAX_CONNECTIONS,
            max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
            keepalive_expiry=KEEPALIVE_EXPIRY,
        ),
        follow_redirects=True,
    )
    state.warmup_task = asyncio.create_task(_warmup(state))

    app.state.gw = state
    _open_log_fh()

    banner = (
        f"\n{'='*60}\n"
        f"  \033[1m\033[0;31m🐉 RAGNAROK\033[0m\n"
        f"  \033[1m\033[0;36mGPU Model Gateway\033[0m\n"
        f"{'='*60}\n"
        f"  \033[0;90mModels:\033[0m    {_MODEL_LIST}\n"
        f"  \033[0;90mDefault:\033[0m   {MODEL_NAME}\n"
        f"  \033[0;90mPort:\033[0m      {PORT}\n"
        f"  \033[0;90mConcurrent:\033[0m {MAX_CONCURRENT}\n"
        f"  \033[0;90mContext:\033[0m   {NUM_CTX}\n"
        f"{'='*60}\n"
    )
    print(banner, flush=True)
    logger.info("FastAPI starting up...")
    logger.info(f"Warming up model '{MODEL_NAME}' in background.")

    yield

    for task in (state.warmup_task,):
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    await state.http_client.aclose()
    if _gw_log_fh is not None:
        _gw_log_fh.flush()
        _gw_log_fh.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_routers(app)


def run_server():
    uvloop.install()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="warning",
        http="httptools",
        loop="uvloop",
        timeout_keep_alive=300,
        access_log=False,
    )


