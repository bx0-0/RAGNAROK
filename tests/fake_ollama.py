"""Fake Ollama server — SSE streaming /api/chat that tracks open connections.

Purpose: prove that when RAGNAROK's client (pusher) closes the per-request
stream on disconnect, the FAKE OLLAMA observes the connection close (the
abort path). We track:
  - open_streams: request_ids currently being served
  - closed_streams: request_ids whose connection was closed by the client
  - aborted: request_ids where we detected the client disconnect (GeneratorExit/CancelledError)

Endpoints:
  POST /api/chat   -> SSE stream of N chunks, ~delay between each
  GET  /api/ps     -> {"models": [{"name": MODEL, "size_vram": 123}]}
  POST /api/generate -> minimal non-stream (for keep_alive=0 unload probe)
"""

import asyncio
import time
import json

from starlette.applications import Starlette
from starlette.responses import StreamingResponse, JSONResponse
from starlette.routing import Route
import asyncio as _asyncio

class _WatchedStreamingResponse(StreamingResponse):
    async def __call__(self, scope, receive, send):
        async def _stream():
            await self.stream_response(send)
        async def _watch():
            while True:
                msg = await receive()
                if msg["type"] == "http.disconnect":
                    return
        s = _asyncio.create_task(_stream())
        w = _asyncio.create_task(_watch())
        done, pending = await _asyncio.wait({s, w}, return_when=_asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
            try:
                await t
            except BaseException:
                pass
        for t in done:
            if not t.cancelled() and t.exception() is not None:
                raise t.exception()
        if self.background is not None:
            await self.background()

from starlette.requests import Request

MODEL_NAME = "qwen3.5:9b"
import os as _os
TOTAL_CHUNKS = int(_os.environ.get("FAKE_TOTAL_CHUNKS", "40"))
CHUNK_DELAY = float(_os.environ.get("FAKE_CHUNK_DELAY", "0.10"))

# ── connection tracking ──
open_streams: dict[str, dict] = {}      # req -> {started, sent}
closed_clean: list[str] = []            # finished all chunks
aborted: list[str] = []                 # client hung up mid-stream


def _req_id(request_id: str) -> str:
    return request_id or "unknown"


async def _chat_sse(request: Request):
    body = await request.json()
    stream = bool(body.get("stream", True))
    # client may or may not send a request id; we derive one from time
    # but we also allow an "x-test-id" header for correlation
    rid = request.headers.get("x-test-id") or f"t{int(time.time()*1000)}"

    # Non-stream: warmup / unload probe -> single ChatResponse
    if not stream:
        return JSONResponse({
            "model": body.get("model", MODEL_NAME),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
            "eval_count": 1,
            "prompt_eval_count": 1,
        })

    sent = 0
    open_streams[rid] = {"started": time.monotonic(), "sent": 0}

    async def gen():
        nonlocal sent
        try:
            for i in range(TOTAL_CHUNKS):
                chunk = {
                    "model": MODEL_NAME,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "message": {"role": "assistant", "content": f"tok{i} "},
                    "done": False,
                    "eval_count": i + 1,
                }
                # Ollama native stream = raw NDJSON (one JSON obj per line)
                yield json.dumps(chunk).encode() + b"\n"
                sent += 1
                open_streams[rid]["sent"] = sent
                await asyncio.sleep(CHUNK_DELAY)

            # final done frame
            final = {
                "model": MODEL_NAME,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
                "eval_count": TOTAL_CHUNKS,
            }
            yield json.dumps(final).encode() + b"\n"
            closed_clean.append(rid)
            open_streams.pop(rid, None)
        except (asyncio.CancelledError, GeneratorExit):
            # Client disconnected / we were cancelled -> Ollama would abort
            aborted.append(rid)
            open_streams.pop(rid, None)
            raise

    return _WatchedStreamingResponse(gen(), media_type="application/x-ndjson")


async def _generate(request: Request):
    # keep_alive=0 unload probe; just return a tiny completion
    body = await request.json()
    return JSONResponse({
        "model": body.get("model", MODEL_NAME),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "response": " ",
        "done": True,
        "eval_count": 1,
    })


async def _ps(request: Request):
    return JSONResponse({
        "models": [
            {
                "name": MODEL_NAME,
                "size": 5000000000,
                "size_vram": 5000000000,
                "expires_at": "2099-01-01T00:00:00Z",
            }
        ]
    })



async def _state(request: Request):
    return JSONResponse({
        "open_streams": {k: v for k, v in open_streams.items()},
        "closed_clean": closed_clean,
        "aborted": aborted,
    })

def build_app():

    routes = [
        Route("/api/chat", _chat_sse, methods=["POST"]),
        Route("/api/generate", _generate, methods=["POST"]),
        Route("/api/ps", _ps, methods=["GET"]),
        Route("/api/tags", _ps, methods=["GET"]),
        Route("/_state", _state, methods=["GET"]),
    ]
    return Starlette(routes=routes)


if __name__ == "__main__":
    import os
    import uvicorn
    app = build_app()
    port = int(os.environ.get("FAKE_PORT", "11434"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
