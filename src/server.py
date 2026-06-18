import os
import time
import asyncio
import collections
import contextlib
from functools import lru_cache

import orjson
import httpx
import uvloop
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from src.logging import setup_logging, logger
from src.utils import (
    extract_text_content,
    convert_messages_to_ollama,
    format_tool_calls_openai,
    _fast_id,
)

# ─── Config ───
_RAW_MODEL_LIST = os.environ.get("MODEL_NAME", "qwen3:8b").split()
# Replace full HF names with short aliases (created by ollama cp in install_model.sh)
_SHORT_ALIASES = {name: name.split("/", 3)[-1] for name in _RAW_MODEL_LIST if name.startswith("hf.co/")}
_MODEL_LIST = [_SHORT_ALIASES.get(m, m) for m in _RAW_MODEL_LIST]
MODEL_NAME = _MODEL_LIST[0]  # Default = first model
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "1"))
NUM_CTX = int(os.environ.get("NUM_CTX", "68768"))
NUM_PREDICT = int(os.environ.get("NUM_PREDICT", "16384"))
NUM_BATCH = int(os.environ.get("NUM_BATCH", "2444"))
FLASH_ATTN = os.environ.get("FLASH_ATTN", "True").lower() in ("true", "1", "yes")
NUM_GPU = int(os.environ.get("NUM_GPU", "-1"))
KEEP_ALIVE = os.environ.get("KEEP_ALIVE", "60m")
PORT = int(os.environ.get("PORT", "8000"))
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
VERBOSE_LOG = os.environ.get("VERBOSE_LOG", "True").lower() in ("true", "1", "yes")
REQUEST_LOG_FILE = os.environ.get("REQUEST_LOG_FILE", "/tmp/gateway-requests.log")
MAX_STREAM_SECONDS = int(os.environ.get("MAX_STREAM_SECONDS", "1800"))  # 30 min
HTTP_CONNECT_TIMEOUT = float(os.environ.get("HTTP_CONNECT_TIMEOUT", "60.0"))
HTTP_READ_TIMEOUT = float(os.environ.get("HTTP_READ_TIMEOUT", "900.0"))
HTTP_WRITE_TIMEOUT = float(os.environ.get("HTTP_WRITE_TIMEOUT", "60.0"))
HTTP_POOL_TIMEOUT = float(os.environ.get("HTTP_POOL_TIMEOUT", "900.0"))
MAX_CONNECTIONS = int(os.environ.get("MAX_CONNECTIONS", "2000"))
MAX_KEEPALIVE_CONNECTIONS = int(os.environ.get("MAX_KEEPALIVE_CONNECTIONS", "500"))
KEEPALIVE_EXPIRY = int(os.environ.get("KEEPALIVE_EXPIRY", "300"))
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", "256"))
MAX_KEEPALIVE_PINGS = int(os.environ.get("MAX_KEEPALIVE_PINGS", "120"))

# Precompute ollama options (never changes at runtime)
_OLLAMA_OPTS_BASE = {
    "num_ctx": NUM_CTX,
    "num_batch": NUM_BATCH,
    "flash_attn": FLASH_ATTN,
    "num_gpu": NUM_GPU,
}

_OLLAMA_OPTS = {
    "num_ctx": NUM_CTX,
    "num_batch": NUM_BATCH,
    "flash_attn": FLASH_ATTN,
    "num_gpu": NUM_GPU,
    "num_predict": NUM_PREDICT,
}

_OLLAMA_OPTS_WARMUP = {
    "num_ctx": NUM_CTX,
    "num_batch": NUM_BATCH,
    "flash_attn": FLASH_ATTN,
    "num_gpu": NUM_GPU,
    "num_predict": 1,
}

# ─── SSE template factory (cached per-model) ───
_SSE_MARKER_ID = "\xffID"
_SSE_MARKER_TS = "\xffTS"

@lru_cache(maxsize=16)
def _sse_template_for_model(model: str):
    """Build the SSE JSON envelope once per model with unique string markers.
    Returns (prefix_bytes, suffix_bytes) ready for:
        yield prefix + orjson.dumps(delta) + suffix
    """
    tpl = orjson.dumps({
        "id": _SSE_MARKER_ID,
        "object": "chat.completion.chunk",
        "created": 0,  # placeholder — replaced below
        "model": model,
        "choices": [{"delta": "__DELTA__", "index": 0, "finish_reason": None}],
    })
    _marker = b'"__DELTA__"'
    _id_marker = f'"{_SSE_MARKER_ID}"'.encode()
    
    # Split around the delta marker
    _dpos = tpl.index(_marker)
    prefix = tpl[:_dpos]
    suffix = tpl[_dpos + len(_marker):]
    return (b"data: " + prefix, suffix + b"\n\n")


def _make_sse_frames(model: str, request_id_str: str, created: int):
    """Inject real id + timestamp into cached SSE envelope."""
    prefix, suffix = _sse_template_for_model(model)
    # Replace id marker
    _id_placeholder = f'"{_SSE_MARKER_ID}"'.encode()
    _real_id = f'"{request_id_str}"'.encode()
    prefix = prefix.replace(_id_placeholder, _real_id, 1)
    # Replace timestamp — use a safe byte pattern that won't collide
    # Since created is always the integer after "created":, we patch "created":0 → real value
    # But 0 could appear elsewhere. Safer: rebuild just the two fields.
    # Actually the safest is to not cache at all and just serialize — but that's what we want to avoid.
    # Compromise: use a large sentinel for created so it's unique in the byte output.
    prefix = prefix.replace(b'"created":0', f'"created":{created}'.encode(), 1)
    return (prefix, suffix)


# ─── Chunked body reader (avoids buffering huge prompts in RAM) ───
async def _read_body(request: Request, max_size_mb: int = 50) -> bytes:
    """Read request body without loading >max_size_mb all at once.
    Returns raw bytes. Raises ValueError if Content-Length exceeds limit."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_size_mb * 1024 * 1024:
        raise ValueError(f"Body too large: {int(content_length)} bytes")
    # Read in 64KB chunks — FastAPI/Starlette already streams internally
    chunks = []
    async for chunk in request.stream():
        chunks.append(chunk)
        total = sum(len(c) for c in chunks)
        if total > max_size_mb * 1024 * 1024:
            raise ValueError(f"Body exceeds {max_size_mb}MB limit")
    return b"".join(chunks)


# Precompute static SSE frame bytes
_SSE_DONE = b"data: [DONE]\n\n"
_SSE_KEEPALIVE = b': ping\n\n'
_RATE_LIMIT_RESPONSE = Response(status_code=429, content=orjson.dumps({
    "error": {
        "message": "Server is busy. Try again shortly.",
        "type": "rate_limit_error",
    },
}), media_type="application/json")
_BAD_JSON_RESPONSE = Response(status_code=400, content=b'{"error":"Invalid JSON"}', media_type="application/json")


# ─── Async-safe log queue (non-blocking) ───
_log_fh = None
_request_log = collections.deque(maxlen=500)


def _open_log_fh():
    global _log_fh
    if _log_fh is None:
        _log_fh = open(REQUEST_LOG_FILE, "a", buffering=8192)


def _status_color(code):
    if code < 300:
        return "\033[0;32m"
    if code == 429:
        return "\033[0;33m"
    if code < 500:
        return "\033[0;31m"
    return "\033[0;35m"


def _status_label(code):
    if code < 300:
        return "OK"
    if code == 429:
        return "Busy"
    if code == 400:
        return "Bad"
    if code < 500:
        return "Err"
    return "Fatal"


def _build_log_line(tag, req_id, status_or_method, path=None, extra=None, duration=None, t_in=None, t_out=None):
    ts = time.strftime("%H:%M:%S")
    line = f"\033[0;36m[{ts}]\033[0m \033[0;90m{tag}\033[0m \033[1m{req_id}\033[0m "
    if duration is not None:
        color = _status_color(status_or_method)
        label = _status_label(status_or_method)
        line += (
            f"{color}{status_or_method} {label}\033[0m "
            f"{duration}s "
            f"\033[0;90mt:{t_in}\u2192{t_out}\033[0m"
        )
    else:
        line += f"{status_or_method} {path}"
    if extra:
        line += f"  \033[0;33m{extra}\033[0m"
    return line


async def log_request_start(req_id, method, path, extra=""):
    line = _build_log_line("◀", req_id, method, path, extra=extra)
    if VERBOSE_LOG:
        print(line, flush=True)
    _enqueue_log(line)


def _enqueue_log(line: str):
    _open_log_fh()
    try:
        _log_fh.write(line + "\n")
    except OSError:
        pass


async def log_request(req_id, method, path, status, duration, t_in, t_out, extra=""):
    _request_log.append({
        "id": req_id, "method": method, "path": path,
        "status": status, "duration": duration,
        "t_in": t_in, "t_out": t_out, "extra": extra,
    })
    line = _build_log_line(
        "▶", req_id, status, path,
        duration=duration, t_in=t_in, t_out=t_out, extra=extra,
    )
    if VERBOSE_LOG:
        print(line, flush=True)
    _enqueue_log(line)


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
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "keep_alive": KEEP_ALIVE,
            "options": _OLLAMA_OPTS_WARMUP,
        }
        async with state.http_client.stream(
            "POST", OLLAMA_CHAT_URL,
            json=payload,
            timeout=300.0,
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                if orjson.loads(line).get("done"):
                    break
        state.is_warm = True
        logger.info(f"Model '{MODEL_NAME}' is warm and ready!")
    except Exception as e:
        logger.warning(f"Warm-up skipped: {e}")
        state.is_warm = True


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    state = GatewayState()
    state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=HTTP_CONNECT_TIMEOUT, read=HTTP_READ_TIMEOUT, write=HTTP_WRITE_TIMEOUT, pool=HTTP_POOL_TIMEOUT),
        limits=httpx.Limits(
            max_connections=MAX_CONNECTIONS,
            max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
            keepalive_expiry=KEEPALIVE_EXPIRY,
        ),
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
    logger.info(f"Warming up model '{MODEL_NAME}' in background. ..")

    yield

    for task in (state.warmup_task,):
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    await state.http_client.aclose()
    if _log_fh is not None:
        _log_fh.flush()
        _log_fh.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": model,
            "object": "model",
            "owned_by": "local",
        } for model in _MODEL_LIST],
    }


@app.get("/health")
async def health_check(request: Request):
    state = _get_state(request)
    return {
        "status": "ready" if state.is_warm else "warming",
        "models": _MODEL_LIST,
        "default": MODEL_NAME,
    }


@app.post("/v1/embeddings")
async def openai_embeddings(request: Request):
    state = _get_state(request)
    request_id = _fast_id()
    start_time = time.monotonic()
    await log_request_start(request_id, "POST", "/v1/embeddings")

    try:
        body = orjson.loads(await _read_body(request))
    except Exception:
        await log_request(request_id, "POST", "/v1/embeddings", 400, 0, 0, 0, "BAD_JSON")
        return Response(status_code=400, content=b'{"error":"Invalid JSON"}', media_type="application/json")

    model = body.get("model", MODEL_NAME)
    input_data = body.get("input")
    if not input_data:
        await log_request(request_id, "POST", "/v1/embeddings", 400, 0, 0, 0, "MISSING_INPUT")
        return Response(
            status_code=400,
            content=orjson.dumps({"error": {"message": "'input' is required", "type": "invalid_request_error"}}),
            media_type="application/json",
        )

    if isinstance(input_data, str):
        input_data = [input_data]

    payload = {"model": model, "input": input_data}

    try:
        resp = await state.http_client.post(f"{OLLAMA_BASE_URL}/api/embed", json=payload)
        elapsed = round(time.monotonic() - start_time, 2)
        await log_request(request_id, "POST", "/v1/embeddings", resp.status_code, elapsed, 0, 0, "EMBED")
        return Response(status_code=resp.status_code, content=resp.content, media_type="application/json")
    except Exception as e:
        elapsed = round(time.monotonic() - start_time, 2)
        await log_request(request_id, "POST", "/v1/embeddings", 502, elapsed, 0, 0, f"ERR:{str(e)[:40]}")
        return Response(
            status_code=502,
            content=orjson.dumps({"error": {"message": str(e), "type": "upstream_error"}}),
            media_type="application/json",
        )


@app.post("/v1/chat/completions")
async def openai_completions(request: Request):
    state = _get_state(request)

    if not state.is_warm:
        return Response(
            status_code=503,
            content=orjson.dumps({
                "error": {
                    "message": "Model is still loading. Please try again shortly.",
                    "type": "server_error",
                    "param": None,
                    "code": "model_loading",
                },
            }),
            media_type="application/json",
        )

    request_id = _fast_id()
    start_time = time.monotonic()

    await log_request_start(request_id, "POST", "/v1/chat/completions")

    # Atomic semaphore — reject immediately if busy
    try:
        await asyncio.wait_for(state.semaphore.acquire(), timeout=0.05)
    except asyncio.TimeoutError:
        logger.warning(f"[{request_id}] Rejected: busy")
        await log_request(request_id, "POST", "/v1/chat/completions", 429, 0, 0, 0, "RATE_LIMITED")
        return _RATE_LIMIT_RESPONSE

    # Parse body with orjson (faster than FastAPI's stdlib json)
    try:
        body = orjson.loads(await _read_body(request))
    except Exception:
        state.semaphore.release()
        await log_request(request_id, "POST", "/v1/chat/completions", 400, 0, 0, 0, "BAD_JSON")
        return _BAD_JSON_RESPONSE

    # ═══ Validate request body ═══
    messages = body.get("messages")
    if not messages or not isinstance(messages, list) or len(messages) == 0:
        state.semaphore.release()
        await log_request(request_id, "POST", "/v1/chat/completions", 400, 0, 0, 0, "MISSING_MESSAGES")
        return Response(
            status_code=400,
            content=orjson.dumps({"error": {"message": "'messages' must be a non-empty list", "type": "invalid_request_error"}}),
            media_type="application/json",
        )

    # ═══ Log request info for debugging ═══
    tools = body.get("tools")
    msg_count = len(body.get("messages", []))
    total_chars = sum(len(str(m.get("content", ""))) for m in body.get("messages", []))
    client_name = request.headers.get("user-agent", "unknown")[:40]
    tool_names = [t.get("function", {}).get("name") for t in tools if isinstance(t, dict)] if tools else []
    logger.info(
        f"[{request_id}] Client={client_name} | Msgs={msg_count} | "
        f"Chars={total_chars} | Tools={tool_names}"
    )
    # ═══ End log ═══

    is_streaming = body.get("stream", False)

    # ── Resolve model: use client's model param if valid, else default ──
    requested_model = body.get("model", "")
    if requested_model and requested_model in _MODEL_LIST:
        active_model = requested_model
    else:
        active_model = MODEL_NAME

    ollama_messages = convert_messages_to_ollama(body.get("messages", []), has_tools=bool(tools))

    # Build ollama payload with orjson
    # Dynamic options: enable thinking if requested
    opts = dict(_OLLAMA_OPTS)
    if body.get("thinking", False):
        opts["thinking"] = {"enabled": True}
    else:
        opts["thinking"] = {"enabled": False}

    ollama_payload_dict = {
        "model": active_model,
        "messages": ollama_messages,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": opts,
    }
    if tools:
        ollama_payload_dict["tools"] = tools
    tool_choice = body.get("tool_choice")
    if tool_choice:
        ollama_payload_dict["tool_choice"] = tool_choice

    created = int(time.time())
    if not is_streaming:
        try:
            result = await _handle_non_stream(state, request_id, ollama_payload_dict, start_time, created, active_model, request)
            return result
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[{request_id}] Non-stream handler crashed: {e}")
            elapsed = round(time.monotonic() - start_time, 2)
            await log_request(request_id, "POST", "/v1/chat/completions", 500, elapsed, 0, 0, f"CRASH:{str(e)[:40]}")
            return Response(
                status_code=500,
                content=orjson.dumps({"error": {"message": "Internal server error", "type": "server_error"}}),
                media_type="application/json",
            )
        finally:
            state.semaphore.release()
    else:
        return _handle_stream(state, request_id, ollama_payload_dict, start_time, active_model)


async def _handle_non_stream(state, request_id, ollama_payload, start_time, created, active_model, request):
    content_parts = []
    thinking_parts = []
    all_tool_calls = []
    prompt_tokens = completion_tokens = 0

    try:
        async with state.http_client.stream(
            "POST", OLLAMA_CHAT_URL,
            json=ollama_payload,
        ) as response:
            if response.status_code != 200:
                err = await response.aread()
                elapsed = round(time.monotonic() - start_time, 2)
                logger.error(f"[{request_id}] Ollama Upstream Error: {err.decode()[:300]}")
                await log_request(request_id, "POST", "/v1/chat/completions", response.status_code, elapsed, 0, 0, "UPSTREAM_ERR")
                return Response(status_code=response.status_code, content=err, media_type="application/json")

            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                # Abort immediately if client disconnected
                if await request.is_disconnected():
                    await response.aclose()
                    raise asyncio.CancelledError

                data = orjson.loads(line)
                msg = data.get("message", {})

                content = msg.get("content")
                if content:
                    content_parts.append(content)
                thinking = msg.get("thinking")
                if thinking:
                    thinking_parts.append(thinking)
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    all_tool_calls.extend(format_tool_calls_openai(tool_calls))

                if data.get("done"):
                    prompt_tokens = data.get("prompt_eval_count", 0)
                    completion_tokens = data.get("eval_count", 0)
                    break
    except asyncio.CancelledError:
        elapsed = round(time.monotonic() - start_time, 2)
        await log_request(request_id, "POST", "/v1/chat/completions", 499, elapsed, 0, 0, "CLIENT_DISCONNECTED")
        raise
    except Exception as e:
        elapsed = round(time.monotonic() - start_time, 2)
        await log_request(request_id, "POST", "/v1/chat/completions", 500, elapsed, 0, 0, f"ERR:{str(e)[:40]}")
        return Response(status_code=500, content=orjson.dumps({"error": str(e)}), media_type="application/json")

    elapsed = round(time.monotonic() - start_time, 2)
    await log_request(request_id, "POST", "/v1/chat/completions", 200, elapsed, prompt_tokens, completion_tokens, "NON-STREAM")

    resp_message = {
        "role": "assistant",
        "content": "".join(content_parts) or None,
    }
    if thinking_parts:
        resp_message["reasoning_content"] = "".join(thinking_parts)
    if all_tool_calls:
        resp_message["tool_calls"] = all_tool_calls

    return Response(content=orjson.dumps({
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": created,
        "model": active_model,
        "choices": [{
            "index": 0,
            "message": resp_message,
            "finish_reason": "tool_calls" if all_tool_calls else "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }), media_type="application/json")



def _handle_stream(state, request_id, ollama_payload, start_time, active_model):
    request_id_str = f"chatcmpl-{request_id}"
    created = int(time.time())

    # Use cached SSE template — avoids re-serializing the static envelope per request
    _sfx, _efx = _make_sse_frames(active_model, request_id_str, created)

    async def stream_generator():
        first_chunk = True
        has_tool_calls = False
        tool_call_index = 0
        prompt_tokens = completion_tokens = 0
        released = False

        try:
            async with state.http_client.stream(
                "POST", OLLAMA_CHAT_URL,
                json=ollama_payload,
            ) as response:
                if response.status_code != 200:
                    err_body = await response.aread()
                    elapsed = round(time.monotonic() - start_time, 2)
                    # Log the actual upstream error from Ollama
                    logger.error(f"[{request_id}] Ollama Upstream Error: {err_body.decode()[:300]}")
                    await log_request(request_id, "POST", "/v1/chat/completions", response.status_code, elapsed, 0, 0, "UPSTREAM_ERR")
                    # Send the real error to the client so they understand the issue
                    yield b"data: " + err_body + b"\n\n"
                    yield _SSE_DONE
                    return

                line_q = asyncio.Queue(maxsize=256)
                reader_error = [None]

                async def _reader():
                    try:
                        async for raw in response.aiter_lines():
                            await line_q.put(raw)
                    except Exception as e:
                        reader_error[0] = e
                    finally:
                        await line_q.put(None)

                reader = asyncio.create_task(_reader())
                keepalive_count = 0
                graceful = False
                try:
                    while True:
                        # ═══ Hard Timeout ═══
                        stream_elapsed = time.monotonic() - start_time
                        if stream_elapsed > MAX_STREAM_SECONDS:
                            logger.warning(f"[{request_id}] Hard timeout after {int(stream_elapsed)}s")
                            yield b"data: " + orjson.dumps({
                                "error": {
                                    "message": f"Generation exceeded {MAX_STREAM_SECONDS}s limit",
                                    "type": "timeout",
                                }
                            }) + b"\n\n"
                            yield _SSE_DONE
                            return
                        # ═══ End Hard Timeout ═══

                        try:
                            raw = await asyncio.wait_for(line_q.get(), timeout=10.0)
                        except asyncio.TimeoutError:
                            keepalive_count += 1
                            if keepalive_count > 120:
                                yield b"data: " + orjson.dumps({"error": {"message": "Upstream timeout", "type": "timeout"}}) + b"\n\n"
                                yield _SSE_DONE
                                return
                            yield _SSE_KEEPALIVE
                            continue

                        if raw is None:
                            break
                        keepalive_count = 0

                        if not raw.strip():
                            continue

                        try:
                            data = orjson.loads(raw)
                        except Exception as e:
                            logger.error(f"[{request_id}] Parse fail: {e} | line={raw[:100]}")
                            continue

                        if data.get("error"):
                            logger.error(f"[{request_id}] Ollama error: {data.get('error')}")
                            yield b"data: " + orjson.dumps({"error": {"message": data["error"]}}) + b"\n\n"
                            yield _SSE_DONE
                            return

                        message = data.get("message", {})

                        content = message.get("content")
                        if isinstance(content, list):
                            content = " ".join(
                                item.get("text", "")
                                for item in content
                                if isinstance(item, dict) and item.get("type") == "text"
                            )
                        elif not isinstance(content, str):
                            content = content if content is not None else ""

                        thinking = message.get("thinking", "")
                        tool_calls = message.get("tool_calls")

                        if first_chunk or thinking or content or tool_calls:
                            delta = {}
                            if first_chunk:
                                delta["role"] = "assistant"
                                first_chunk = False
                            if thinking:
                                delta["reasoning_content"] = thinking
                            if content:
                                delta["content"] = content

                            if tool_calls:
                                try:
                                    has_tool_calls = True
                                    formatted = []
                                    for tc in tool_calls:
                                        tc_name = tc.get("function", {}).get("name") or "?"
                                        # Log tool name clearly
                                        logger.info(f"[{request_id}] 🔧 Tool Call: {tc_name}")
                                        tc_args = tc.get("function", {}).get("arguments", "")
                                        if isinstance(tc_args, str):
                                            tc_args_json = tc_args
                                        else:
                                            tc_args_json = orjson.dumps(tc_args).decode() if tc_args else ""
                                        formatted.append({
                                            "index": tool_call_index,
                                            "id": tc.get("id") or f"call_{_fast_id()}",
                                            "type": "function",
                                            "function": {
                                                "name": tc_name,
                                                "arguments": tc_args_json,
                                            },
                                        })
                                        tool_call_index += 1
                                    delta["tool_calls"] = formatted
                                except Exception as e:
                                    logger.error(f"[{request_id}] Tool format error: {e}")
                                    delta["tool_calls"] = []
                                if "content" in delta and not delta["content"]:
                                    del delta["content"]

                            try:
                                yield _sfx + orjson.dumps(delta) + _efx
                            except Exception as e:
                                logger.error(f"[{request_id}] Serialize delta failed: {e}")
                                continue

                        if data.get("done"):
                            prompt_tokens = data.get("prompt_eval_count", 0)
                            completion_tokens = data.get("eval_count", 0)
                            graceful = True

                            yield b"data: " + orjson.dumps({
                                "id": request_id_str,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": active_model,
                                "choices": [{
                                    "delta": {},
                                    "index": 0,
                                    "finish_reason": "tool_calls" if has_tool_calls else "stop",
                                }],
                                "usage": {
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": prompt_tokens + completion_tokens,
                                },
                            }) + b"\n\n"
                            break
                finally:
                    reader.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await reader

                if reader_error[0]:
                    logger.error(f"[{request_id}] Reader error: {reader_error[0]}")
                    yield b"data: " + orjson.dumps({"error": {"message": str(reader_error[0]), "type": "upstream_error"}}) + b"\n\n"

                if not graceful and not reader_error[0]:
                    yield b"data: " + orjson.dumps({
                        "id": request_id_str,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": active_model,
                        "choices": [{
                            "delta": {},
                            "index": 0,
                            "finish_reason": "tool_calls" if has_tool_calls else "stop",
                        }],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }) + b"\n\n"

                yield _SSE_DONE

        except asyncio.CancelledError:
            # Client disconnected
            logger.warning(f"[{request_id}] Stream cancelled (client disconnected)")
        except httpx.RemoteProtocolError:
            logger.error(f"[{request_id}] Ollama connection reset")
            yield b"data: " + orjson.dumps({"error": {"message": "Upstream connection reset", "type": "upstream_error"}}) + b"\n\n"
            yield _SSE_DONE
        except httpx.ConnectError:
            logger.error(f"[{request_id}] Cannot connect to Ollama")
            yield b"data: " + orjson.dumps({"error": {"message": "Cannot connect to upstream", "type": "upstream_error"}}) + b"\n\n"
            yield _SSE_DONE
        except httpx.ReadTimeout:
            logger.error(f"[{request_id}] Ollama read timeout")
            yield b"data: " + orjson.dumps({"error": {"message": "Upstream read timeout", "type": "upstream_error"}}) + b"\n\n"
            yield _SSE_DONE
        except Exception as e:
            logger.error(f"[{request_id}] STREAM CRASH: {e}")
            yield b"data: " + orjson.dumps({"error": {"message": "Internal server error", "type": "server_error"}}) + b"\n\n"
            yield _SSE_DONE
        finally:
            if not released:
                released = True
                # Release the semaphore
                state.semaphore.release()
                
                # Log inside try block so it does not crash if the task is cancelled
                elapsed = round(time.monotonic() - start_time, 2)
                try:
                    await log_request(request_id, "POST", "/v1/chat/completions", 200, elapsed, prompt_tokens, completion_tokens, "STREAM")
                    logger.info(f"[{request_id}] Done {elapsed}s | P:{prompt_tokens} C:{completion_tokens}")
                except asyncio.CancelledError:
                    # Shield: log on stderr as fallback so we never lose the record
                    logger.warning(f"[{request_id}] Cancelled before logging | {elapsed}s | P:{prompt_tokens} C:{completion_tokens}")
                    pass

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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


if __name__ == "__main__":
    setup_logging(os.environ.get("DEBUG_MODE", "False").lower() in ("true", "1"))
    logger.info(f"Starting server on :{PORT} default_model={MODEL_NAME} models={_MODEL_LIST}")
    run_server()