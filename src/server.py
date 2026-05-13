import os
import time
import asyncio
import contextlib

import orjson
import httpx
import uvloop
import uvicorn
from collections import deque
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.logging import setup_logging, logger
from src.utils import (
    extract_text_content,
    convert_messages_to_ollama,
    format_tool_calls_openai,
)

# ─── Config ───
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3:8b")
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "1"))
NUM_CTX = int(os.environ.get("NUM_CTX", "68768"))
NUM_PREDICT = int(os.environ.get("NUM_PREDICT", "16384"))
NUM_BATCH = int(os.environ.get("NUM_BATCH", "2444"))
FLASH_ATTN = os.environ.get("FLASH_ATTN", "True").lower() in ("true", "1", "yes")
NUM_GPU = int(os.environ.get("NUM_GPU", "-1"))
KEEP_ALIVE = os.environ.get("KEEP_ALIVE", "60m")
PORT = int(os.environ.get("PORT", "8000"))
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
VERBOSE_LOG = os.environ.get("VERBOSE_LOG", "True").lower() in ("true", "1", "yes")
REQUEST_LOG_FILE = os.environ.get("REQUEST_LOG_FILE", "/tmp/gateway-requests.log")

# Precompute ollama options (never changes at runtime)
_OLLAMA_OPTS_BASE = orjson.dumps({
    "num_ctx": NUM_CTX,
    "num_batch": NUM_BATCH,
    "flash_attn": FLASH_ATTN,
    "num_gpu": NUM_GPU,
}).decode()

_OLLAMA_OPTS = orjson.dumps({
    "num_ctx": NUM_CTX,
    "num_batch": NUM_BATCH,
    "flash_attn": FLASH_ATTN,
    "num_gpu": NUM_GPU,
    "num_predict": NUM_PREDICT,
}).decode()

_OLLAMA_OPTS_WARMUP = orjson.dumps({
    "num_ctx": NUM_CTX,
    "num_batch": NUM_BATCH,
    "flash_attn": FLASH_ATTN,
    "num_gpu": NUM_GPU,
    "num_predict": 1,
}).decode()

# Precompute static SSE frame bytes
_MODEL_NAME_BYTES = MODEL_NAME.encode()
_SSE_DONE = b"data: [DONE]\n\n"
_SSE_KEEPALIVE = b': ping\n\n'
_SSE_UPSTREAM_ERR = b'data: {"error":{"message":"Upstream error"}}\n\ndata: [DONE]\n\n'
_RATE_LIMIT_RESPONSE = JSONResponse(status_code=429, content={
    "error": {
        "message": "Server is busy. Try again shortly.",
        "type": "rate_limit_error",
    },
})
_BAD_JSON_RESPONSE = JSONResponse(status_code=400, content={"error": "Invalid JSON"})


# ─── Async-safe log queue (non-blocking) ───
_log_queue = asyncio.Queue(maxsize=200)
_request_log = deque(maxsize=50)


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


async def _log_writer():
    f = open(REQUEST_LOG_FILE, "a")
    buf = []
    try:
        while True:
            try:
                line = await asyncio.wait_for(_log_queue.get(), timeout=2.0)
                buf.append(line + "\n")
                if len(buf) >= 10:
                    f.writelines(buf)
                    f.flush()
                    buf.clear()
            except asyncio.TimeoutError:
                if buf:
                    f.writelines(buf)
                    f.flush()
                    buf.clear()
    except (asyncio.CancelledError, RuntimeError):
        if buf:
            f.writelines(buf)
            f.flush()
    finally:
        f.close()


async def _enqueue_log(line: str):
    try:
        await _log_queue.put(line)
    except asyncio.QueueFull:
        pass


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
    await _enqueue_log(line)


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
    await _enqueue_log(line)


# ─── Fast UUID (4 bytes = 8 hex chars) ───
def _fast_id():
    return os.urandom(4).hex()


# ─── State ───
class GatewayState:
    __slots__ = ("http_client", "semaphore", "log_writer_task", "warmup_task", "is_warm")

    def __init__(self):
        self.http_client = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self.log_writer_task = None
        self.warmup_task = None
        self.is_warm = False


def _get_state(request: Request) -> GatewayState:
    return request.app.state.gw


async def _warmup(state: GatewayState):
    try:
        payload = orjson.dumps({
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "keep_alive": KEEP_ALIVE,
            "options": _OLLAMA_OPTS_WARMUP,
        })
        async with state.http_client.stream(
            "POST", OLLAMA_CHAT_URL,
            content=payload, headers={"Content-Type": "application/json"},
            timeout=300.0,
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                if orjson.loads(line).get("done"):
                    break
        state.is_warm = True
        logger.info("Model is warm and ready!")
    except Exception as e:
        logger.warning(f"Warm-up skipped: {e}")
        state.is_warm = True


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    state = GatewayState()
    state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=60.0, read=900.0, write=60.0, pool=900.0),
        limits=httpx.Limits(
            max_connections=2000,
            max_keepalive_connections=500,
            keepalive_expiry=300,
        ),
    )
    state.log_writer_task = asyncio.create_task(_log_writer())
    state.warmup_task = asyncio.create_task(_warmup(state))

    app.state.gw = state

    banner = (
        f"\n{'='*60}\n"
        f"  \033[1m\033[0;31m🐉 RAGNAROK\033[0m\n"
        f"  \033[1m\033[0;36mGPU Model Gateway\033[0m\n"
        f"{'='*60}\n"
        f"  \033[0;90mModel:\033[0m     {MODEL_NAME}\n"
        f"  \033[0;90mPort:\033[0m      {PORT}\n"
        f"  \033[0;90mConcurrent:\033[0m {MAX_CONCURRENT}\n"
        f"  \033[0;90mContext:\033[0m   {NUM_CTX}\n"
        f"{'='*60}\n"
    )
    print(banner, flush=True)
    logger.info("FastAPI starting up...")
    logger.info(f"Warming up model '{MODEL_NAME}' in background. ..")

    yield

    for task in (state.warmup_task, state.log_writer_task):
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    await state.http_client.aclose()


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
            "id": MODEL_NAME,
            "object": "model",
            "owned_by": "local",
        }],
    }


@app.get("/health")
async def health_check(request: Request):
    state = _get_state(request)
    return {
        "status": "ready" if state.is_warm else "warming",
        "model": MODEL_NAME,
    }


@app.post("/v1/chat/completions")
async def openai_completions(request: Request):
    state = _get_state(request)
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
        body = orjson.loads(await request.body())
    except Exception:
        await log_request(request_id, "POST", "/v1/chat/completions", 400, 0, 0, 0, "BAD_JSON")
        return _BAD_JSON_RESPONSE

    is_streaming = body.get("stream", False)
    ollama_messages = convert_messages_to_ollama(body.get("messages", []))

    msg_count = len(ollama_messages)
    total_chars = sum(len(m.get("content", "")) if isinstance(m.get("content"), str) else 0 for m in ollama_messages)
    logger.info(f"[{request_id}] {msg_count} msgs, ~{total_chars} chars, stream={is_streaming}")

    # Build ollama payload with orjson
    ollama_payload_dict = {
        "model": MODEL_NAME,
        "messages": ollama_messages,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": _OLLAMA_OPTS,
    }
    tools = body.get("tools")
    if tools:
        ollama_payload_dict["tools"] = tools
    tool_choice = body.get("tool_choice")
    if tool_choice:
        ollama_payload_dict["tool_choice"] = tool_choice

    try:
        if not is_streaming:
            return await _handle_non_stream(state, request_id, ollama_payload_dict, start_time)
        return _handle_stream(state, request_id, ollama_payload_dict, start_time)
    finally:
        state.semaphore.release()


async def _handle_non_stream(state, request_id, ollama_payload, start_time):
    content_parts = []
    thinking_parts = []
    all_tool_calls = []
    prompt_tokens = completion_tokens = 0

    try:
        async with state.http_client.stream(
            "POST", OLLAMA_CHAT_URL,
            content=orjson.dumps(ollama_payload),
            headers={"Content-Type": "application/json"},
        ) as response:
            if response.status_code != 200:
                err = await response.aread()
                elapsed = round(time.monotonic() - start_time, 2)
                await log_request(request_id, "POST", "/v1/chat/completions", response.status_code, elapsed, 0, 0, "UPSTREAM_ERR")
                return JSONResponse(status_code=response.status_code, content=orjson.loads(err))

            async for line in response.aiter_lines():
                if not line.strip():
                    continue
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
    except Exception as e:
        elapsed = round(time.monotonic() - start_time, 2)
        await log_request(request_id, "POST", "/v1/chat/completions", 500, elapsed, 0, 0, f"ERR:{str(e)[:40]}")
        return JSONResponse(status_code=500, content={"error": str(e)})

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

    return JSONResponse(content={
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
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
    })


def _suppress_task_exception(task):
    try:
        task.result()
    except (asyncio.CancelledError, Exception):
        pass


def _handle_stream(state, request_id, ollama_payload, start_time):
    request_id_bytes = f"chatcmpl-{request_id}".encode()

    async def stream_generator():
        pending_task = None
        first_chunk = True
        has_tool_calls = False
        prompt_tokens = completion_tokens = 0
        created = int(time.time())

        try:
            async with state.http_client.stream(
                "POST", OLLAMA_CHAT_URL,
                content=orjson.dumps(ollama_payload),
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status_code != 200:
                    elapsed = round(time.monotonic() - start_time, 2)
                    await log_request(request_id, "POST", "/v1/chat/completions", response.status_code, elapsed, 0, 0, "UPSTREAM_ERR")
                    yield _SSE_UPSTREAM_ERR
                    return

                aiter = response.aiter_lines()
                pending_task = asyncio.ensure_future(aiter.__anext__())
                pending_task.add_done_callback(_suppress_task_exception)

                while True:
                    done, _ = await asyncio.wait({pending_task}, timeout=2)

                    if not done:
                        yield _SSE_KEEPALIVE
                        continue

                    try:
                        line = pending_task.result()
                    except StopAsyncIteration:
                        break
                    except Exception:
                        break
                    finally:
                        pending_task = None

                    pending_task = asyncio.ensure_future(aiter.__anext__())
                    pending_task.add_done_callback(_suppress_task_exception)

                    if not line.strip():
                        continue
                    try:
                        data = orjson.loads(line)
                    except Exception:
                        continue
                    if data.get("error"):
                        break

                    message = data.get("message", {})
                    content = extract_text_content(message.get("content"))
                    thinking = message.get("thinking", "")

                    delta = {}
                    if first_chunk:
                        delta["role"] = "assistant"
                        first_chunk = False

                    if thinking:
                        delta["reasoning_content"] = thinking
                    if content:
                        delta["content"] = content

                    tool_calls = message.get("tool_calls")
                    if tool_calls:
                        has_tool_calls = True
                        delta["tool_calls"] = format_tool_calls_openai(tool_calls)
                        for tc in tool_calls:
                            logger.info(f"[{request_id}] Tool: {tc.get('function', {}).get('name', '?')}")
                        if "content" in delta and not delta["content"]:
                            del delta["content"]

                    if not delta and not data.get("done"):
                        continue

                    # orjson.dumps returns bytes directly — no decode/encode round-trip
                    yield b"data: " + orjson.dumps({
                        "id": request_id_bytes,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": _MODEL_NAME_BYTES,
                        "choices": [{
                            "delta": delta,
                            "index": 0,
                            "finish_reason": None,
                        }],
                    }) + b"\n\n"

                    if data.get("done"):
                        prompt_tokens = data.get("prompt_eval_count", 0)
                        completion_tokens = data.get("eval_count", 0)

                        yield b"data: " + orjson.dumps({
                            "id": request_id_bytes,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": _MODEL_NAME_BYTES,
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

        except Exception:
            pass
        finally:
            if pending_task and not pending_task.done():
                pending_task.cancel()
            elapsed = round(time.monotonic() - start_time, 2)
            await log_request(request_id, "POST", "/v1/chat/completions", 200, elapsed, prompt_tokens, completion_tokens, "STREAM")
            logger.info(f"[{request_id}] Done {elapsed}s | P:{prompt_tokens} C:{completion_tokens}")
            yield _SSE_DONE

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
    logger.info(f"Starting server on :{PORT} model={MODEL_NAME}")
    run_server()
