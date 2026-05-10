import os
import time
import uuid
import json
import asyncio

try:
    import orjson
    def _json_dumps(obj):
        return orjson.dumps(obj).decode("utf-8")
    def _json_loads(text):
        return orjson.loads(text)
except ImportError:
    def _json_dumps(obj):
        return json.dumps(obj, ensure_ascii=False)
    def _json_loads(text):
        return json.loads(text)

import uvloop
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from src.logging import setup_logging, logger
from src.utils import (
    extract_text_content,
    convert_messages_to_ollama,
    format_tool_calls_openai,
)

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

VERBOSE_LOG = os.environ.get("VERBOSE_LOG", "True").lower() in ("true", "1", "yes")

# Request log file — tail this to see live requests
REQUEST_LOG_FILE = os.environ.get("REQUEST_LOG_FILE", "/tmp/gateway-requests.log")

request_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
http_client = None

# ---- In-memory request log (circular, keeps last N) ----
MAX_LOG_ENTRIES = 50
request_log = []

def log_request(req_id, method, path, status, duration, tokens_in, tokens_out, extra=""):
    entry = {
        "id": req_id,
        "method": method,
        "path": path,
        "status": status,
        "duration": round(duration, 2),
        "t_in": tokens_in,
        "t_out": tokens_out,
        "extra": extra,
        "time": time.strftime("%H:%M:%S"),
    }
    request_log.append(entry)
    if len(request_log) > MAX_LOG_ENTRIES:
        request_log.pop(0)
    line = _format_request_line(entry)
    # Print to both stdout (for direct python runs) and log file
    if _print_to_stdout:
        print(line, flush=True)
    with open(REQUEST_LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()

def _format_request_line(entry):
    status_color = "\033[0;32m" if entry["status"] == 200 else "\033[0;31m"
    reset = "\033[0m"
    line = (
        f"\n  ┌─ {entry['time']} | {entry['method']} {entry['path']}"
        f"\n  │  Status: {status_color}{entry['status']}{reset}  "
        f"Duration: {entry['duration']}s  "
        f"Tokens: {entry['t_in']}→{entry['t_out']}"
    )
    if entry["extra"]:
        line += f"  │  {entry['extra']}"
    line += "\n  └─"
    return line

_print_to_stdout = False  # Turned on by --verbose-log via env

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    logger.info("FastAPI starting up...")
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=60.0, read=900.0, write=60.0, pool=900.0),
        limits=httpx.Limits(
            max_connections=200,
            max_keepalive_connections=100,
            keepalive_expiry=300,
        ),
    )

    logger.info(f"Warming up model '{MODEL_NAME}'...")
    try:
        warmup_payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "keep_alive": KEEP_ALIVE,
            "options": {
                "num_ctx": NUM_CTX,
                "num_predict": 1,
                "num_batch": NUM_BATCH,
                "flash_attn": FLASH_ATTN,
                "num_gpu": NUM_GPU,
            },
        }
        async with http_client.stream(
            "POST", f"{OLLAMA_BASE_URL}/api/chat", json=warmup_payload, timeout=300.0
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                data = _json_loads(line)
                if data.get("done"):
                    break
        logger.info("Model is warm and ready!")
    except Exception as e:
        logger.warning(f"Warm-up skipped: {e}")

    yield
    try:
        await http_client.aclose()
    except Exception:
        pass

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
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }

@app.post("/v1/chat/completions")
async def openai_completions(request: Request):
    request_id = uuid.uuid4().hex[:8]
    start_time = time.monotonic()

    if request_semaphore.locked():
        logger.warning(f"[{request_id}] Rejected: busy")
        log_request(request_id, "POST", "/v1/chat/completions", 429, 0, 0, 0, "RATE_LIMITED")
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": "Server is busy. Try again shortly.",
                    "type": "rate_limit_error",
                }
            },
        )

    try:
        body = await request.json()
    except Exception:
        log_request(request_id, "POST", "/v1/chat/completions", 400, 0, 0, 0, "BAD_JSON")
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    is_client_streaming = body.get("stream", False)
    ollama_messages = convert_messages_to_ollama(body.get("messages", []))

    msg_count = len(ollama_messages)
    total_chars = sum(len(str(m.get("content", ""))) for m in ollama_messages)
    logger.info(f"[{request_id}] {msg_count} msgs, ~{total_chars} chars, stream={is_client_streaming}")

    ollama_payload = {
        "model": MODEL_NAME,
        "messages": ollama_messages,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "num_batch": NUM_BATCH,
            "flash_attn": FLASH_ATTN,
            "num_gpu": NUM_GPU,
        },
    }

    if "tools" in body and body["tools"]:
        ollama_payload["tools"] = body["tools"]
    if "tool_choice" in body:
        ollama_payload["tool_choice"] = body["tool_choice"]

    ollama_url = f"{OLLAMA_BASE_URL}/api/chat"

    async with request_semaphore:
        if not is_client_streaming:
            result = await handle_non_stream(
                request_id, ollama_url, ollama_payload, start_time
            )
        else:
            result = handle_stream(
                request_id, ollama_url, ollama_payload, start_time
            )
        return result

async def handle_non_stream(request_id, ollama_url, ollama_payload, start_time):
    content_parts = []
    thinking_parts = []
    all_tool_calls = []
    prompt_tokens = 0
    completion_tokens = 0

    try:
        async with http_client.stream(
            "POST", ollama_url, json=ollama_payload
        ) as response:
            if response.status_code != 200:
                err = await response.aread()
                elapsed = round(time.monotonic() - start_time, 2)
                log_request(request_id, "POST", "/v1/chat/completions", response.status_code, elapsed, 0, 0, "UPSTREAM_ERR")
                return JSONResponse(
                    status_code=response.status_code,
                    content=_json_loads(err.decode()),
                )

            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                data = _json_loads(line)
                msg = data.get("message", {})

                if msg.get("content"):
                    content_parts.append(msg["content"])
                if msg.get("thinking"):
                    thinking_parts.append(msg["thinking"])
                if "tool_calls" in msg and msg["tool_calls"]:
                    all_tool_calls.extend(
                        format_tool_calls_openai(msg["tool_calls"])
                    )

                if data.get("done"):
                    prompt_tokens = data.get("prompt_eval_count", 0)
                    completion_tokens = data.get("eval_count", 0)
                    break
    except Exception as e:
        elapsed = round(time.monotonic() - start_time, 2)
        log_request(request_id, "POST", "/v1/chat/completions", 500, elapsed, 0, 0, f"ERR:{str(e)[:40]}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    elapsed = round(time.monotonic() - start_time, 2)
    log_request(request_id, "POST", "/v1/chat/completions", 200, elapsed, prompt_tokens, completion_tokens, "NON-STREAM")

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

def handle_stream(request_id, ollama_url, ollama_payload, start_time):
    async def stream_generator():
        first_chunk = True
        has_tool_calls = False
        prompt_tokens = 0
        completion_tokens = 0

        try:
            async with http_client.stream(
                "POST", ollama_url, json=ollama_payload
            ) as response:
                if response.status_code != 200:
                    elapsed = round(time.monotonic() - start_time, 2)
                    log_request(request_id, "POST", "/v1/chat/completions", response.status_code, elapsed, 0, 0, "UPSTREAM_ERR")
                    yield (
                        b"data: "
                        + b'{"error":{"message":"Upstream error"}}'
                        + b"\n\ndata: [DONE]\n\n"
                    )
                    return

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        data = _json_loads(line)
                    except Exception:
                        continue

                    if "error" in data:
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

                    if "tool_calls" in message and message["tool_calls"]:
                        has_tool_calls = True
                        delta["tool_calls"] = format_tool_calls_openai(
                            message["tool_calls"]
                        )
                        for tc in message["tool_calls"]:
                            tool_name = tc.get("function", {}).get(
                                "name", "unknown_tool"
                            )
                            logger.info(f"[{request_id}] Tool: {tool_name}")
                        if "content" in delta and not delta["content"]:
                            del delta["content"]

                    if not delta and not data.get("done"):
                        continue

                    chunk = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": MODEL_NAME,
                        "choices": [{
                            "delta": delta,
                            "index": 0,
                            "finish_reason": None,
                        }],
                    }
                    yield b"data: " + _json_dumps(chunk).encode() + b"\n\n"

                    if data.get("done"):
                        prompt_tokens = data.get("prompt_eval_count", 0)
                        completion_tokens = data.get("eval_count", 0)
                        break

        except Exception:
            pass
        finally:
            elapsed = round(time.monotonic() - start_time, 2)
            log_request(request_id, "POST", "/v1/chat/completions", 200, elapsed, prompt_tokens, completion_tokens, "STREAM")
            logger.info(f"[{request_id}] Done {elapsed}s | P:{prompt_tokens} C:{completion_tokens}")
            yield b"data: [DONE]\n\n"

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
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="warning",
        http="httptools",
        loop="uvloop",
        timeout_keep_alive=300,
        access_log=False,
    )
    server = uvicorn.Server(config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.serve())

if __name__ == "__main__":
    setup_logging(os.environ.get("DEBUG_MODE", "False").lower() in ("true", "1"))
    logger.info(f"Starting server on :{PORT} model={MODEL_NAME}")
    run_server()
