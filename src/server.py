import sys
import os
import time
import uuid
import json
import asyncio
import threading

try:
    import orjson
    def json_dumps(obj):
        return orjson.dumps(obj).decode("utf-8")
    def json_loads(text):
        return orjson.loads(text)
except ImportError:
    def json_dumps(obj):
        return json.dumps(obj, ensure_ascii=False)
    def json_loads(text):
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

request_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
http_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    logger.info("FastAPI starting up...")
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=60.0, read=900.0, write=60.0, pool=900.0),
        limits=httpx.Limits(max_keepalive_connections=500, max_connections=2000),
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
                data = json_loads(line)
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

    if request_semaphore.locked():
        logger.warning(f"[{request_id}] Rejected: Server is busy")
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": "Server is busy. Only one request at a time.",
                    "type": "rate_limit_error",
                }
            },
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    is_client_streaming = body.get("stream", False)
    ollama_messages = convert_messages_to_ollama(body.get("messages", []))

    msg_count = len(ollama_messages)
    total_chars = sum(len(m.get("content", "")) for m in ollama_messages)
    logger.info(f"[{request_id}] Payload: {msg_count} msgs, ~{total_chars} chars")

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
        logger.info(f"[{request_id}] Processing started (stream={is_client_streaming})")

        if not is_client_streaming:
            return await handle_non_stream(
                request_id, ollama_url, ollama_payload
            )
        else:
            return handle_stream(request_id, ollama_url, ollama_payload)

async def handle_non_stream(request_id, ollama_url, ollama_payload):
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
                return JSONResponse(
                    status_code=response.status_code,
                    content=json_loads(err.decode()),
                )

            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                data = json_loads(line)
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
        return JSONResponse(status_code=500, content={"error": str(e)})

    resp_message = {
        "role": "assistant",
        "content": "".join(content_parts) or None,
    }
    if thinking_parts:
        resp_message["reasoning_content"] = "".join(thinking_parts)
    if all_tool_calls:
        resp_message["tool_calls"] = all_tool_calls

    logger.info(
        f"[{request_id}] Done | P:{prompt_tokens} C:{completion_tokens}"
    )

    return JSONResponse(
        content={
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": resp_message,
                    "finish_reason": "tool_calls" if all_tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )

def suppress_task_exception(task):
    try:
        task.result()
    except (
        asyncio.CancelledError,
        httpx.ReadError,
        httpx.RemoteProtocolError,
        Exception,
    ):
        pass

def handle_stream(request_id, ollama_url, ollama_payload):
    async def stream_generator():
        pending_task = None
        first_chunk = True
        has_tool_calls = False
        prompt_tokens = 0
        completion_tokens = 0

        try:
            async with http_client.stream(
                "POST", ollama_url, json=ollama_payload
            ) as response:
                if response.status_code != 200:
                    err = await response.aread()
                    yield (
                        b"data: "
                        + orjson.dumps(
                            {"error": {"message": "Upstream error"}}
                        )
                        + b"\n\ndata: [DONE]\n\n"
                    )
                    return

                aiter = response.aiter_lines()
                pending_task = asyncio.ensure_future(aiter.__anext__())
                pending_task.add_done_callback(suppress_task_exception)

                while True:
                    done, _ = await asyncio.wait({pending_task}, timeout=5.0)

                    if not done:
                        yield (
                            b'data: {"id":"keep-alive","choices":'
                            b'[{"delta":{},"finish_reason":null}]}\n\n'
                        )
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
                    pending_task.add_done_callback(suppress_task_exception)

                    if not line.strip():
                        continue

                    try:
                        data = json_loads(line)
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
                            logger.info(
                                f"[{request_id}] Tool call: {tool_name}"
                            )
                        if (
                            "content" in delta
                            and not delta["content"]
                        ):
                            del delta["content"]

                    if not delta and not data.get("done"):
                        continue

                    chunk = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": MODEL_NAME,
                        "choices": [
                            {
                                "delta": delta,
                                "index": 0,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield b"data: " + orjson.dumps(chunk) + b"\n\n"

                    if data.get("done"):
                        prompt_tokens = data.get("prompt_eval_count", 0)
                        completion_tokens = data.get("eval_count", 0)

                        logger.info(
                            f"[{request_id}] Done | P:{prompt_tokens} C:{completion_tokens}"
                        )

                        final_chunk = {
                            "id": f"chatcmpl-{request_id}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": MODEL_NAME,
                            "choices": [
                                {
                                    "delta": {},
                                    "index": 0,
                                    "finish_reason": "tool_calls"
                                    if has_tool_calls
                                    else "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens
                                + completion_tokens,
                            },
                        }
                        yield (
                            b"data: " + orjson.dumps(final_chunk) + b"\n\n"
                        )
                        yield b"data: [DONE]\n\n"
                        break

        except httpx.ReadError:
            pass
        except httpx.RemoteProtocolError:
            pass
        except GeneratorExit:
            pass
        except Exception:
            pass
        finally:
            if pending_task and not pending_task.done():
                pending_task.cancel()
            logger.info(f"[{request_id}] Connection closed")

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
    )
    server = uvicorn.Server(config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.serve())

if __name__ == "__main__":
    setup_logging(os.environ.get("DEBUG_MODE", "False").lower() in ("true", "1"))
    logger.info(f"Starting server on port {PORT}, model={MODEL_NAME}")
    run_server()
