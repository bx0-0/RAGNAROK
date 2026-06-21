"""POST /v1/chat/completions — main chat endpoint + non-stream handler."""

import asyncio
import time

import orjson
from fastapi import APIRouter, Request
from fastapi.responses import Response

from src.config import (
    MODEL_NAME,
    KEEP_ALIVE,
    OLLAMA_BASE_URL,
    MAX_STREAM_SECONDS,
    _OLLAMA_OPTS,
)
from src.state import _get_state
from src.logging import log_request_start, log_request
from src.utils import (
    convert_messages_to_ollama,
    format_tool_calls_openai,
    _read_body,
    _fast_id,
)
from src.errors import _RATE_LIMIT_RESPONSE, _BAD_JSON_RESPONSE
from src.streaming import handle_stream

router = APIRouter()


@router.post("/v1/chat/completions")
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
        import sys
        sys.stderr.write(f"[WARN] [{request_id}] Rejected: busy\n")
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
    import sys
    sys.stderr.write(
        f"[INFO] [{request_id}] Client={client_name} | Msgs={msg_count} | "
        f"Chars={total_chars} | Tools={tool_names}\n"
    )

    is_streaming = body.get("stream", False)

    # ── Resolve model: use client's model param if valid, else default ──
    requested_model = body.get("model", "")
    from src.config import _MODEL_LIST
    if requested_model and requested_model in _MODEL_LIST:
        active_model = requested_model
    else:
        active_model = MODEL_NAME

    ollama_messages = convert_messages_to_ollama(body.get("messages", []), has_tools=bool(tools))

    # Build ollama payload with orjson
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
            import sys
            sys.stderr.write(f"[ERROR] [{request_id}] Non-stream handler crashed: {e}\n")
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
        return handle_stream(state, request_id, ollama_payload_dict, start_time, active_model,
                            MAX_STREAM_SECONDS, OLLAMA_BASE_URL + "/api/chat")


async def _handle_non_stream(state, request_id, ollama_payload, start_time, created, active_model, request):
    content_parts = []
    thinking_parts = []
    all_tool_calls = []
    prompt_tokens = completion_tokens = 0

    try:
        # Build chat kwargs from the ollama payload dict
        chat_kwargs = {
            "model": ollama_payload["model"],
            "messages": ollama_payload["messages"],
            "stream": False,
            "keep_alive": ollama_payload.get("keep_alive"),
            "options": ollama_payload.get("options"),
        }
        if "tools" in ollama_payload:
            chat_kwargs["tools"] = ollama_payload["tools"]
        if "tool_choice" in ollama_payload:
            chat_kwargs["tool_choice"] = ollama_payload["tool_choice"]

        response = await state.http_client.chat(**chat_kwargs)

        # Extract from ChatResponse
        msg = response.message
        if msg.content:
            content_parts.append(msg.content)
        thinking = getattr(msg, 'thinking', None)
        if thinking:
            thinking_parts.append(thinking)
        if msg.tool_calls:
            all_tool_calls.extend(format_tool_calls_openai(msg.tool_calls))
        prompt_tokens = response.prompt_eval_count or 0
        completion_tokens = response.eval_count or 0

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
