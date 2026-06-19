"""Streaming SSE generator — extracted from server for readability."""

import os
import time
import asyncio

import orjson
import httpx

from fastapi.responses import StreamingResponse

from src.logging import logger, log_request
from src.utils import _fast_id
from src.sse import (
    _SSE_DONE,
    _SSE_KEEPALIVE,
    make_sse_frames,
    build_done_chunk,
)
from src.errors import build_sse_error_frame

# These are read from server at runtime; we avoid importing them to prevent circular deps.
# They're passed via config dict instead.

MAX_RETRIES = 2


def _should_retry_empty() -> bool:
    return os.environ.get("RETRY_ON_EMPTY", "False").lower() in ("true", "1", "yes")


async def stream_generator(state, request_id, ollama_payload, start_time,
                           request_id_str, created, active_model, max_stream_s,
                           ollama_chat_url, sfx, efx):
    """Core async generator that yields SSE frames from Ollama's streaming API."""

    first_chunk = True
    has_tool_calls = False
    tool_call_index = 0
    prompt_tokens = completion_tokens = 0
    released = False
    retry_count = 0

    # ── Immediate ping so client doesn't timeout while we wait for Ollama ──
    yield _SSE_KEEPALIVE

    while retry_count <= MAX_RETRIES:
        # ── Check client disconnected before retry ──
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            logger.warning(f"[{request_id}] Client disconnected — aborting")
            return

        try:
            async with state.http_client.stream(
                "POST", ollama_chat_url,
                json=ollama_payload,
            ) as response:
                if response.status_code != 200:
                    err_body = await response.aread()
                    elapsed = round(time.monotonic() - start_time, 2)
                    logger.error(f"[{request_id}] Ollama Upstream Error: {err_body.decode()[:300]}")
                    await log_request(request_id, "POST", "/v1/chat/completions", response.status_code, elapsed, 0, 0, "UPSTREAM_ERR")
                    yield err_body if err_body.startswith(b"{") else b"data: " + err_body
                    yield _SSE_DONE
                    return

                # ── Send immediate ping to keep connection alive while model thinks ──
                yield _SSE_KEEPALIVE

                # ── Setup direct iterator (no Queue indirection) ──
                line_iter = response.aiter_lines().__aiter__()
                keepalive_count = 0
                graceful = False
                KEEPALIVE_S = 10.0
                raw_lines_captured = 0

                # ── Token batching ──
                batch_content: list[str] = []
                batch_thinking: list[str] = []
                batch_timer = time.monotonic()

                def _flush_batch():
                    nonlocal batch_content, batch_thinking, first_chunk, batch_timer
                    if not batch_content and not batch_thinking:
                        return None
                    delta: dict = {}
                    if first_chunk:
                        delta["role"] = "assistant"
                        first_chunk = False
                    if batch_thinking:
                        delta["reasoning_content"] = "".join(batch_thinking)
                    if batch_content:
                        delta["content"] = "".join(batch_content)
                    batch_content.clear()
                    batch_thinking.clear()
                    batch_timer = time.monotonic()
                    return sfx + orjson.dumps(delta) + efx

                try:
                    while True:
                        # ── Hard Timeout ──
                        stream_elapsed = time.monotonic() - start_time
                        if stream_elapsed > max_stream_s:
                            logger.warning(f"[{request_id}] Hard timeout after {int(stream_elapsed)}s")
                            yield build_sse_error_frame(
                                f"Generation exceeded {max_stream_s}s limit", "timeout"
                            )
                            yield build_done_chunk(
                                request_id_str, created, active_model,
                                has_tool_calls, prompt_tokens, completion_tokens,
                            )
                            yield _SSE_DONE
                            return

                        # ── Race: Ollama data vs keepalive ──
                        try:
                            raw = await asyncio.wait_for(
                                line_iter.__anext__(),
                                timeout=KEEPALIVE_S,
                            )
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError:
                            keepalive_count += 1
                            if keepalive_count > 200:
                                frame = _flush_batch()
                                if frame:
                                    yield frame
                                yield build_sse_error_frame("Upstream timeout", "timeout")
                                yield build_done_chunk(
                                    request_id_str, created, active_model,
                                    has_tool_calls, prompt_tokens, completion_tokens,
                                )
                                yield _SSE_DONE
                                return
                            yield _SSE_KEEPALIVE
                            continue

                        keepalive_count = 0

                        if not raw.strip():
                            continue

                        # ── DEBUG: capture raw Ollama lines ──
                        raw_lines_captured += 1
                        if raw_lines_captured <= 3 or (raw_lines_captured % 50 == 0):
                            logger.info(f"[{request_id}] RAW#{raw_lines_captured}: {raw[:200]}")

                        try:
                            data = orjson.loads(raw)
                        except Exception as e:
                            logger.error(f"[{request_id}] Parse fail: {e} | line={raw[:100]}")
                            continue

                        # ── Ollama-side error ──
                        if data.get("error"):
                            logger.error(f"[{request_id}] Ollama error: {data.get('error')}")
                            frame = _flush_batch()
                            if frame:
                                yield frame
                            yield build_sse_error_frame(data["error"])
                            yield build_done_chunk(
                                request_id_str, created, active_model,
                                has_tool_calls, prompt_tokens, completion_tokens,
                            )
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

                        # ── Accumulate into batch ──
                        if thinking:
                            batch_thinking.append(thinking)
                        if content:
                            batch_content.append(content)

                        should_flush = False

                        # ── Tool calls force immediate flush ──
                        if tool_calls:
                            has_tool_calls = True
                            formatted = []
                            for tc in tool_calls:
                                tc_name = tc.get("function", {}).get("name") or "?"
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

                            # Flush buffered text first, then yield tool call
                            frame = _flush_batch()
                            if frame:
                                yield frame
                            try:
                                delta: dict = {}
                                if first_chunk:
                                    delta["role"] = "assistant"
                                    first_chunk = False
                                delta["tool_calls"] = formatted
                                yield sfx + orjson.dumps(delta) + efx
                            except Exception as ex:
                                logger.error(f"[{request_id}] Tool call serialize failed: {ex}")
                            should_flush = True

                        # ── Time-based flush ──
                        if (time.monotonic() - batch_timer) > 0.1:
                            should_flush = True

                        if should_flush:
                            frame = _flush_batch()
                            if frame:
                                yield frame
                            batch_timer = time.monotonic()

                        # ── Done from Ollama ──
                        if data.get("done"):
                            prompt_tokens = data.get("prompt_eval_count", 0)
                            completion_tokens = data.get("eval_count", 0)
                            graceful = True
                            logger.info(
                                f"[{request_id}] DONE chunk | prompt_eval_count={data.get('prompt_eval_count')} "
                                f"eval_count={data.get('eval_count')} done_reason={data.get('done_reason')} "
                                f"total_duration={data.get('total_duration')} load_duration={data.get('load_duration')} "
                                f"lines_received={raw_lines_captured}"
                            )

                            # Flush remaining tokens before done chunk
                            frame = _flush_batch()
                            if frame:
                                yield frame

                            yield build_done_chunk(
                                request_id_str, created, active_model,
                                has_tool_calls, prompt_tokens, completion_tokens,
                            )
                            break
                finally:
                    # ── Zero-token detection: retry or graceful exit ──
                    if not graceful and prompt_tokens == 0 and completion_tokens == 0:
                        retry_count += 1
                        logger.warning(
                            f"[{request_id}] Empty stream (attempt {retry_count}/{MAX_RETRIES})"
                        )
                        if _should_retry_empty() and retry_count <= MAX_RETRIES:
                            yield _SSE_KEEPALIVE
                            await asyncio.sleep(1)
                            continue  # retry the request
                        # Either retries disabled or exhausted — yield valid empty completion
                        yield build_done_chunk(
                            request_id_str, created, active_model,
                            has_tool_calls, prompt_tokens, completion_tokens,
                        )
                    elif not graceful:
                        yield build_done_chunk(
                            request_id_str, created, active_model,
                            has_tool_calls, prompt_tokens, completion_tokens,
                        )

            yield _SSE_DONE
            break  # success — exit retry loop

        except asyncio.CancelledError:
            logger.warning(f"[{request_id}] Stream cancelled (client disconnected)")
            return
        except httpx.RemoteProtocolError:
            logger.error(f"[{request_id}] Ollama connection reset")
            yield build_sse_error_frame("Upstream connection reset", "upstream_error")
            yield build_done_chunk(
                request_id_str, created, active_model,
                has_tool_calls, prompt_tokens, completion_tokens,
            )
            yield _SSE_DONE
            return
        except httpx.ConnectError:
            logger.error(f"[{request_id}] Cannot connect to Ollama")
            yield build_sse_error_frame("Cannot connect to upstream", "upstream_error")
            yield build_done_chunk(
                request_id_str, created, active_model,
                has_tool_calls, prompt_tokens, completion_tokens,
            )
            yield _SSE_DONE
            return
        except httpx.ReadTimeout:
            logger.error(f"[{request_id}] Ollama read timeout")
            yield build_sse_error_frame("Upstream read timeout", "upstream_error")
            yield build_done_chunk(
                request_id_str, created, active_model,
                has_tool_calls, prompt_tokens, completion_tokens,
            )
            yield _SSE_DONE
            return
        except Exception as e:
            logger.error(f"[{request_id}] STREAM CRASH: {e}")
            yield build_sse_error_frame("Internal server error", "server_error")
            yield build_done_chunk(
                request_id_str, created, active_model,
                has_tool_calls, prompt_tokens, completion_tokens,
            )
            yield _SSE_DONE
            return
        finally:
            if not released:
                released = True
                state.semaphore.release()
                elapsed = round(time.monotonic() - start_time, 2)
                try:
                    await log_request(request_id, "POST", "/v1/chat/completions", 200,
                                      elapsed, prompt_tokens, completion_tokens, "STREAM")
                    logger.info(f"[{request_id}] Done {elapsed}s | P:{prompt_tokens} C:{completion_tokens}")
                except asyncio.CancelledError:
                    logger.warning(
                        f"[{request_id}] Cancelled before logging | {elapsed}s "
                        f"| P:{prompt_tokens} C:{completion_tokens}"
                    )


def handle_stream(state, request_id, ollama_payload, start_time, active_model,
                  max_stream_seconds: int, ollama_chat_url: str):
    """Entry point called from server route handler."""
    import time as _time

    request_id_str = f"chatcmpl-{request_id}"
    created = int(_time.time())
    sfx, efx = make_sse_frames(active_model, request_id_str, created)

    return StreamingResponse(
        stream_generator(state, request_id, ollama_payload, start_time,
                         request_id_str, created, active_model,
                         max_stream_seconds, ollama_chat_url, sfx, efx),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )