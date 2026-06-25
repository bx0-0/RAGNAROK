"""Streaming SSE generator — extracted from server for readability.

Uses raw httpx streaming to Ollama so we own the full HTTP connection
lifecycle. On client disconnect we explicitly call response.aclose()
to abort the upstream request and free GPU immediately.
"""

import os
import time
import asyncio
import json as _json

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
from src.config import KEEP_ALIVE, OLLAMA_BASE_URL
from src.errors import build_sse_error_frame

MAX_RETRIES = 2


def _should_retry_empty() -> bool:
    return os.environ.get("RETRY_ON_EMPTY", "False").lower() in ("true", "1", "yes")


async def stream_generator(state, request_id, ollama_payload, start_time,
                           request_id_str, created, active_model, max_stream_s,
                           ollama_chat_url, sfx, efx):
    """Core async generator that yields SSE frames from Ollama's streaming API.

    Semaphore release is handled by the caller via a try/finally wrapper around
    this generator to guarantee cleanup even on abrupt disconnect / GC without
    finally execution.
    """

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
        except (asyncio.CancelledError, GeneratorExit):
            logger.warning(f"[{request_id}] Client disconnected — aborting")
            return

        try:
            # ── Send immediate ping to keep connection alive while model thinks ──
            yield _SSE_KEEPALIVE

            # ── Token batching ──
            batch_content: list[str] = []
            batch_thinking: list[str] = []
            batch_timer = time.monotonic()
            chunks_captured = 0
            died_mid_stream = False
            graceful = False

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
                # ── Raw httpx stream — we own the HTTP connection lifecycle ──
                async with httpx.AsyncClient(follow_redirects=True).stream(
                    "POST", ollama_chat_url, json=ollama_payload, timeout=1800.0
                ) as response:
                    response.raise_for_status()

                    # Create iterator ONCE — reusing it prevents StreamError on retry
                    line_iter = response.aiter_lines()

                    while True:
                        # ── Hard Timeout ──
                        stream_elapsed = time.monotonic() - start_time
                        if stream_elapsed > max_stream_s:
                            logger.warning(f"[{request_id}] Hard timeout after {int(stream_elapsed)}s")
                            frame = _flush_batch()
                            if frame:
                                yield frame
                            yield build_sse_error_frame(
                                f"Generation exceeded {max_stream_s}s limit", "timeout"
                            )
                            yield build_done_chunk(
                                request_id_str, created, active_model,
                                has_tool_calls, prompt_tokens, completion_tokens,
                            )
                            yield _SSE_DONE
                            return

                        # ── Read next line with timeout for keepalive ──
                        CHUNK_TIMEOUT = 60
                        try:
                            line = await asyncio.wait_for(
                                line_iter.__anext__(),
                                timeout=CHUNK_TIMEOUT,
                            )
                        except StopAsyncIteration:
                            # No more data from Ollama — stream ended
                            break
                        except asyncio.TimeoutError:
                            logger.info(f"[{request_id}] Keepalive ping (Ollama gap > {CHUNK_TIMEOUT}s)")
                            yield _SSE_KEEPALIVE
                            continue

                        if not line.strip():
                            continue

                        chunks_captured += 1
                        chunk = _json.loads(line)

                        # ── Extract message data ──
                        msg = chunk.get("message", {}) or {}
                        if not msg and "done" in chunk:
                            # Final metadata chunk
                            prompt_tokens = chunk.get("prompt_eval_count", 0) or 0
                            completion_tokens = chunk.get("eval_count", 0) or 0
                            graceful = True
                            logger.debug(
                                f"[{request_id}] DONE | P={chunk.get('prompt_eval_count')} C={chunk.get('eval_count')} "
                                f"reason={chunk.get('done_reason')} chunks={chunks_captured}"
                            )
                            frame = _flush_batch()
                            if frame:
                                yield frame
                            yield build_done_chunk(
                                request_id_str, created, active_model,
                                has_tool_calls, prompt_tokens, completion_tokens,
                            )
                            break
                        elif not msg:
                            continue

                        # ── Extract content ──
                        content = msg.get("content") or ""
                        if isinstance(content, list):
                            content = " ".join(
                                item.get("text", "")
                                for item in content
                                if isinstance(item, dict) and item.get("type") == "text"
                            )

                        # ── Extract thinking ──
                        thinking = msg.get("thinking", "") or ""

                        # ── Accumulate into batch (guard against None) ──
                        if thinking:
                            batch_thinking.append(thinking)
                        if content:
                            batch_content.append(content)

                        should_flush = False

                        # ── Tool calls force immediate flush ──
                        tool_calls = msg.get("tool_calls")
                        if tool_calls:
                            has_tool_calls = True
                            formatted = []
                            for tc in tool_calls:
                                if tc is None:
                                    continue
                                tc_func = tc.get("function", {}) or {}
                                tc_name = tc_func.get("name", "?") or "?"
                                tc_args = tc_func.get("arguments", "") or ""
                                if isinstance(tc_args, dict):
                                    tc_args_json = orjson.dumps(tc_args).decode()
                                else:
                                    tc_args_json = str(tc_args) if tc_args else "{}"
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

                        # ── Time-based flush (50ms on slow GPUs) ──
                        if (time.monotonic() - batch_timer) > 0.05:
                            should_flush = True

                        if should_flush:
                            frame = _flush_batch()
                            if frame:
                                yield frame
                            batch_timer = time.monotonic()

                # ── `async with` exited — response is already closed/aborted on any exit path ──

            except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout) as e:
                died_mid_stream = True
                logger.error(f"[{request_id}] Ollama connection error: {e}")
            except asyncio.CancelledError:
                # Client disconnected — `async with` already aborted the request above
                logger.warning(f"[{request_id}] Client disconnected, aborting generation")
                raise
            except GeneratorExit:
                # Starlette calls .aclose() on our generator → raises GeneratorExit here
                logger.warning(f"[{request_id}] Generator closed (client disconnect), aborting generation")
                raise
            except Exception as e:
                logger.error(f"[{request_id}] Stream loop error: {e}")
            else:
                # ── Stream exited normally WITHOUT chunk.done ──
                if not graceful and chunks_captured > 0:
                    logger.warning(
                        f"[{request_id}] Stream ended without finish_reason "
                        f"after {chunks_captured} chunks — yielding error to trigger retry"
                    )
                    frame = _flush_batch()
                    if frame:
                        yield frame
                    yield build_sse_error_frame(
                        "Stream ended without finish_reason", "incomplete_stream"
                    )

            # ── Flush remaining thinking/content on ANY exit (cancelled, timeout, or normal) ──
            frame = _flush_batch()
            if frame:
                yield frame

            # ── Zero-token detection: retry or graceful exit ──
            if not graceful and prompt_tokens == 0 and completion_tokens == 0:
                if died_mid_stream:
                    logger.error(
                        f"[{request_id}] Ollama died mid-stream after {chunks_captured} chunks — retrying"
                    )
                    retry_count += 1
                    if retry_count <= MAX_RETRIES:
                        await asyncio.sleep(2)
                        continue  # retry
                    yield build_sse_error_frame("Upstream model crashed mid-generation", "upstream_error")
                    yield build_done_chunk(
                        request_id_str, created, active_model,
                        has_tool_calls, prompt_tokens, completion_tokens,
                    )
                else:
                    retry_count += 1
                    logger.warning(
                        f"[{request_id}] Empty stream (attempt {retry_count}/{MAX_RETRIES})"
                    )
                    if _should_retry_empty() and retry_count <= MAX_RETRIES:
                        yield _SSE_KEEPALIVE
                        await asyncio.sleep(1)
                        continue  # retry the request
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

        except (asyncio.CancelledError, GeneratorExit):
            logger.warning(f"[{request_id}] Stream cancelled (client disconnected)")
            return
        except Exception as e:
            elapsed = round(time.monotonic() - start_time, 2)
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
                except (asyncio.CancelledError, GeneratorExit):
                    logger.warning(
                        f"[{request_id}] Cancelled before logging | {elapsed}s "
                        f"| P:{prompt_tokens} C:{completion_tokens}"
                    )


def handle_stream(state, request_id, ollama_payload, start_time, active_model,
                  max_stream_seconds: int, ollama_chat_url: str):
    """Entry point called from server route handler."""

    request_id_str = f"chatcmpl-{request_id}"
    created = int(time.time())
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
