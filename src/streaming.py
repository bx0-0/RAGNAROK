"""Streaming SSE generator — extracted from server for readability."""

import os
import time
import asyncio

import orjson
import httpx
import ollama

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

    # ── Build chat kwargs from the ollama payload dict ──
    chat_kwargs = {
        "model": ollama_payload["model"],
        "messages": ollama_payload["messages"],
        "stream": True,
    }
    if ollama_payload.get("keep_alive"):
        chat_kwargs["keep_alive"] = ollama_payload["keep_alive"]
    if ollama_payload.get("options"):
        chat_kwargs["options"] = ollama_payload["options"]
    if ollama_payload.get("tools"):
        chat_kwargs["tools"] = ollama_payload["tools"]
    # NB: ollama lib does not accept tool_choice kwarg; the API itself
    # defaults to "auto" when tools are provided, which is correct behavior.

    # ── Immediate ping so client doesn't timeout while we wait for Ollama ──
    yield _SSE_KEEPALIVE

    pusher_task = None  # track for cancellation on disconnect

    while retry_count <= MAX_RETRIES:
        # ── Check client disconnected before retry ──
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
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
                # ── Queue-based consumer — single task drives the async generator ──
                chunk_queue: asyncio.Queue = asyncio.Queue()

                async def _queue_pusher():
                    """Drive the async generator and push chunks into queue."""
                    chat_stream = None
                    try:
                        chat_stream = await state.http_client.chat(**chat_kwargs)
                        async for chunk in chat_stream:
                            await chunk_queue.put(chunk)
                        # Signal end-of-stream
                        await chunk_queue.put(StopAsyncIteration)
                    except StopAsyncIteration:
                        await chunk_queue.put(StopAsyncIteration)
                    except asyncio.CancelledError:
                        # Closing the HTTP stream aborts Ollama generation → frees GPU immediately
                        if chat_stream is not None:
                            try:
                                await chat_stream.aclose()
                            except Exception:
                                pass
                        raise
                    except Exception as e:
                        await chunk_queue.put(e)

                CHUNK_TIMEOUT = 60  # send keepalive if no chunk arrives in 60s
                _pusher = asyncio.create_task(_queue_pusher())
                pusher_task = _pusher  # save ref for disconnect handling

                while True:
                    # Wait for next chunk with a timeout so we can detect long gaps
                    try:
                        chunk_or_sentinel = await asyncio.wait_for(
                            chunk_queue.get(), timeout=CHUNK_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        # No chunk arrived within CHUNK_TIMEOUT seconds — send keepalive
                        logger.info(f"[{request_id}] Keepalive ping (Ollama gap > {CHUNK_TIMEOUT}s)")
                        yield _SSE_KEEPALIVE
                        continue

                    # Handle queue errors
                    if isinstance(chunk_or_sentinel, Exception):
                        raise chunk_or_sentinel
                    if chunk_or_sentinel is StopAsyncIteration:
                        # No more chunks from Ollama — stream ended
                        break

                    chunk = chunk_or_sentinel

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

                    chunks_captured += 1

                    msg = chunk.message
                    if msg is None:
                        # Empty final chunk — done
                        continue

                    # ── Extract content ──
                    content = msg.content or ""
                    if isinstance(content, list):
                        content = " ".join(
                            item.get("text", "")
                            for item in content
                            if isinstance(item, dict) and item.get("type") == "text"
                        )

                    # ── Extract thinking ──
                    thinking = getattr(msg, "thinking", "") or ""

                    # ── Accumulate into batch (guard against None) ──
                    if thinking:
                        batch_thinking.append(thinking)
                    if content:
                        batch_content.append(content)

                    should_flush = False

                    # ── Tool calls force immediate flush ──
                    tool_calls = msg.tool_calls
                    if tool_calls:  # guard against None
                        has_tool_calls = True
                        formatted = []
                        for tc in tool_calls:
                            if tc is None:
                                continue
                            tc_func = getattr(tc, "function", None)
                            if tc_func is None:
                                tc_name = "?"
                                tc_args_json = "{}"
                            else:
                                tc_name = getattr(tc_func, "name", "?") or "?"
                                tc_args = getattr(tc_func, "arguments", None) or ""
                                if isinstance(tc_args, str):
                                    tc_args_json = tc_args
                                else:
                                    tc_args_json = orjson.dumps(tc_args).decode() if tc_args else "{}"
                            is_write_tool = tc_name == "write"
                            formatted.append({
                                "index": tool_call_index,
                                "id": getattr(tc, "id", None) or f"call_{_fast_id()}",
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

                    # ── Time-based flush (30ms on slow GPUs, 100ms was too much latency) ──
                    if (time.monotonic() - batch_timer) > 0.05:
                        should_flush = True

                    if should_flush:
                        frame = _flush_batch()
                        if frame:
                            yield frame
                        batch_timer = time.monotonic()

                    # ── Done from Ollama ──
                    if chunk.done:
                        prompt_tokens = chunk.prompt_eval_count or 0
                        completion_tokens = chunk.eval_count or 0
                        graceful = True
                        logger.debug(
                            f"[{request_id}] DONE | P={chunk.prompt_eval_count} C={chunk.eval_count} "
                            f"reason={chunk.done_reason} chunks={chunks_captured}"
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

            except asyncio.CancelledError:
                # Client disconnected — abort Ollama generation immediately to free GPU
                logger.warning(f"[{request_id}] Client disconnected, aborting generation")
                if pusher_task and not pusher_task.done():
                    pusher_task.cancel()
                    try:
                        await pusher_task
                    except (asyncio.CancelledError, Exception):
                        pass
                raise  # propagate to outer handler
            except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout) as e:
                died_mid_stream = True
                logger.error(f"[{request_id}] Ollama connection error: {e}")
                break
            except Exception as e:
                logger.error(f"[{request_id}] Stream loop error: {e}")
                break
            else:
                # ── Stream exited normally WITHOUT chunk.done ──
                logger.warning(
                    f"[{request_id}] Stream loop exited | graceful={graceful} "
                    f"chunks_captured={chunks_captured} died_mid_stream={died_mid_stream} "
                    f"content_buf={len(''.join(batch_content))} thinking_buf={len(''.join(batch_thinking))} "
                    f"has_tool_calls={has_tool_calls} prompt_tokens={prompt_tokens} "
                    f"completion_tokens={completion_tokens}"
                )
                if not graceful and chunks_captured > 0:
                    logger.warning(
                        f"[{request_id}] Stream ended without finish_reason "
                        f"after {chunks_captured} chunks — yielding error to trigger retry"
                    )
                    # Flush any remaining buffered tokens so client has context
                    frame = _flush_batch()
                    if frame:
                        yield frame
                    # DO NOT send a fake done chunk — Pi agent will silently accept it
                    # as valid. Instead send an error so its retry logic kicks in.
                    yield build_sse_error_frame(
                        "Stream ended without finish_reason", "incomplete_stream"
                    )
            finally:
                # Cancel the pusher task if it's still running (e.g. on break/retry)
                if pusher_task and not pusher_task.done():
                    pusher_task.cancel()
                    try:
                        await pusher_task
                    except asyncio.CancelledError:
                        pass

                # Flush remaining thinking/content on ANY exit (cancelled, timeout, or normal)
                frame = _flush_batch()
                if frame:
                    yield frame

                # ── Zero-token detection: retry or graceful exit ──
                if not graceful and prompt_tokens == 0 and completion_tokens == 0:
                    if died_mid_stream:
                        # Ollama crashed mid-stream — this is not an empty response, it's a failure
                        logger.error(
                            f"[{request_id}] Ollama died mid-stream after {chunks_captured} chunks — retrying"
                        )
                        retry_count += 1
                        if retry_count <= MAX_RETRIES:
                            await asyncio.sleep(2)
                            continue  # retry
                        # retries exhausted
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
        except GeneratorExit:
            # FastAPI calls .aclose() on the generator when client disconnects
            logger.warning(f"[{request_id}] Generator closed (client disconnect)")
            if pusher_task and not pusher_task.done():
                pusher_task.cancel()
                try:
                    await pusher_task
                except (asyncio.CancelledError, Exception):
                    pass
            return
        except ollama.ResponseError as e:
            elapsed = round(time.monotonic() - start_time, 2)
            logger.error(f"[{request_id}] Ollama ResponseError {e.status_code}: {e.error}")
            yield build_sse_error_frame(str(e.error)[:100], "upstream_error")
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