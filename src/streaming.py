"""Streaming SSE generator — extracted from server for readability."""

import os
import time
import asyncio
from dataclasses import dataclass

import orjson
import httpx
import ollama

from fastapi.responses import StreamingResponse

from src.logging import logger, log_request
from src.utils.helpers import fast_id
from src.sse import (
    _SSE_DONE,
    _SSE_KEEPALIVE,
    make_sse_frames,
    build_done_chunk,
)
from src.errors import build_sse_error_frame
from src.models.chat import build_chat_kwargs

# These are read from server at runtime; we avoid importing them to prevent circular deps.
# They're passed via config dict instead.

MAX_RETRIES = 2


def _should_retry_empty() -> bool:
    return os.environ.get("RETRY_ON_EMPTY", "False").lower() in ("true", "1", "yes")


@dataclass
class _ParsedChunk:
    """Normalized, serializable view of one Ollama streaming chunk.

    Built by `_parse_chunk` from the raw ollama object so all the messy
    extraction (content shape, thinking, tool-call formatting) lives in one
    pure, unit-testable place. `tool_calls` is a list only when the chunk
    carries them (else empty); `content`/`thinking` are always str.
    """
    content: str = ""
    thinking: str = ""
    tool_calls: list = None  # populated (non-empty) when the chunk has tool calls
    done: bool = False
    prompt_eval_count: int = 0
    eval_count: int = 0
    done_reason: str = ""


def _format_tool_calls(tool_calls, tool_call_index: int) -> list:
    """Convert Ollama tool_calls objects to the OpenAI delta shape.

    `tool_call_index` is the next index to assign; the return value is a
    2-tuple is avoided (to keep it trivially testable) — instead the list of
    formatted dicts is returned and the caller increments its own counter by
    `len(formatted)`.
    """
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
        formatted.append({
            "index": tool_call_index + len(formatted),
            "id": getattr(tc, "id", None) or f"call_{fast_id()}",
            "type": "function",
            "function": {
                "name": tc_name,
                "arguments": tc_args_json,
            },
        })
    return formatted


def _parse_chunk(chunk) -> _ParsedChunk:
    """Extract the fields we care about from one Ollama chunk.

    Pure: reads attributes off the chunk, no I/O, no server, no mutation.
    Handles the `content` shape variants (str or list-of-parts) and the
    `thinking` field uniformly so the main loop stays simple.
    """
    msg = chunk.message
    if msg is None:
        # Ollama's empty final chunk — no payload
        return _ParsedChunk()

    content = msg.content or ""
    if isinstance(content, list):
        content = " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )

    thinking = getattr(msg, "thinking", "") or ""

    tool_calls = msg.tool_calls
    formatted = []
    if tool_calls:
        formatted = _format_tool_calls(tool_calls, 0)

    return _ParsedChunk(
        content=content,
        thinking=thinking,
        tool_calls=formatted,
        done=bool(chunk.done),
        prompt_eval_count=chunk.prompt_eval_count or 0,
        eval_count=chunk.eval_count or 0,
        done_reason=chunk.done_reason or "",
    )


class _StreamBatcher:
    """Accumulates reasoning/content deltas and emits batched SSE frames.

    Owns the "first chunk gets role" invariant and the batching buffer so the
    main loop never has to juggle those locals itself. `tool_call_index` is
    owned here too — the next index to assign to a new tool call.
    """

    def __init__(self, sfx: bytes, efx: bytes):
        self._sfx = sfx
        self._efx = efx
        self._first = True
        self._content: list = []
        self._thinking: list = []
        self.tool_call_index = 0

    # -- introspection ------------------------------------------------------
    def _delta(self, extra: dict | None = None) -> dict:
        delta = {}
        if self._first:
            delta["role"] = "assistant"
            self._first = False
        if self._thinking:
            delta["reasoning_content"] = "".join(self._thinking)
        if self._content:
            delta["content"] = "".join(self._content)
        if extra:
            delta.update(extra)
        return delta

    def dirty(self) -> bool:
        """True if there is unflushed reasoning/content buffered."""
        return bool(self._content or self._thinking)

    def content_len(self) -> int:
        return len("".join(self._content))

    def thinking_len(self) -> int:
        return len("".join(self._thinking))

    def reset(self) -> None:
        """Clear buffered text/thinking (keeps the first-frame flag and the
        tool_call_index). Called at the start of each retry attempt."""
        self._content.clear()
        self._thinking.clear()

    # -- emission -----------------------------------------------------------
    def flush(self) -> bytes | None:
        """Emit a buffered delta frame and clear the buffer. None if empty."""
        if not self._content and not self._thinking:
            return None
        delta = self._delta()
        self._content.clear()
        self._thinking.clear()
        return self._sfx + orjson.dumps(delta) + self._efx

    def emit(self, extra: dict) -> bytes | None:
        """Emit a delta frame carrying *extra* keys (e.g. tool_calls).

        Any buffered text/thinking is folded into the same frame first, then
        the buffer is cleared — a tool call and its preceding text ship together.
        """
        if not self._content and not self._thinking and not extra:
            return None
        delta = self._delta(extra)
        self._content.clear()
        self._thinking.clear()
        return self._sfx + orjson.dumps(delta) + self._efx

    def add(self, parsed: _ParsedChunk) -> None:
        """Accumulate a parsed chunk's text/thinking into the buffer.

        Tool calls are NOT accumulated — they force an immediate flush via
        `emit`; the caller handles those separately.
        """
        if parsed.content:
            self._content.append(parsed.content)
        if parsed.thinking:
            self._thinking.append(parsed.thinking)


async def stream_generator(state, request_id, ollama_payload, start_time,
                           request_id_str, created, active_model, max_stream_s,
                           ollama_chat_url, sfx, efx):
    """Core async generator that yields SSE frames from Ollama's streaming API."""

    # _StreamBatcher owns the "first frame gets role" invariant and the
    # tool_call_index counter (both persist across retries, as before); the
    # text/thinking buffer is reset at the start of each retry attempt.
    batcher = _StreamBatcher(sfx, efx)
    has_tool_calls = False
    prompt_tokens = completion_tokens = 0
    released = False
    retry_count = 0
    stream_error = None  # tracked for the live request log

    # ── Shared kwargs builder (single source of truth for chat kwargs) ──
    chat_kwargs = build_chat_kwargs(ollama_payload, stream=True)

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
            batcher.reset()  # clear any buffer left from a previous retry
            batch_timer = time.monotonic()
            chunks_captured = 0
            died_mid_stream = False
            graceful = False

            def _flush_batch():
                nonlocal batch_timer
                frame = batcher.flush()
                if frame is not None:
                    batch_timer = time.monotonic()
                return frame

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

                    if chunk.message is None:
                        # Empty final chunk — no payload to process
                        continue

                    parsed = _parse_chunk(chunk)

                    # ── Accumulate text/thinking into the batcher ──
                    batcher.add(parsed)

                    should_flush = False

                    # ── Tool calls force immediate flush ──
                    if parsed.tool_calls:
                        has_tool_calls = True
                        # Flush buffered text first, then emit the tool-call delta.
                        frame = _flush_batch()
                        if frame:
                            yield frame
                        formatted = _format_tool_calls(parsed.tool_calls, batcher.tool_call_index)
                        batcher.tool_call_index += len(formatted)
                        try:
                            yield sfx + orjson.dumps(batcher._delta({"tool_calls": formatted})) + efx
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
                stream_error = f"{type(e).__name__}: {str(e)[:60]}"
                logger.error(f"[{request_id}] Ollama connection error: {e}")
                break
            except Exception as e:
                stream_error = f"{type(e).__name__}: {str(e)[:60]}"
                logger.error(f"[{request_id}] Stream loop error: {e}")
                break
            else:
                # ── Stream exited normally WITHOUT chunk.done ──
                logger.warning(
                    f"[{request_id}] Stream loop exited | graceful={graceful} "
                    f"chunks_captured={chunks_captured} died_mid_stream={died_mid_stream} "
                    f"content_buf={batcher.content_len()} thinking_buf={batcher.thinking_len()} "
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
                            stream_error = None  # reset — retry may succeed
                            continue  # retry
                        # retries exhausted
                        stream_error = stream_error or f"model crashed mid-generation after {retry_count} attempts"
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
                            stream_error = None  # reset — retry may succeed
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
            stream_error = f"Ollama {e.status_code}: {str(e.error)[:60]}"
            logger.error(f"[{request_id}] Ollama ResponseError {e.status_code}: {e.error}")
            yield build_sse_error_frame(str(e.error)[:100], "upstream_error")
            yield build_done_chunk(
                request_id_str, created, active_model,
                has_tool_calls, prompt_tokens, completion_tokens,
            )
            yield _SSE_DONE
            return
        except Exception as e:
            stream_error = f"CRASH {type(e).__name__}: {str(e)[:60]}"
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
                    if stream_error:
                        # Make Ollama/upstream errors visible in the live request log
                        await log_request(request_id, "POST", "/v1/chat/completions", 500,
                                          elapsed, prompt_tokens, completion_tokens,
                                          f"ERR:{stream_error[:40]}")
                        logger.warning(f"[{request_id}] Done {elapsed}s ERROR | {stream_error}")
                    else:
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