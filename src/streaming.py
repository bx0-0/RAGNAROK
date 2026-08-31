"""Streaming SSE generator — extracted from server for readability."""

import time
import asyncio

import httpx
import ollama

from fastapi.responses import StreamingResponse

from src.logging import logger, log_request
from src.sse import (
    _SSE_DONE,
    _SSE_KEEPALIVE,
    make_sse_frames,
    build_done_chunk,
)
from src.errors import build_sse_error_frame
from src.models.chat import build_chat_kwargs
from src.retry import RetryPolicy
from src.batcher import StreamBatcher

# These are read from server at runtime; we avoid importing them to prevent circular deps.
# They're passed via config dict instead.

# Retry configuration is read once at import (server start) — consistent with the
# rest of the config in src/config.py. Tests monkeypatch this to shrink backoffs
# or change max_retries.
_RETRY = RetryPolicy.default()


# ── Parsing / normalization ──
# The Ollama→normalized-data layer lives in src/parser.py. The underscored
# names below are re-exported so existing importers (streaming internals,
# tests) keep working unchanged; prefer the parser.py names for new code.
from src.parser import ParsedChunk, format_tool_calls, parse_chunk

_ParsedChunk = ParsedChunk
_format_tool_calls = format_tool_calls
_parse_chunk = parse_chunk


_PUMP_CHUNK = "chunk"
_PUMP_KEEPALIVE = "keepalive"
_PUMP_ERROR = "error"
_PUMP_END = "end"

# Send a keepalive if no chunk arrives within this many seconds (long Ollama gap).
PUMP_CHUNK_TIMEOUT = 60


async def _pump_chunk_item(queue: "asyncio.Queue", request_id: str):
    """Fetch the next item from the Ollama chunk queue with a 60s gap timeout.

    Pure pump: owns the queue wait, the long-gap keepalive decision, and the
    end/error sentinels — returns a (kind, payload) tag so the caller (the
    orchestrator) can decide what to yield / raise. Kinds:

      _PUMP_KEEPALIVE  (payload=None)          -> caller yields _SSE_KEEPALIVE
      _PUMP_ERROR      (payload=Exception)     -> caller raises payload
      _PUMP_END        (payload=None)          -> caller breaks
      _PUMP_CHUNK      (payload=chunk object)  -> caller processes chunk

    One responsibility: getting the next chunk (or timeout/sentinel) out of the
    queue. No SSE formatting, no parsing, no retry, no pusher-task lifecycle —
    the caller owns all of those.
    """
    try:
        item = await asyncio.wait_for(queue.get(), timeout=PUMP_CHUNK_TIMEOUT)
    except asyncio.TimeoutError:
        logger.info(f"[{request_id}] Keepalive ping (Ollama gap > {PUMP_CHUNK_TIMEOUT}s)")
        return _PUMP_KEEPALIVE, None

    if isinstance(item, Exception):
        return _PUMP_ERROR, item
    if item is StopAsyncIteration:
        return _PUMP_END, None
    return _PUMP_CHUNK, item


def _finalize_frames(request_id_str: str, created: int, active_model: str,
                     has_tool_calls: bool, prompt_tokens: int, completion_tokens: int,
                     message: str, err_type: str) -> tuple:
    """Build the 3-frame terminal sequence (error + done + [DONE]) for an
    aborted stream. Pure: no I/O, no side effects. The caller (stream_generator)
    yields these in order and returns — a hard stop that does NOT fall through
    to the success path.

    Used by the timeout / ResponseError / generic-crash paths, which all share
    the identical "error frame, then usage, then [DONE], then exit" shape.
    """
    return (
        build_sse_error_frame(message, err_type),
        build_done_chunk(request_id_str, created, active_model,
                         has_tool_calls, prompt_tokens, completion_tokens),
        _SSE_DONE,
    )


async def stream_generator(state, request_id, ollama_payload, start_time,
                           request_id_str, created, active_model, max_stream_s,
                           sfx, efx):
    """Core async generator that yields SSE frames from Ollama's streaming API."""

    # StreamBatcher owns the "first frame gets role" invariant and the
    # tool_call_index counter (both persist across retries, as before); the
    # text/thinking buffer is reset at the start of each retry attempt.
    batcher = StreamBatcher(sfx, efx)
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

    while retry_count <= _RETRY.max_retries:
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

                _pusher = asyncio.create_task(_queue_pusher())
                pusher_task = _pusher  # save ref for disconnect handling

                while True:
                    kind, payload = await _pump_chunk_item(chunk_queue, request_id)
                    if kind == _PUMP_KEEPALIVE:
                        yield _SSE_KEEPALIVE
                        continue
                    if kind == _PUMP_ERROR:
                        raise payload
                    if kind == _PUMP_END:
                        break
                    chunk = payload

                    # ── Hard Timeout ──
                    stream_elapsed = time.monotonic() - start_time
                    if stream_elapsed > max_stream_s:
                        logger.warning(f"[{request_id}] Hard timeout after {int(stream_elapsed)}s")
                        frame = _flush_batch()
                        if frame:
                            yield frame
                        for f in _finalize_frames(
                            request_id_str, created, active_model,
                            has_tool_calls, prompt_tokens, completion_tokens,
                            f"Generation exceeded {max_stream_s}s limit", "timeout",
                        ):
                            yield f
                        return

                    chunks_captured += 1

                    if chunk.message is None:
                        # Empty final chunk — no payload to process
                        continue

                    parsed = _parse_chunk(chunk)

                    # ── Accumulate text/thinking into the batcher ──
                    batcher.add(parsed)

                    should_flush = False

                    # ── Tool calls force immediate emission ──
                    if parsed.tool_calls:
                        has_tool_calls = True
                        # batcher.emit() folds any buffered text/thinking into
                        # the same frame and clears the buffer — one atomic
                        # delta carries both, no separate flush needed.
                        formatted = _format_tool_calls(parsed.tool_calls, batcher.tool_call_index)
                        batcher.tool_call_index += len(formatted)
                        frame = batcher.emit({"tool_calls": formatted})
                        if frame is not None:
                            try:
                                yield frame
                            except Exception as ex:
                                logger.error(f"[{request_id}] Tool call emit failed: {ex}")

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
                        if _RETRY.should_retry_crashed(retry_count):
                            await asyncio.sleep(_RETRY.next_delay("crashed", retry_count))
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
                            f"[{request_id}] Empty stream (attempt {retry_count}/{_RETRY.max_retries})"
                        )
                        if _RETRY.should_retry_empty(retry_count):
                            yield _SSE_KEEPALIVE
                            await asyncio.sleep(_RETRY.next_delay("empty", retry_count))
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
            logger.warning(f"[{request_id}] CLIENT_DISCONNECTED — cancelling pusher")
            if pusher_task and not pusher_task.done():
                pusher_task.cancel()
                try:
                    await pusher_task
                except (asyncio.CancelledError, Exception):
                    pass
            raise
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
            for f in _finalize_frames(
                request_id_str, created, active_model,
                has_tool_calls, prompt_tokens, completion_tokens,
                str(e.error)[:100], "upstream_error",
            ):
                yield f
            return
        except Exception as e:
            stream_error = f"CRASH {type(e).__name__}: {str(e)[:60]}"
            logger.error(f"[{request_id}] STREAM CRASH: {e}")
            for f in _finalize_frames(
                request_id_str, created, active_model,
                has_tool_calls, prompt_tokens, completion_tokens,
                "Internal server error", "server_error",
            ):
                yield f
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

            


# ─── Disconnect-aware response ───

class DisconnectAwareStreamingResponse(StreamingResponse):
    """StreamingResponse that detects client disconnect via ASGI receive().

    On spec_version >= 2.4 (uvicorn+httptools), Starlette does NOT spawn a
    disconnect watcher — it just calls stream_response(send) directly.
    This class adds the watcher back: a concurrent receive() loop that
    detects http.disconnect and cancels the streaming task, triggering the
    generator's CancelledError path (which cancels the Ollama pusher).
    """

    def __init__(self, body_iterator, *, state=None, request_id=None, model=None,
                 media_type=None, status_code=200, headers=None, background=None):
        super().__init__(
            content=body_iterator,
            status_code=status_code,
            media_type=media_type,
            headers=headers,
            background=background,
        )
        self._state = state
        self._request_id = request_id
        self._model = model

    async def __call__(self, scope, receive, send):
        async def _stream():
            await self.stream_response(send)

        async def _watch():
            while True:
                msg = await receive()
                if msg["type"] == "http.disconnect":
                    rid = self._request_id or "?"
                    logger.info(f"[{rid}] CLIENT_DISCONNECT_DETECTED")
                    return

        stream_task = asyncio.create_task(_stream())
        watch_task = asyncio.create_task(_watch())

        # Register active stream for stop/unload endpoints
        if self._state and self._request_id:
            self._state.register_stream(self._request_id, self._model, stream_task)

        try:
            done, pending = await asyncio.wait(
                {stream_task, watch_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        except BaseException:
            stream_task.cancel()
            watch_task.cancel()
            raise

        # Cancel the remaining task
        for t in pending:
            t.cancel()
            try:
                await t
            except BaseException:
                pass

        # Unregister (idempotent)
        if self._state and self._request_id:
            self._state.unregister_stream(self._request_id)

        # Propagate any exception from completed tasks
        for t in done:
            if not t.cancelled() and t.exception() is not None:
                raise t.exception()

        if self.background is not None:
            await self.background()


def handle_stream(state, request_id, ollama_payload, start_time, active_model,
                  max_stream_seconds: int):
    """Entry point called from server route handler."""
    request_id_str = f"chatcmpl-{request_id}"
    created = int(time.time())
    sfx, efx = make_sse_frames(active_model, request_id_str, created)

    gen = stream_generator(state, request_id, ollama_payload, start_time,
                           request_id_str, created, active_model,
                           max_stream_seconds, sfx, efx)

    return DisconnectAwareStreamingResponse(
        gen,
        state=state,
        request_id=request_id,
        model=active_model,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "x-request-id": request_id,
        },
    )