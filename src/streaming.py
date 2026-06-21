"""Streaming SSE generator — extracted from server for readability."""

import os
import time
import asyncio
import traceback

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
from src.config import OLLAMA_STREAM_LOG
from src.errors import build_sse_error_frame

_ollama_stream_fh = None


def _open_ollama_stream_log():
    global _ollama_stream_fh
    try:
        _ollama_stream_fh = open(OLLAMA_STREAM_LOG, "a", buffering=1)
    except Exception as e:
        logger.warning(f"Could not open {OLLAMA_STREAM_LOG}: {e}")


def _close_ollama_stream_log():
    global _ollama_stream_fh
    if _ollama_stream_fh:
        try:
            _ollama_stream_fh.flush()
            _ollama_stream_fh.close()
        except Exception:
            pass
        _ollama_stream_fh = None


def _log_chunk(request_id, chunk_num, msg, done, prompt_eval, eval_count, done_reason, total_duration):
    """Append a single chunk dump to the ollama-stream.log file."""
    if _ollama_stream_fh is None:
        return
    try:
        entry = {
            "request_id": request_id,
            "chunk_num": chunk_num,
            "done": done,
            "prompt_eval_count": prompt_eval,
            "eval_count": eval_count,
            "done_reason": done_reason,
            "total_duration_ms": round(total_duration / 1_000_000, 2) if total_duration else None,
        }
        if msg:
            entry["content"] = msg.content if hasattr(msg, "content") else None
            entry["thinking"] = getattr(msg, "thinking", None)
            entry["tool_calls"] = [
                {
                    "id": tc.id if hasattr(tc, "id") else None,
                    "name": tc.function.name if hasattr(tc, "function") else None,
                    "args_preview": (
                        str(tc.function.arguments)[:200] if hasattr(tc, "function") else None
                    ),
                }
                for tc in (msg.tool_calls or [])
            ]
        _ollama_stream_fh.write(orjson.dumps(entry).decode() + "\n")
        _ollama_stream_fh.flush()
    except Exception:
        pass


def _log_stream_event(request_id, event, **kwargs):
    """Log a stream lifecycle event (retry, error, finally, etc.) to the file."""
    if _ollama_stream_fh is None:
        return
    try:
        entry = {"request_id": request_id, "event": event, **kwargs}
        _ollama_stream_fh.write(orjson.dumps(entry).decode() + "\n")
        _ollama_stream_fh.flush()
    except Exception:
        pass

# These are read from server at runtime; we avoid importing them to prevent circular deps.
# They're passed via config dict instead.

MAX_RETRIES = 2


def _should_retry_empty() -> bool:
    return os.environ.get("RETRY_ON_EMPTY", "False").lower() in ("true", "1", "yes")


async def stream_generator(state, request_id, ollama_payload, start_time,
                           request_id_str, created, active_model, max_stream_s,
                           ollama_chat_url, sfx, efx):
    """Core async generator that yields SSE frames from Ollama's streaming API."""

    _open_ollama_stream_log()
    _log_stream_event(request_id, "START", model=active_model, max_stream_s=max_stream_s)

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
    if ollama_payload.get("tool_choice"):
        chat_kwargs["tool_choice"] = ollama_payload["tool_choice"]

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
                # ── Stream using asyncio.wait() so we can inject keepalives during Ollama gaps ──
                _ai = (await state.http_client.chat(**chat_kwargs)).__aiter__()
                CHUNK_TIMEOUT = 10  # send keepalive if no chunk arrives in 10s

                while True:
                    # Wait for next chunk with a timeout so we can detect long gaps
                    done, _pending = await asyncio.wait(
                        [_ai.__anext__()],
                        timeout=CHUNK_TIMEOUT,
                    )
                    if not done:
                        # No chunk arrived within CHUNK_TIMEOUT seconds — send keepalive
                        logger.info(f"[{request_id}] Keepalive ping (Ollama gap > {CHUNK_TIMEOUT}s)")
                        yield _SSE_KEEPALIVE
                        continue

                    # Got a chunk
                    try:
                        chunk = done.pop().result()
                    except StopAsyncIteration:
                        # No more chunks from Ollama
                        break

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
                    _log_chunk(request_id, chunks_captured, msg, chunk.done,
                              chunk.prompt_eval_count, chunk.eval_count,
                              chunk.done_reason, chunk.total_duration)
                    _c = (msg.content or '')[:30]
                    _t = (getattr(msg, 'thinking', '') or '')[:20]
                    raw_preview = f"content={repr(_c)} thinking={repr(_t)} done={chunk.done}"
                    logger.info(f"[{request_id}] CHUNK#{chunks_captured}: {raw_preview}")
                    logger.info(f"[{request_id}] CHUNK#{chunks_captured}: {raw_preview}")

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
                            args_len = len(tc_args_json)
                            logger.info(
                                f"[{request_id}] Tool [{tc_name}] args_len={args_len}"
                            )
                            if is_write_tool:
                                logger.info(f"[{request_id}] WRITE args preview: {tc_args_json[:500]}")
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

                    # ── Time-based flush ──
                    if (time.monotonic() - batch_timer) > 0.1:
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
                        logger.info(
                            f"[{request_id}] DONE chunk | prompt_eval_count={chunk.prompt_eval_count} "
                            f"eval_count={chunk.eval_count} done_reason={chunk.done_reason} "
                            f"total_duration={chunk.total_duration} load_duration={chunk.load_duration} "
                            f"chunks_received={chunks_captured}"
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
                raise  # will be caught by outer handler                        break

            except asyncio.CancelledError:
                raise  # will be caught by outer handler
            except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout) as e:
                died_mid_stream = True
                logger.error(f"[{request_id}] Ollama connection error: {e}")
                break
            except Exception as e:
                logger.error(f"[{request_id}] Stream loop error: {e}")
                logger.error(f"[{request_id}] Trace: {traceback.format_exc()[:500]}")
                break
            else:
                _log_stream_event(
                    request_id, "EXIT_NO_DONE", graceful=graceful, chunks=chunks_captured,
                    died_mid=died_mid_stream, P=prompt_tokens, C=completion_tokens,
                    has_tool_calls=has_tool_calls,
                    content_buf=len("".join(batch_content)),
                    thinking_buf=len("".join(batch_thinking)),
                )
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
                _log_stream_event(
                    request_id, "FINALLY", graceful=graceful, chunks=chunks_captured,
                    died_mid=died_mid_stream, P=prompt_tokens, C=completion_tokens,
                    has_tool_calls=has_tool_calls,
                )
                logger.info(
                    f"[{request_id}] FINALLY | graceful={graceful} chunks={chunks_captured} "
                    f"died_mid={died_mid_stream} P={prompt_tokens} C={completion_tokens} "
                    f"has_tool_calls={has_tool_calls}"
                )
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
                            _log_stream_event(
                                request_id, "RETRY", attempt=retry_count, chunks=chunks_captured,
                                died_mid=died_mid_stream, P=prompt_tokens, C=completion_tokens,
                            )
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

            _log_stream_event(
                request_id, "DONE_SUCCESS", retry=retry_count, chunks=chunks_captured,
                P=prompt_tokens, C=completion_tokens,
            )
            yield _SSE_DONE
            break  # success — exit retry loop

        except asyncio.CancelledError:
            logger.warning(f"[{request_id}] Stream cancelled (client disconnected)")
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
            logger.error(f"[{request_id}] Trace: {traceback.format_exc()[:500]}")
            yield build_sse_error_frame("Internal server error", "server_error")
            yield build_done_chunk(
                request_id_str, created, active_model,
                has_tool_calls, prompt_tokens, completion_tokens,
            )
            yield _SSE_DONE
            return
        finally:
            _log_stream_event(request_id, "OUTER_FINALLY_PRE", P=prompt_tokens, C=completion_tokens)
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
            _log_stream_event(request_id, "END", P=prompt_tokens, C=completion_tokens)
            _close_ollama_stream_log()


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