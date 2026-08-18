"""Focused tests for the Step 10-extracted helpers in src/streaming:

* ``_pump_chunk_item`` — the queue pump: 60s gap keepalive, error/end/chunk
  tagging. (Owns queue wait + long-gap keepalive decision.)
* ``_finalize_frames`` — the terminal error/done/[DONE] sequence. (Owns
  final/error/done emission shape.)

These cover code the existing golden/retry tests don't directly hit (the
pump-timeout branch and the aborted-stream terminal shape).
"""
import asyncio

import pytest

import src.streaming as st
from src.sse import _SSE_DONE


# ── _pump_chunk_item ─────────────────────────────────────────────────────

class TestPumpChunkItem:

    def test_returns_chunk_for_message_item(self):
        q = asyncio.Queue()
        item = object()
        q.put_nowait(item)
        kind, payload = asyncio.new_event_loop().run_until_complete(
            st._pump_chunk_item(q, "req"))
        assert kind == st._PUMP_CHUNK
        assert payload is item

    def test_returns_end_for_stop_sentinel(self):
        q = asyncio.Queue()
        q.put_nowait(StopAsyncIteration)
        kind, payload = asyncio.new_event_loop().run_until_complete(
            st._pump_chunk_item(q, "req"))
        assert kind == st._PUMP_END
        assert payload is None

    def test_returns_error_for_exception_item(self):
        q = asyncio.Queue()
        err = RuntimeError("boom")
        q.put_nowait(err)
        kind, payload = asyncio.new_event_loop().run_until_complete(
            st._pump_chunk_item(q, "req"))
        assert kind == st._PUMP_ERROR
        assert payload is err

    def test_returns_keepalive_on_timeout(self, monkeypatch):
        # Shrink the gap timeout so the test is instant: an empty queue + a
        # sub-second timeout forces the keepalive branch deterministically.
        monkeypatch.setattr(st, "PUMP_CHUNK_TIMEOUT", 0.001)

        async def run():
            return await st._pump_chunk_item(asyncio.Queue(), "req")

        kind, payload = asyncio.new_event_loop().run_until_complete(run())
        assert kind == st._PUMP_KEEPALIVE
        assert payload is None


# ── _finalize_frames ─────────────────────────────────────────────────────

class TestFinalizeFrames:

    def test_terminal_sequence_shape(self):
        frames = st._finalize_frames(
            "chatcmpl-req", 0, "m",
            has_tool_calls=True, prompt_tokens=5, completion_tokens=9,
            message="Generation exceeded 300s limit", err_type="timeout",
        )
        assert len(frames) == 3
        error, done, sentinel = frames
        assert sentinel == _SSE_DONE

        # error frame carries our message + type
        import orjson
        e = orjson.loads(error[len(b"data: "):])
        assert e["error"]["message"] == "Generation exceeded 300s limit"
        assert e["error"]["type"] == "timeout"

        # done chunk carries usage + finish_reason=tool_calls (because has_tool_calls)
        d = orjson.loads(done[len(b"data: "):])
        assert d["choices"][0]["finish_reason"] == "tool_calls"
        assert d["usage"]["prompt_tokens"] == 5
        assert d["usage"]["completion_tokens"] == 9
        assert d["usage"]["total_tokens"] == 14

    def test_no_tool_calls_finish_reason_stop(self):
        import orjson
        _, done, _ = st._finalize_frames(
            "chatcmpl-req", 0, "m",
            has_tool_calls=False, prompt_tokens=1, completion_tokens=2,
            message="Internal server error", err_type="server_error",
        )
        d = orjson.loads(done[len(b"data: "):])
        assert d["choices"][0]["finish_reason"] == "stop"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
