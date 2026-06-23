"""Tests for SSE envelope construction — pure functions, no I/O."""

import orjson

from src.sse import (
    _SSE_DONE,
    _SSE_KEEPALIVE,
    _sse_template_for_model,
    make_sse_frames,
    build_done_chunk,
)


class TestStaticSSE:
    def test_done_bytes(self):
        assert _SSE_DONE == b"data: [DONE]\n\n"

    def test_keepalive_is_valid_sse(self):
        assert _SSE_KEEPALIVE.startswith(b"data: ")
        assert _SSE_KEEPALIVE.endswith(b"\n\n")
        data = orjson.loads(_SSE_KEEPALIVE[len(b"data: "):-2])
        assert data["choices"][0]["finish_reason"] is None

    def test_template_cached(self):
        r1 = _sse_template_for_model("qwen3:8b")
        r2 = _sse_template_for_model("qwen3:8b")
        assert r1 is r2

    def test_template_different_models(self):
        t1 = _sse_template_for_model("model-a")
        t2 = _sse_template_for_model("model-b")
        # Model name baked into prefix
        assert b"model-a" in t1[0]
        assert b"model-b" in t2[0]


class TestMakeSSEFrames:
    def test_injects_id_and_created(self):
        sfx, efx = make_sse_frames("qwen3:8b", "chatcmpl-abc123", 1700000000)
        assert b"chatcmpl-abc123" in sfx
        assert b'"created":1700000000' in sfx

    def test_roundtrip_valid_json(self):
        """prefix + delta json + suffix must produce valid SSE line."""
        sfx, efx = make_sse_frames("m", "id42", 99)
        delta = orjson.dumps({"role": "assistant", "content": "hello"})
        frame = sfx + delta + efx
        # Should end with double newline
        assert frame.endswith(b"\n\n")
        line = frame[len(b"data: "):-2]
        obj = orjson.loads(line)
        assert obj["id"] == "id42"
        assert obj["created"] == 99
        assert obj["model"] == "m"
        assert obj["object"] == "chat.completion.chunk"
        choices = obj["choices"]
        assert len(choices) == 1
        assert choices[0]["delta"]["role"] == "assistant"
        assert choices[0]["delta"]["content"] == "hello"
        assert choices[0]["index"] == 0


class TestBuildDoneChunk:
    def test_stop_finish(self):
        chunk = build_done_chunk("rid", 100, "m", False, 50, 20)
        obj = orjson.loads(chunk[len(b"data: "):-2])
        assert obj["choices"][0]["finish_reason"] == "stop"
        assert obj["usage"]["prompt_tokens"] == 50
        assert obj["usage"]["completion_tokens"] == 20
        assert obj["usage"]["total_tokens"] == 70

    def test_tool_calls_finish(self):
        chunk = build_done_chunk("rid", 100, "m", True, 30, 10)
        obj = orjson.loads(chunk[len(b"data: "):-2])
        assert obj["choices"][0]["finish_reason"] == "tool_calls"
        assert obj["id"] == "rid"
        assert obj["model"] == "m"

    def test_zero_tokens(self):
        chunk = build_done_chunk("rid", 1, "x", False, 0, 0)
        obj = orjson.loads(chunk[len(b"data: "):-2])
        assert obj["usage"]["total_tokens"] == 0
