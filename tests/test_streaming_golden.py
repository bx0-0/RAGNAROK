"""Golden-path tests: drive the REAL stream_generator / non-stream handler
end-to-end with a fake Ollama client (no server, no GPU).

These are the regression net that would have caught the "Tool ? not found"
double-format bug: they assert the exact SSE byte stream and the exact
non-stream JSON the client receives, including tool-call name, arguments,
and id round-trip.
"""
import asyncio
import time
from types import SimpleNamespace as NS

import orjson
import pytest

from src.streaming import stream_generator
from src.routes.chat import _handle_non_stream
from src.models.chat import build_chat_kwargs
from src.sse import make_sse_frames


# ── Fakes ────────────────────────────────────────────────────────────────

def _raw_tool_call(name="read", args=None, cid="call-1"):
    args = args if args is not None else {"path": "/tmp/x"}
    # Real ollama ToolCall objects are BOTH attribute-accessible AND subscriptable
    # (SubscriptableBaseModel). The non-stream path (format_tool_calls_openai) uses
    # tc.get("function") / tc["function"]["name"], so the fake must support .get too.
    # `function` must be attribute-accessible (streaming: tc.function.name) AND
    # its name/arguments must also be reachable (non-stream: tc["function"]["name"]).
    func = NS(name=name, arguments=args)

    class _TC:
        def __init__(self):
            self.function = func
            self.id = cid
        def get(self, k, default=None):
            return {"function": {"name": name, "arguments": args}, "id": cid}.get(k, default)
        def __getitem__(self, k):
            return {"function": {"name": name, "arguments": args}, "id": cid}[k]
        def __contains__(self, k):
            return k in {"function", "id"}
    return _TC()


def _raw_chunk(content=None, tool_calls=None, done=False,
               prompt=0, evalc=0, reason="stop"):
    return NS(
        message=NS(content=content, thinking="", tool_calls=tool_calls),
        done=done, prompt_eval_count=prompt, eval_count=evalc,
        done_reason=reason,
    )


class FakeClient:
    """Mimics ollama.AsyncClient.chat() returning an async generator."""

    def __init__(self, chunks):
        self._chunks = chunks

    async def chat(self, **kwargs):
        async def _gen():
            for c in self._chunks:
                yield c
        return _gen()


class FakeState:
    def __init__(self, chunks):
        self.http_client = FakeClient(chunks)
        self.semaphore = asyncio.Semaphore(2)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _stream_raw(chunks):
    """Run the real stream_generator over *chunks*; return raw SSE bytes."""
    state = FakeState(chunks)
    sfx, efx = make_sse_frames("m", "chatcmpl-req", 0)

    async def run():
        gen = stream_generator(
            state, "req", {"model": "m", "messages": []},
            time.monotonic(), "chatcmpl-req", 0, "m", 300,
            sfx, efx,
        )
        return b"".join([f async for f in gen])

    return _run(run())


def _frames(raw: bytes) -> list:
    """Parse an SSE byte stream into a list of JSON payloads
    (with a "[DONE]" marker for the sentinel)."""
    out = []
    for line in raw.split(b"\n\n"):
        line = line.strip()
        if not line.startswith(b"data: "):
            continue
        body = line[len(b"data: "):]
        if body == b"[DONE]":
            out.append("[DONE]")
        else:
            out.append(orjson.loads(body))
    return out


def _tool_frames(frames) -> list:
    return [
        tc
        for f in frames
        if isinstance(f, dict)
        for tc in (f.get("choices", [{}])[0].get("delta", {}).get("tool_calls") or [])
    ]


def _delta(f, key):
    return f.get("choices", [{}])[0].get("delta", {}).get(key)


def _finish(frame):
    return frame.get("choices", [{}])[0].get("finish_reason") if isinstance(frame, dict) else None


# ── stream: true (SSE generator) ─────────────────────────────────────────

class TestStreamTrueGolden:

    def test_tool_call_roundtrip(self):
        raw = _stream_raw([
            _raw_chunk(content="Let me read that file."),
            _raw_chunk(tool_calls=[_raw_tool_call("read", {"path": "/tmp/x"})],
                       done=True, prompt=11, evalc=22),
        ])
        frames = _frames(raw)

        tools = _tool_frames(frames)
        assert len(tools) == 1, f"expected 1 tool call, got: {tools}"
        tc = tools[0]
        assert tc["function"]["name"] == "read"
        assert tc["function"]["arguments"] == '{"path":"/tmp/x"}'
        assert tc["id"] == "call-1"
        assert tc["type"] == "function"
        assert tc["index"] == 0

        # the tool-call frame must also carry the buffered text (emitted
        # atomically by batcher.emit), OR a preceding text frame — one of the
        # two, with "Let me read that file." present exactly once across frames
        text_occurrences = [fr for fr in frames if isinstance(fr, dict)
                            and _delta(fr, "content") == "Let me read that file."]
        assert len(text_occurrences) == 1

        finished = [f for f in frames if _finish(f) in ("tool_calls", "stop")]
        assert finished and _finish(finished[-1]) == "tool_calls"
        assert frames[-1] == "[DONE]"

    def test_multiple_tool_calls_keep_index_and_ids(self):
        raw = _stream_raw([
            _raw_chunk(tool_calls=[
                _raw_tool_call("read", {"path": "a"}, cid="c1"),
                _raw_tool_call("bash", {"cmd": "ls"}, cid="c2"),
            ], done=True, prompt=3, evalc=7),
        ])
        tools = _tool_frames(_frames(raw))
        assert len(tools) == 2
        assert [t["function"]["name"] for t in tools] == ["read", "bash"]
        assert [t["index"] for t in tools] == [0, 1]
        assert [t["id"] for t in tools] == ["c1", "c2"]
        assert tools[1]["function"]["arguments"] == '{"cmd":"ls"}'

    def test_text_only_stream(self):
        raw = _stream_raw([
            _raw_chunk(content="Hel"),
            _raw_chunk(content="lo", done=True, prompt=5, evalc=2),
        ])
        frames = _frames(raw)
        text = "".join(
            _delta(f, "content") or "" for f in frames if isinstance(f, dict)
        )
        assert text == "Hello"
        assert all("tool_calls" not in (f.get("choices", [{}])[0].get("delta", {}))
                   for f in frames if isinstance(f, dict))
        finished = [f for f in frames if _finish(f) in ("tool_calls", "stop")]
        assert finished and _finish(finished[-1]) == "stop"
        assert frames[-1] == "[DONE]"


# ── stream: false (JSON handler) ─────────────────────────────────────────

class TestStreamFalseGolden:

    def test_tool_call_roundtrip(self):
        ollama_response = NS(
            message=NS(content=None, thinking=None,
                       tool_calls=[_raw_tool_call("read", {"path": "/tmp/x"})]),
            prompt_eval_count=11, eval_count=22,
        )

        async def fake_chat(**kwargs):
            return ollama_response

        state = NS(http_client=NS(chat=fake_chat), semaphore=asyncio.Semaphore(2))
        payload = {"model": "m", "messages": []}

        resp = _run(_handle_non_stream(state, "req", payload, 0.0, 123, "m"))
        body = orjson.loads(resp.body)

        assert body["choices"][0]["finish_reason"] == "tool_calls"
        tc = body["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "read"
        assert tc["function"]["arguments"] == '{"path":"/tmp/x"}'
        assert tc["id"] == "call-1"
        assert tc["index"] == 0

    def test_text_only(self):
        ollama_response = NS(
            message=NS(content="plain answer", thinking=None, tool_calls=None),
            prompt_eval_count=4, eval_count=9,
        )

        async def fake_chat(**kwargs):
            return ollama_response

        state = NS(http_client=NS(chat=fake_chat), semaphore=asyncio.Semaphore(2))
        payload = {"model": "m", "messages": []}

        resp = _run(_handle_non_stream(state, "req", payload, 0.0, 123, "m"))
        body = orjson.loads(resp.body)
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["choices"][0]["message"]["content"] == "plain answer"
        assert "tool_calls" not in body["choices"][0]["message"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
