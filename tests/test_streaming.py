"""Unit tests for the extracted streaming helpers.

These cover `_parse_chunk` and `StreamBatcher` — the two pieces that were
pulled out of `stream_generator` so they can be tested in isolation (no
server, no I/O). They do NOT drive the full generator; that is exercised
end-to-end in test_tool_calls.py / test_live.py.
"""
from types import SimpleNamespace as NS

from src.streaming import (
    _ParsedChunk,
    _parse_chunk,
    _format_tool_calls,
)
from src.batcher import StreamBatcher


# ── tiny fakes mirroring the ollama chunk shape ──────────────────────────
def _tool_call(name="read", args='{"path":"x"}', cid=None, index_hint=None):
    return NS(function=NS(name=name, arguments=args), id=cid)


def _chunk(content=None, thinking="", tool_calls=None, done=False,
           prompt=0, evalc=0, reason=""):
    return NS(
        message=NS(content=content, thinking=thinking, tool_calls=tool_calls),
        done=done,
        prompt_eval_count=prompt,
        eval_count=evalc,
        done_reason=reason,
    )


# ── _parse_chunk ─────────────────────────────────────────────────────────
def test_parse_plain_string_content():
    p = _parse_chunk(_chunk(content="hello"))
    assert p.content == "hello"
    assert p.thinking == ""
    assert p.tool_calls == []
    assert p.done is False


def test_parse_list_of_parts_content():
    parts = [{"type": "text", "text": "a"},
             {"type": "text", "text": "b"},
             {"type": "tool_call", "text": "ignore-me"}]
    p = _parse_chunk(_chunk(content=parts))
    assert p.content == "a b"  # non-text parts dropped


def test_parse_thinking_and_usage():
    p = _parse_chunk(_chunk(thinking="think", done=True, prompt=11, evalc=22, reason="stop"))
    assert p.thinking == "think"
    assert p.done is True
    assert p.prompt_eval_count == 11
    assert p.eval_count == 22
    assert p.done_reason == "stop"


def test_parse_none_message():
    p = _parse_chunk(NS(message=None, done=True, prompt_eval_count=0,
                        eval_count=0, done_reason=""))
    assert p.content == ""
    assert not p.tool_calls  # None or [] — falsy either way


def test_parse_tool_calls_formatted():
    tcs = [_tool_call("read", '{"p":"a"}', "id-1"), _tool_call("write", "{}", None)]
    p = _parse_chunk(_chunk(tool_calls=tcs))
    assert len(p.tool_calls) == 2
    assert p.tool_calls[0]["id"] == "id-1"
    assert p.tool_calls[0]["function"]["name"] == "read"
    assert p.tool_calls[1]["id"].startswith("call_")  # generated when absent


# ── StreamBatcher ───────────────────────────────────────────────────────
def test_batcher_first_frame_gets_role_then_content():
    b = StreamBatcher(b"pre", b"\n\n")
    b.add(_parse_chunk(_chunk(content="He")))
    b.add(_parse_chunk(_chunk(content="llo")))
    frame = b.flush()
    assert frame is not None
    assert b"role" in frame and b"assistant" in frame
    assert b"He" in frame and b"llo" in frame


def test_batcher_second_frame_has_no_role():
    b = StreamBatcher(b"", b"")
    b.add(_parse_chunk(_chunk(content="first")))
    b.flush()
    b.add(_parse_chunk(_chunk(content="second")))
    frame = b.flush()
    assert b"role" not in frame
    assert b"second" in frame


def test_batcher_flush_empty_returns_none():
    b = StreamBatcher(b"", b"")
    assert b.flush() is None
    assert b.dirty() is False


def test_batcher_accumulates_thinking_and_content():
    b = StreamBatcher(b"", b"")
    b.add(_parse_chunk(_chunk(content="c", thinking="t")))
    frame = b.flush()
    assert b"reasoning_content" in frame
    assert b"content" in frame


def test_batcher_reset_keeps_first_flag():
    b = StreamBatcher(b"", b"")
    b.add(_parse_chunk(_chunk(content="stale")))
    b.reset()
    assert b.dirty() is False
    # first-frame flag still intact -> next flush still carries role
    b.add(_parse_chunk(_chunk(content="fresh")))
    frame = b.flush()
    assert b"role" in frame


def test_format_tool_calls_indices_and_args():
    tcs = [_tool_call("a", '{"x":1}', "i0"), _tool_call("b", '{"y":2}', None)]
    out = _format_tool_calls(tcs, 5)
    assert out[0]["index"] == 5
    assert out[1]["index"] == 6
    assert out[0]["id"] == "i0"
    assert out[1]["id"].startswith("call_")
    assert out[0]["function"]["arguments"] == '{"x":1}'


def test_format_tool_calls_dict_arguments():
    out = _format_tool_calls([_tool_call("f", {"k": 1})], 0)
    assert out[0]["function"]["arguments"] == '{"k":1}'


if __name__ == "__main__":
    import sys
    fns = [v for k, v in list(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"OK — {len(fns)} tests passed")
    sys.exit(0)
